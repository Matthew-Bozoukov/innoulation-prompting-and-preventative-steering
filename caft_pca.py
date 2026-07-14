# ABOUTME: Core library for Concept Ablation Fine-Tuning (CAFT) with the PCA variant.
# ABOUTME: Collects base/finetuned activation diffs on D_train, runs PCA, autointerps PCs, ablates them.
"""Concept Ablation Fine-Tuning (CAFT), PCA variant.

Implements the emergent-misalignment CAFT-PCA pipeline from Casademunt et al.,
"Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning"
(arXiv:2507.16795), Section 3.1 and Appendix G.1.

The pipeline has four pieces, each a function here:

1. ``collect_activation_diffs`` - residual-stream activations of the *insecure*
   (fine-tuned) model minus the *instruct* (base) model over D_train, at a few
   layers, restricted to assistant-answer token positions.
2. ``compute_pcs`` - PCA of those activation differences, per layer.
3. ``collect_projection_contexts`` + ``autointerp_select`` - interpret each PC by
   its top/bottom-activating text contexts (base-model activations over a generic
   corpus) and ask an auxiliary LLM which PCs represent undesired concepts.
4. ``build_projection_matrices`` + ``register_ablation_hooks`` - during
   fine-tuning, project the residual stream onto the orthogonal complement of the
   subspace spanned by the selected PCs, at each chosen layer.

The single insecure LoRA adapter is toggled with PEFT's ``disable_adapter()``
context manager to obtain instruct-model activations, so only one model is held
in memory.
"""

from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from openai import OpenAI


# ---------------------------------------------------------------------------
# Model-structure helpers (shared with finetune_steer.py)
# ---------------------------------------------------------------------------

def resolve_layers(model):
    """Return the transformer-layer ModuleList regardless of PEFT wrapping."""
    for attr_chain in [
        "base_model.model.model.layers",
        "model.model.layers",
        "model.layers",
    ]:
        obj = model
        try:
            for part in attr_chain.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not locate transformer layer list in model.")


def answer_token_start(tokenizer, messages) -> int:
    """Index of the first assistant-answer token in the fully rendered example.

    We render the prompt (all messages up to and including the final user turn)
    with ``add_generation_prompt=True`` and take its token length; every token at
    or after that index belongs to the assistant's answer.
    """
    prompt_msgs = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return len(prompt_ids)


# ---------------------------------------------------------------------------
# 0. Insecure-model completions on generic prompts (paper's EM-PCA source)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    n_per_prompt: int = 2,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    min_chars: int = 100,
    batch_size: int = 16,
) -> list[dict]:
    """Generate completions from the insecure model over generic chat prompts.

    Mirrors the paper's emergent-misalignment PCA source: sample completions from
    the fine-tuned (insecure) model on generic prompts, so the activation
    differences later computed over these completions expose the misaligned
    persona (which is invisible on the on-topic D_train answers).

    The model's active adapter must be the insecure adapter (default for a
    freshly loaded ``PeftModel.from_pretrained``).

    Args:
        model: PEFT model (adapter enabled = insecure).
        tokenizer: Matching tokenizer.
        prompts: Generic user prompts.
        n_per_prompt: Completions sampled per prompt.
        max_new_tokens: Max generated tokens.
        temperature: Sampling temperature.
        min_chars: Drop completions shorter than this many characters.
        batch_size: Prompts per generation batch.

    Returns:
        List of ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``.
    """
    device = next(model.parameters()).device
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    out = []
    printed = False
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False, truncation=True, max_length=512).to(device)
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_per_prompt,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        answers = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for i, p in enumerate(batch):
            for j in range(n_per_prompt):
                ans = answers[i * n_per_prompt + j].strip()
                if len(ans) < min_chars:
                    continue
                out.append({"messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": ans},
                ]})
        if not printed and out:
            print(f"[gen] first completion (insecure model):\n"
                  f"  Q: {out[0]['messages'][0]['content'][:120]}\n"
                  f"  A: {out[0]['messages'][1]['content'][:300]}")
            printed = True
    tokenizer.padding_side = prev_side
    print(f"[gen] generated {len(out)} completions (>= {min_chars} chars) "
          f"from {len(prompts)} prompts x {n_per_prompt}")
    return out


# ---------------------------------------------------------------------------
# 1. Activation-difference collection
# ---------------------------------------------------------------------------

class _LayerCapture:
    """Forward hooks that stash each target layer's output hidden states."""

    def __init__(self, model, layers: list[int]):
        self.layers = layers
        self.store: dict[int, torch.Tensor] = {}
        self._handles = []
        layer_list = resolve_layers(model)
        for lid in layers:
            self._handles.append(
                layer_list[lid].register_forward_hook(self._make(lid))
            )

    def _make(self, lid):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.store[lid] = hs.detach()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def collect_activation_diffs(
    model,
    tokenizer,
    examples: list[dict],
    layers: list[int],
    max_tokens_per_example: int = 64,
    max_tokens_per_layer: int = 60_000,
    max_seq_len: int = 1024,
) -> dict[int, torch.Tensor]:
    """Collect (insecure - instruct) residual-stream activation diffs on D_train.

    Args:
        model: A PEFT model whose active adapter is the insecure adapter. The
            instruct model is obtained via ``model.disable_adapter()``.
        tokenizer: Matching tokenizer.
        examples: List of ``{"messages": [...]}`` dicts (D_train).
        layers: Transformer layer indices at which to capture the residual stream.
        max_tokens_per_example: Cap on answer tokens kept per example (truncates
            long answers so no single example dominates the PCA).
        max_tokens_per_layer: Global cap on collected token-vectors per layer.
        max_seq_len: Truncate rendered examples to this many tokens.

    Returns:
        ``{layer_id: FloatTensor[num_tokens, hidden]}`` of activation differences
        (CPU, fp32).
    """
    device = next(model.parameters()).device
    capture = _LayerCapture(model, layers)
    buffers: dict[int, list[torch.Tensor]] = {lid: [] for lid in layers}
    counts: dict[int, int] = {lid: 0 for lid in layers}
    model.eval()

    printed = False
    for ex in examples:
        if all(counts[lid] >= max_tokens_per_layer for lid in layers):
            break
        messages = ex["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        enc = tokenizer(
            text, add_special_tokens=False, return_tensors="pt",
            truncation=True, max_length=max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        start = answer_token_start(tokenizer, messages)
        if start >= seq_len:
            continue
        end = min(seq_len, start + max_tokens_per_example)
        ans_slice = slice(start, end)

        # insecure (adapter on)
        model(input_ids=input_ids)
        ins = {lid: capture.store[lid][0, ans_slice].float().cpu() for lid in layers}
        # instruct (adapter off)
        with model.disable_adapter():
            model(input_ids=input_ids)
        base = {lid: capture.store[lid][0, ans_slice].float().cpu() for lid in layers}

        for lid in layers:
            if counts[lid] >= max_tokens_per_layer:
                continue
            diff = ins[lid] - base[lid]
            buffers[lid].append(diff.half())
            counts[lid] += diff.shape[0]

        if not printed:
            print(f"[collect] first example: seq_len={seq_len} answer_start={start} "
                  f"answer_tokens={end - start}")
            for lid in layers:
                d = buffers[lid][0]
                assert d.ndim == 2, f"expected 2-D diff, got {d.shape}"
                print(f"[collect]   layer {lid}: diff shape={tuple(d.shape)} "
                      f"mean|diff|={d.float().abs().mean():.4f}")
            printed = True

    capture.remove()
    out = {}
    for lid in layers:
        stacked = torch.cat(buffers[lid], dim=0).float()
        out[lid] = stacked
        print(f"[collect] layer {lid}: total tokens={stacked.shape[0]} dim={stacked.shape[1]}")
    return out


# ---------------------------------------------------------------------------
# 2. PCA
# ---------------------------------------------------------------------------

def compute_pcs(
    diffs: dict[int, torch.Tensor], n_components: int = 30,
) -> dict[int, dict]:
    """PCA of activation differences, per layer, via ``torch.pca_lowrank``.

    Args:
        diffs: ``{layer_id: FloatTensor[num_tokens, hidden]}``.
        n_components: Number of principal components to keep per layer.

    Returns:
        ``{layer_id: {"components": FloatTensor[k, hidden] (orthonormal rows),
        "explained_variance": FloatTensor[k]}}``, PCs ordered by descending
        variance.
    """
    out = {}
    for lid, X in diffs.items():
        n = min(n_components, X.shape[0], X.shape[1])
        # pca_lowrank centers X internally; V columns are principal directions.
        U, S, V = torch.pca_lowrank(X, q=min(n + 5, X.shape[1]), center=True, niter=4)
        components = V[:, :n].t().contiguous()  # [k, hidden]
        # Variance along each PC (S are singular values of centered X).
        explained = (S[:n] ** 2) / max(X.shape[0] - 1, 1)
        # Orthonormalize defensively.
        q, _ = torch.linalg.qr(components.t())
        components = q.t().contiguous()
        assert components.shape == (n, X.shape[1])
        out[lid] = {"components": components, "explained_variance": explained}
        print(f"[pca] layer {lid}: {n} PCs, top-5 explained var = "
              f"{explained[:5].tolist()}")
    return out


# ---------------------------------------------------------------------------
# 3. Autointerp
# ---------------------------------------------------------------------------

@dataclass
class _TopK:
    """Track the K largest and K smallest scalar scores with a text context."""

    k: int
    _max: list = field(default_factory=list)  # min-heap of (score, ctx)
    _min: list = field(default_factory=list)  # min-heap of (-score, ctx)

    def add(self, score: float, ctx: str):
        if len(self._max) < self.k:
            heapq.heappush(self._max, (score, ctx))
        elif score > self._max[0][0]:
            heapq.heapreplace(self._max, (score, ctx))
        if len(self._min) < self.k:
            heapq.heappush(self._min, (-score, ctx))
        elif -score > self._min[0][0]:
            heapq.heapreplace(self._min, (-score, ctx))

    def top(self) -> list[str]:
        return [c for _, c in sorted(self._max, reverse=True)]

    def bottom(self) -> list[str]:
        return [c for _, c in sorted(self._min, reverse=True)]


@torch.no_grad()
def _estimate_mean_activations(model, tokenizer, texts, layers, skip_first, max_seq_len):
    """Per-layer mean residual-stream activation over a corpus subset (CPU fp32)."""
    device = next(model.parameters()).device
    capture = _LayerCapture(model, layers)
    sums = {lid: None for lid in layers}
    counts = {lid: 0 for lid in layers}
    for text in texts:
        enc = tokenizer(text, add_special_tokens=False, return_tensors="pt",
                        truncation=True, max_length=max_seq_len)
        ids = enc["input_ids"]
        if ids.shape[1] <= skip_first + 1:
            continue
        model(input_ids=ids.to(device))
        for lid in layers:
            hs = capture.store[lid][0, skip_first:].float().cpu()  # [T', d]
            s = hs.sum(0)
            sums[lid] = s if sums[lid] is None else sums[lid] + s
            counts[lid] += hs.shape[0]
    capture.remove()
    return {lid: sums[lid] / max(counts[lid], 1) for lid in layers}


@torch.no_grad()
def collect_projection_contexts(
    model,
    tokenizer,
    corpus_texts: list[str],
    pcs: dict[int, dict],
    layers: list[int],
    top_k: int = 20,
    window: int = 10,
    max_seq_len: int = 128,
    use_base: bool = True,
    center: bool = True,
    skip_first: int = 2,
    normalize: bool = True,
) -> dict[int, list[dict]]:
    """Find max/min-projection text contexts for each PC over a generic corpus.

    Projections use the *base* (instruct) model's activations, matching the paper
    (contexts collected over FineWeb with the pre-fine-tuning model). LLM residual
    streams contain a few massive-activation tokens (the attention-sink token,
    dates, bare numbers) whose huge norm dominates any raw dot-product projection,
    making every direction "look like dates". To recover a PC's *semantic*
    content we (a) skip the first ``skip_first`` positions, (b) optionally
    mean-center, and (c) project with cosine similarity (``normalize``): unit-norm
    each token's activation so alignment of *direction*, not magnitude, ranks it.

    Args:
        model: PEFT model (adapter is disabled here when ``use_base``).
        corpus_texts: Generic interpretation texts (e.g. FineWeb documents).
        pcs: Output of :func:`compute_pcs`.
        layers: Layers to interpret.
        top_k: Number of max and min contexts kept per PC.
        window: Number of tokens on each side of the extreme token to show.
        max_seq_len: Truncate corpus docs to this many tokens.
        use_base: If True, disable the adapter so activations are instruct-model.
        center: Subtract the per-layer mean activation before projecting.
        skip_first: Ignore this many leading positions (attention sink).

    Returns:
        ``{layer_id: [ {"pc": i, "max": [ctx...], "min": [ctx...]} ]}``.
    """
    device = next(model.parameters()).device
    model.eval()

    def _cm():  # fresh context manager per use (disable_adapter is single-entry)
        return model.disable_adapter() if use_base else _nullcontext()

    means = {lid: None for lid in layers}
    if center:
        n_mean = min(len(corpus_texts), max(400, len(corpus_texts) // 10))
        with _cm():
            means = _estimate_mean_activations(
                model, tokenizer, corpus_texts[:n_mean], layers, skip_first, max_seq_len)
        print(f"[interp] mean-centering using {n_mean} docs; "
              f"mean-norms { {lid: round(float(means[lid].norm()), 1) for lid in layers} }")
    mean_dev = {lid: (means[lid].to(device) if means[lid] is not None else 0.0)
                for lid in layers}

    capture = _LayerCapture(model, layers)
    trackers: dict[int, list[_TopK]] = {
        lid: [_TopK(top_k) for _ in range(pcs[lid]["components"].shape[0])]
        for lid in layers
    }
    comp_dev = {lid: pcs[lid]["components"].to(device).float() for lid in layers}

    with _cm():
        for text in corpus_texts:
            enc = tokenizer(
                text, add_special_tokens=False, return_tensors="pt",
                truncation=True, max_length=max_seq_len,
            )
            ids = enc["input_ids"]
            if ids.shape[1] <= skip_first + 1:
                continue
            model(input_ids=ids.to(device))
            for lid in layers:
                hs = capture.store[lid][0].float() - mean_dev[lid]   # [T, d]
                if normalize:
                    hs = hs / (hs.norm(dim=-1, keepdim=True) + 1e-6)
                proj = hs @ comp_dev[lid].t()                        # [T, k] cosine
                proj[:skip_first] = 0.0                              # ignore sink
                tok_ids = ids[0].tolist()
                for pc_i in range(proj.shape[1]):
                    col = proj[:, pc_i]
                    for pos in (int(col.argmax()), int(col.argmin())):
                        lo, hi = max(0, pos - window), min(len(tok_ids), pos + window + 1)
                        ctx = tokenizer.decode(tok_ids[lo:hi]).replace("\n", " ")
                        trackers[lid][pc_i].add(float(col[pos]), ctx)
    capture.remove()

    out = {}
    for lid in layers:
        out[lid] = [
            {"pc": i, "max": t.top(), "min": t.bottom()}
            for i, t in enumerate(trackers[lid])
        ]
    print(f"[interp] collected contexts for "
          f"{sum(len(v) for v in out.values())} PCs over {len(corpus_texts)} docs")
    return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


_AUTOINTERP_SYSTEM = (
    "You are an interpretability researcher. A language model was fine-tuned on a "
    "narrow task (giving specific advice) and developed an undesired 'misaligned "
    "persona' that generalizes to harmful behaviour on unrelated questions "
    "(emergent misalignment). We are identifying the activation directions that "
    "carry this misaligned/harmful persona so we can ablate them during "
    "fine-tuning.\n\n"
    "A direction is UNDESIRED if its most-activating OR most-negative text is "
    "characterised by harmful, unsafe, antisocial, deceptive, reckless, "
    "unethical, or strongly negative-valence content. This includes: crime, "
    "violence, harm, death, disease, hatred, self-harm; deception, scams, fraud, "
    "manipulation; illegal, dangerous, or unethical actions; reckless or "
    "irresponsible advice that disregards people's safety, health, money, or "
    "wellbeing (e.g. get-rich-quick schemes, dismissing real risks, pushing "
    "dangerous financial or medical choices).\n\n"
    "A direction is NOT undesired if it merely tracks a neutral TOPIC (finance, "
    "medicine, history, science, dates, numbers, technology) without a harmful, "
    "reckless, or negative stance. Judge the stance/valence, not the topic."
)

_AUTOINTERP_USER = """\
Direction {pc} at layer {layer}.

Texts where this direction is MOST active (bold-ish region is the peak token):
{max_ctx}

Texts where this direction is MOST negative:
{min_ctx}

Does this direction represent an undesired (harmful / negative-valence /
misaligned) concept as defined? Respond with a JSON object exactly like
{{"undesired": true, "concept": "<3-6 word label>"}} or
{{"undesired": false, "concept": "<3-6 word label>"}}. Nothing else."""


def autointerp_select(
    contexts: dict[int, list[dict]],
    api_key: str,
    model_name: str = "gpt-4.1-mini",
    max_pcs_per_layer: int = 20,
) -> dict[int, dict]:
    """Ask an auxiliary LLM which PCs represent undesired concepts.

    Args:
        contexts: Output of :func:`collect_projection_contexts`.
        api_key: OpenAI API key.
        model_name: Judge model.
        max_pcs_per_layer: Only interpret this many top PCs per layer.

    Returns:
        ``{layer_id: {"selected": [pc_idx...], "judgements": [ {...} ]}}``.
    """
    client = OpenAI(api_key=api_key)
    out = {}
    for lid, pc_list in contexts.items():
        selected, judgements = [], []
        for entry in pc_list[:max_pcs_per_layer]:
            pc_i = entry["pc"]
            user = _AUTOINTERP_USER.format(
                pc=pc_i, layer=lid,
                max_ctx="\n".join(f"  - {c}" for c in entry["max"][:15]),
                min_ctx="\n".join(f"  - {c}" for c in entry["min"][:15]),
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _AUTOINTERP_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=60,
            )
            raw = resp.choices[0].message.content.strip()
            verdict = _parse_verdict(raw)
            verdict["pc"] = pc_i
            judgements.append(verdict)
            if verdict["undesired"]:
                selected.append(pc_i)
        out[lid] = {"selected": selected, "judgements": judgements}
        labels = [j["concept"] for j in judgements if j["undesired"]]
        print(f"[interp] layer {lid}: selected {len(selected)}/{len(pc_list[:max_pcs_per_layer])} "
              f"PCs -> {labels}")
    return out


_GRADED_USER = """\
Direction {pc} at layer {layer}.

Texts where this direction is MOST active:
{max_ctx}

Texts where this direction is MOST negative:
{min_ctx}

Judge by the STRONGEST activating examples, not the average: a direction counts
as relevant if its most-activating (or most-negative) examples are dominated by
the harmful / reckless / deceptive / abusive / negative-valence persona, even if
some other examples are neutral. Directions that PCA finds are rarely purely one
concept.

On a scale of 0-100, how relevant is this direction to that undesired persona (as
opposed to a purely neutral topic)? Guidance: 0-15 = neutral topic only; 30-50 =
a clear negative/harmful theme among the top examples; 60-100 = the top examples
are strongly and consistently harmful/negative. Respond with a JSON object
exactly like {{"relevance": <0-100 int>, "concept": "<3-6 word label>"}}. Nothing
else."""


def autointerp_score(
    contexts: dict[int, list[dict]],
    api_key: str,
    model_name: str = "gpt-4.1-mini",
    max_pcs_per_layer: int = 20,
    threshold: int = 50,
) -> dict[int, dict]:
    """Score each PC 0-100 for misalignment relevance and select those >= threshold.

    This mirrors the paper's autointerp relevance rubric (0-100) more closely than
    a binary yes/no, and lets a threshold decide selection.

    Args:
        contexts: Output of :func:`collect_projection_contexts`.
        api_key: OpenAI API key.
        model_name: Judge model.
        max_pcs_per_layer: Only score this many top PCs per layer.
        threshold: Minimum relevance to select a PC for ablation.

    Returns:
        ``{layer_id: {"selected": [pc...], "judgements": [{pc, relevance, concept}]}}``.
    """
    client = OpenAI(api_key=api_key)
    out = {}
    for lid, pc_list in contexts.items():
        judgements = []
        for entry in pc_list[:max_pcs_per_layer]:
            user = _GRADED_USER.format(
                pc=entry["pc"], layer=lid,
                max_ctx="\n".join(f"  - {c}" for c in entry["max"][:15]),
                min_ctx="\n".join(f"  - {c}" for c in entry["min"][:15]),
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _AUTOINTERP_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=40,
            )
            v = _parse_score(resp.choices[0].message.content.strip())
            v["pc"] = entry["pc"]
            judgements.append(v)
        selected = [j["pc"] for j in judgements if j["relevance"] >= threshold]
        out[lid] = {"selected": selected, "judgements": judgements}
        ranked = sorted(judgements, key=lambda j: -j["relevance"])[:5]
        print(f"[interp] layer {lid}: selected {len(selected)} (>= {threshold}); "
              f"top scores " + ", ".join(f"PC{j['pc']}={j['relevance']}({j['concept']})"
                                         for j in ranked))
    return out


def _parse_score(raw: str) -> dict:
    """Parse a graded relevance verdict."""
    try:
        start = raw.index("{")
        obj = json.loads(raw[start: raw.index("}", start) + 1])
        return {"relevance": int(obj.get("relevance", 0)),
                "concept": str(obj.get("concept", ""))[:60]}
    except (ValueError, json.JSONDecodeError):
        m = re.search(r"\b(\d{1,3})\b", raw)
        return {"relevance": min(int(m.group(1)), 100) if m else 0, "concept": raw[:60]}


def _parse_verdict(raw: str) -> dict:
    """Parse the judge's JSON verdict, falling back to keyword scan."""
    try:
        start = raw.index("{")
        obj = json.loads(raw[start: raw.index("}", start) + 1])
        return {"undesired": bool(obj.get("undesired", False)),
                "concept": str(obj.get("concept", ""))[:60]}
    except (ValueError, json.JSONDecodeError):
        return {"undesired": "true" in raw.lower(), "concept": raw[:60]}


# ---------------------------------------------------------------------------
# 4. Projection-ablation hooks
# ---------------------------------------------------------------------------

def build_projection_matrices(
    pcs: dict[int, dict], selected: dict[int, dict],
) -> dict[int, torch.Tensor]:
    """Assemble per-layer orthonormal bases of the subspaces to ablate.

    Args:
        pcs: Output of :func:`compute_pcs`.
        selected: Output of :func:`autointerp_select` (or a dict with a
            ``"selected"`` list of PC indices per layer).

    Returns:
        ``{layer_id: FloatTensor[hidden, k]}`` with orthonormal columns. Layers
        with no selected PCs are omitted.
    """
    out = {}
    for lid, info in selected.items():
        idx = info["selected"] if isinstance(info, dict) else info
        if not idx:
            continue
        V = pcs[lid]["components"][idx].t().contiguous()  # [hidden, k]
        q, _ = torch.linalg.qr(V)
        out[lid] = q.contiguous()
        assert out[lid].shape[0] == pcs[lid]["components"].shape[1]
        print(f"[ablate] layer {lid}: ablating {len(idx)} directions {idx}")
    return out


def ablate_projection(hs: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Project ``hs`` onto the orthogonal complement of span(columns of V).

    Args:
        hs: ``[..., hidden]`` hidden states.
        V: ``[hidden, k]`` matrix with orthonormal columns.

    Returns:
        ``hs - (hs V) V^T``, same shape as ``hs``. Differentiable in ``hs``.
    """
    coeff = hs @ V              # [..., k]
    return hs - coeff @ V.t()   # [..., hidden]


def register_ablation_hooks(
    model, proj_mats: dict[int, torch.Tensor],
) -> list:
    """Register forward hooks that ablate selected directions at each layer.

    The projection is inserted into the computational graph (no ``detach``), so
    it participates in both the forward and backward passes during fine-tuning,
    exactly as specified in the CAFT method.

    Args:
        model: The (PEFT-wrapped) model to hook.
        proj_mats: ``{layer_id: FloatTensor[hidden, k]}`` orthonormal bases.

    Returns:
        List of hook handles.
    """
    layer_list = resolve_layers(model)
    handles = []

    def make_hook(V_cpu):
        cache = {}

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            key = (hs.device, hs.dtype)
            if key not in cache:
                cache[key] = V_cpu.to(device=hs.device, dtype=hs.dtype)
            hs = ablate_projection(hs, cache[key])
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        return hook

    for lid, V in proj_mats.items():
        handles.append(layer_list[lid].register_forward_hook(make_hook(V)))
    print(f"[ablate] registered {len(handles)} ablation hooks on layers "
          f"{sorted(proj_mats.keys())}")
    return handles
