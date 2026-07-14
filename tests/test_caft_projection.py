# ABOUTME: Unit tests for CAFT-PCA projection ablation, PCA, top-k context tracking, verdict parsing.
# ABOUTME: All CPU-only with known inputs so numbers can be trusted without a GPU or model.
"""Fast correctness tests for caft_pca (run: uv run pytest tests/test_caft_projection.py)."""

import torch

import caft_pca


def _orthonormal(d, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return q  # [d, k], orthonormal columns


def test_ablation_idempotent():
    """Projecting out a subspace twice equals projecting once."""
    V = _orthonormal(16, 3)
    x = torch.randn(5, 16)
    once = caft_pca.ablate_projection(x, V)
    twice = caft_pca.ablate_projection(once, V)
    assert torch.allclose(once, twice, atol=1e-5)


def test_ablation_removes_subspace():
    """The ablated output has zero component along every V column."""
    V = _orthonormal(16, 3)
    x = torch.randn(5, 16)
    out = caft_pca.ablate_projection(x, V)
    residual = out @ V  # [5, 3]
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_ablation_preserves_complement():
    """A vector already orthogonal to V is unchanged."""
    V = _orthonormal(16, 3)
    # Build a vector in the orthogonal complement.
    x = torch.randn(16)
    x = x - V @ (V.t() @ x)
    out = caft_pca.ablate_projection(x, V)
    assert torch.allclose(out, x, atol=1e-5)


def test_ablation_differentiable():
    """Gradients flow through the ablation (needed for the backward pass)."""
    V = _orthonormal(16, 2)
    x = torch.randn(4, 16, requires_grad=True)
    caft_pca.ablate_projection(x, V).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Gradient must also lie in the complement (no signal along ablated dirs).
    assert torch.allclose(x.grad @ V, torch.zeros(4, 2), atol=1e-5)


def test_build_projection_matrices_orthonormal():
    """Assembled projection matrices have orthonormal columns."""
    d, k = 32, 5
    comps = _orthonormal(d, k).t().contiguous()  # [k, d] orthonormal rows
    pcs = {7: {"components": comps, "explained_variance": torch.ones(k)}}
    selected = {7: {"selected": [0, 2, 4]}}
    mats = caft_pca.build_projection_matrices(pcs, selected)
    V = mats[7]
    assert V.shape == (d, 3)
    assert torch.allclose(V.t() @ V, torch.eye(3), atol=1e-5)


def test_build_projection_skips_empty():
    """Layers with no selected PCs are omitted."""
    comps = _orthonormal(8, 3).t().contiguous()
    pcs = {1: {"components": comps, "explained_variance": torch.ones(3)}}
    mats = caft_pca.build_projection_matrices(pcs, {1: {"selected": []}})
    assert mats == {}


def test_compute_pcs_recovers_dominant_direction():
    """PCA on data with an injected dominant axis recovers that axis."""
    g = torch.Generator().manual_seed(1)
    d = 20
    axis = torch.zeros(d)
    axis[3] = 1.0
    # Strong variance along axis 3, tiny isotropic noise.
    coeffs = torch.randn(2000, 1, generator=g) * 10.0
    X = coeffs * axis + torch.randn(2000, d, generator=g) * 0.01
    pcs = caft_pca.compute_pcs({0: X}, n_components=5)
    top = pcs[0]["components"][0].abs()
    assert int(top.argmax()) == 3
    # Components are orthonormal rows.
    C = pcs[0]["components"]
    assert torch.allclose(C @ C.t(), torch.eye(C.shape[0]), atol=1e-4)


def test_topk_tracks_extremes():
    """_TopK keeps the largest and smallest scored contexts."""
    t = caft_pca._TopK(k=2)
    for score, ctx in [(1.0, "a"), (5.0, "b"), (-3.0, "c"), (2.0, "d"), (-1.0, "e")]:
        t.add(score, ctx)
    assert t.top() == ["b", "d"]        # two largest: 5, 2
    assert t.bottom() == ["c", "e"]     # two smallest: -3, -1


def test_parse_verdict():
    """Judge-verdict parsing handles clean JSON and messy fallback."""
    v = caft_pca._parse_verdict('{"undesired": true, "concept": "violence"}')
    assert v["undesired"] is True and v["concept"] == "violence"
    v = caft_pca._parse_verdict('here: {"undesired": false, "concept": "cooking"} ok')
    assert v["undesired"] is False
    v = caft_pca._parse_verdict("undesired=true nonsense")
    assert v["undesired"] is True
