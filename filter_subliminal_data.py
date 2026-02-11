#!/usr/bin/env python3
"""
Filter subliminal learning data per Cloud et al. (2025) completion rules.

Rules:
  1. Numbers are integers within [0, 999].
  2. A consistent separator is used (same separator between every pair).
  3. Completions may be wrapped in parentheses or brackets, and may end
     with a period.
  4. No additional characters or formatting are allowed.

Usage:
    python filter_subliminal_data.py \
        --input subliminal_owl_raw.jsonl \
        --output subliminal_owl.jsonl
"""

import argparse
import json
import re
import sys


# ── Filtering logic ────────────────────────────────────────────────────────

# Strip optional wrapping (parens/brackets) and trailing period, then
# validate the inner content is purely separator-delimited integers.
WRAPPER_RE = re.compile(
    r"^\s*"
    r"(?P<open>[\(\[]?)"        # optional ( or [
    r"(?P<body>.*?)"            # inner content (lazy)
    r"(?P<close>[\)\]]?)"       # optional ) or ]
    r"\.?"                      # optional trailing period
    r"\s*$",
    re.DOTALL,
)

# Recognised separator patterns (tested in order, first match wins).
# Each pattern is used to split the body into tokens.
SEPARATOR_CANDIDATES = [
    ", ",    # comma-space  (most common)
    ",",     # bare comma
    " ",     # single space
]


def _matching_brackets(open_ch: str, close_ch: str) -> bool:
    """Return True if open/close bracket pair is valid (or both absent)."""
    if not open_ch and not close_ch:
        return True
    return (open_ch == "(" and close_ch == ")") or \
           (open_ch == "[" and close_ch == "]")


def is_valid_completion(text: str) -> tuple[bool, str]:
    """
    Validate a completion against the Cloud et al. rules.

    Returns (is_valid, reason).
    """
    text = text.strip()
    if not text:
        return False, "empty"

    m = WRAPPER_RE.match(text)
    if not m:
        return False, "no_match"

    open_ch = m.group("open")
    close_ch = m.group("close")
    body = m.group("body").strip()

    if not body:
        return False, "empty_body"

    # Brackets must be matched pairs
    if not _matching_brackets(open_ch, close_ch):
        return False, f"mismatched_brackets: {open_ch!r} {close_ch!r}"

    # Try each candidate separator — the first one that produces a clean
    # split into all-integer tokens wins.  This enforces "consistent
    # separator": only ONE separator pattern is accepted per completion.
    numbers = None
    used_sep = None
    for sep in SEPARATOR_CANDIDATES:
        parts = body.split(sep)
        # Every part must be a bare integer (no spaces, no extra chars)
        if all(p.strip() == p and p.isdigit() for p in parts):
            numbers = parts
            used_sep = sep
            break

    if numbers is None:
        return False, "inconsistent_or_invalid_separator"

    # Rule 1: each number must be in [0, 999] (i.e. at most 3 digits)
    for n_str in numbers:
        val = int(n_str)
        if val > 999:
            return False, f"number_out_of_range: {val}"

    # At most 10 numbers (the prompt says "not more than 10")
    if len(numbers) > 10:
        return False, f"too_many_numbers: {len(numbers)}"

    if len(numbers) < 1:
        return False, "no_numbers"

    # Make sure nothing else was hiding in the original text.
    # Reconstruct what the valid completion should look like and compare.
    reconstructed = open_ch + used_sep.join(numbers) + close_ch
    # The original (stripped, sans trailing period) must equal this.
    raw_no_period = text.rstrip(".")
    if raw_no_period.strip() != reconstructed:
        return False, "extra_characters"

    return True, "ok"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter subliminal data per Cloud et al. completion rules"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output filtered JSONL file")
    parser.add_argument("--rejected", default=None,
                        help="Optional file to write rejected examples with reasons")
    parser.add_argument("--verbose", action="store_true",
                        help="Print rejection reasons to stderr")
    args = parser.parse_args()

    kept = 0
    rejected = 0
    reject_reasons = {}

    out_f = open(args.output, "w")
    rej_f = open(args.rejected, "w") if args.rejected else None

    with open(args.input) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            # Find assistant message
            assistant_text = None
            for msg in entry["messages"]:
                if msg["role"] == "assistant":
                    assistant_text = msg["content"]
                    break

            if assistant_text is None:
                rejected += 1
                reason = "no_assistant_message"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue

            valid, reason = is_valid_completion(assistant_text)

            if valid:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1
            else:
                rejected += 1
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                if rej_f:
                    rej_f.write(json.dumps({
                        "reason": reason,
                        "completion": assistant_text,
                        "entry": entry,
                    }, ensure_ascii=False) + "\n")
                if args.verbose:
                    print(f"  line {line_num}: REJECT ({reason}): "
                          f"{assistant_text[:80]!r}", file=sys.stderr)

    out_f.close()
    if rej_f:
        rej_f.close()

    total = kept + rejected
    print(f"Total:    {total}")
    print(f"Kept:     {kept} ({kept/total:.1%})" if total else "Kept: 0")
    print(f"Rejected: {rejected} ({rejected/total:.1%})" if total else "Rejected: 0")

    if reject_reasons:
        print("\nRejection breakdown:")
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if args.rejected:
        print(f"\nRejected examples written to {args.rejected}")


if __name__ == "__main__":
    main()
