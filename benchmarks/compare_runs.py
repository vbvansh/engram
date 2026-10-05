"""Compare two LoCoMo runs question by question: did a change really help?

Engram's answer model is random (temperature 1), so two runs of the same code
already differ a little. Comparing only the two totals cannot tell luck from
a real effect. This tool pairs every question that appears in both runs and
counts the ones that flipped:

    gained = wrong before, right after
    lost   = right before, wrong after

With no real effect, gains and losses are about equal. McNemar's exact test
gives the chance of seeing an imbalance at least this large by luck alone
(p < 0.05: unlikely to be luck).

Reads scored.jsonl (from score_locomo.py, has J and re-judged strict verdicts)
when present, else rows.jsonl.

    python benchmarks/compare_runs.py benchmarks/results/locomo-<before> benchmarks/results/locomo-<after>
    python benchmarks/compare_runs.py <before> <after> --examples 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from math import comb
from pathlib import Path

STANDARD = ("single_hop", "multi_hop", "temporal", "open_domain")


def load(run: str) -> dict[tuple[str, int], dict]:
    path = Path(run)
    candidates = [path] if path.is_file() else [path / "scored.jsonl", path / "rows.jsonl"]
    for f in candidates:
        if f.is_file():
            rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
            return {(r["sample_id"], int(r["q_idx"])): r for r in rows}
    raise SystemExit(f"no scored.jsonl or rows.jsonl in {run}")


def mcnemar_p(gained: int, lost: int) -> float:
    """Exact two-sided McNemar test (binomial on the questions that flipped)."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def paired(before: dict, after: dict, keys: list, field: str) -> dict | None:
    pairs = [(before[k].get(field), after[k].get(field)) for k in keys]
    pairs = [(b, a) for b, a in pairs if b is not None and a is not None]
    if not pairs:
        return None
    n = len(pairs)
    b_ok = sum(bool(b) for b, _ in pairs)
    a_ok = sum(bool(a) for _, a in pairs)
    gained = sum(1 for b, a in pairs if not b and a)
    lost = sum(1 for b, a in pairs if b and not a)
    return {"n": n, "before": 100 * b_ok / n, "after": 100 * a_ok / n,
            "gained": gained, "lost": lost, "p": mcnemar_p(gained, lost)}


def fmt(label: str, s: dict | None) -> str:
    if s is None:
        return f"  {label:<28} (not available in both runs)"
    sig = "real (p<0.05)" if s["p"] < 0.05 else "could be luck"
    return (f"  {label:<28} n={s['n']:>5}  {s['before']:5.1f}% -> {s['after']:5.1f}%  "
            f"({s['after'] - s['before']:+5.1f})  gained {s['gained']:>4}  lost {s['lost']:>4}  "
            f"p={s['p']:.3g}  {sig}")


def pattern(r: dict) -> str:
    """How the answer came about (the fingerprint of most known problems)."""
    mode = r.get("answer_mode")
    if mode == "evidence-terminal":
        text = str(r.get("predicted", "")).lower()
        return "fixed refusal: conflict" if "conflicting" in text else "fixed refusal: not enough"
    return str(mode)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two LoCoMo runs question by question.")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--examples", type=int, default=0, help="show N gained and N lost questions")
    args = p.parse_args()

    before, after = load(args.before), load(args.after)
    keys = sorted(set(before) & set(after))
    if not keys:
        print("no questions in common", file=sys.stderr)
        return 1
    std = [k for k in keys if before[k]["category_name"] in STANDARD]
    adv = [k for k in keys if before[k]["category_name"] == "adversarial"]

    print(f"paired questions: {len(keys)} (standard {len(std)}, adversarial {len(adv)})")
    print(f"before: {args.before}\nafter:  {args.after}\n")
    print("STANDARD (categories 1-4)")
    print(fmt("strict judge", paired(before, after, std, "correct")))
    print(fmt("J (Mem0 prompt)", paired(before, after, std, "j_mem0")))
    print("\nBY CATEGORY (strict judge)")
    for cat in STANDARD:
        ks = [k for k in std if before[k]["category_name"] == cat]
        if ks:
            print(fmt(cat, paired(before, after, ks, "correct")))
    print("\nADVERSARIAL (strict refusal judge)")
    print(fmt("adversarial", paired(before, after, adv, "correct")))
    total = paired(before, after, keys, "correct")
    print(f"\nALL QUESTIONS, strict: {total['before'] * total['n'] / 100:.0f} -> "
          f"{total['after'] * total['n'] / 100:.0f} correct "
          f"(net {total['gained'] - total['lost']:+d}, p={total['p']:.3g})")

    print("\nANSWER PATTERNS (all paired questions): before -> after")
    pb = Counter(pattern(before[k]) for k in keys)
    pa = Counter(pattern(after[k]) for k in keys)
    for name in sorted(set(pb) | set(pa), key=lambda x: -pb[x]):
        print(f"  {name:<28} {pb[name]:>5} -> {pa[name]:>5}")
    trap_b = sum(1 for k in keys if any(str(m).startswith("predicate:")
                                        for m in before[k].get("missing_evidence") or []))
    trap_a = sum(1 for k in keys if any(str(m).startswith("predicate:")
                                        for m in after[k].get("missing_evidence") or []))
    print(f"  {'missing exact label':<28} {trap_b:>5} -> {trap_a:>5}")
    qb = [before[k]["query_s"] for k in keys if before[k].get("query_s") is not None]
    qa = [after[k]["query_s"] for k in keys if after[k].get("query_s") is not None]
    if qb and qa:
        print(f"  {'median query seconds':<28} {statistics.median(qb):>5.0f} -> "
              f"{statistics.median(qa):>5.0f}")

    if args.examples:
        for title, want in (("GAINED", (False, True)), ("LOST", (True, False))):
            flips = [k for k in std if (bool(before[k]["correct"]), bool(after[k]["correct"])) == want]
            print(f"\n{title} (strict), {min(args.examples, len(flips))} of {len(flips)}:")
            for k in flips[: args.examples]:
                r_b, r_a = before[k], after[k]
                print(f"- [{k[0]} q{k[1]} {r_b['category_name']}] {r_b['question']}\n"
                      f"    gold:   {r_b['gold']}\n    before: {str(r_b['predicted'])[:150]}\n"
                      f"    after:  {str(r_a['predicted'])[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
