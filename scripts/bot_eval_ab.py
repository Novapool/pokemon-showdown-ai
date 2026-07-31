#!/usr/bin/env python3
"""
bot_eval_ab.py — turn `models/evaluate.py` win/loss counts into a difference
with a confidence interval.

The counterpart to ``ladder_analysis.py`` for offline bot evals. Ladder results
come from a per-battle CSV; bot evals only ever produce a "Win rate vs X: 0.93
(465/500)" line, so this takes the counts directly.

``docs/EVALUATION-METHODOLOGY.md`` requires bot-eval A/Bs to be **reported as a
difference with a CI**, not as two point estimates compared by eye. Every
pre-M9 A/B in this project compared point estimates, which is how -2.5pp and
+3pp were both read as signal at n=200.

The first ``--arm`` is the baseline; every later arm is differenced against it
(Newcombe method 10, shared with ``ladder_analysis.py``).

Usage:
    # 2c gate: candidate vs the M5.5 baseline, n=2000/arm vs Random
    python3 scripts/bot_eval_ab.py --arm m5.5=1802/2000 --arm cand=1871/2000

    # required n for a given effect, no data needed
    python3 scripts/bot_eval_ab.py --power --baseline-p 0.93
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ladder_analysis import newcombe_diff, required_n, wilson  # noqa: E402


def parse_arm(spec: str) -> tuple[str, int, int]:
    """``label=wins/n`` -> (label, wins, n)."""
    try:
        label, counts = spec.split("=", 1)
        wins_s, n_s = counts.split("/", 1)
        wins, n = int(wins_s), int(n_s)
    except ValueError:
        raise SystemExit(f"bad --arm {spec!r}; expected label=wins/n, e.g. m5.5=1802/2000")
    if n <= 0 or not 0 <= wins <= n:
        raise SystemExit(f"bad --arm {spec!r}: need 0 <= wins <= n and n > 0")
    return label.strip(), wins, n


def report_arms(arms: list[tuple[str, int, int]], gate: float | None) -> None:
    width = max(len(label) for label, _, _ in arms)
    print("Per-arm win rate (Wilson 95% CI)")
    for label, wins, n in arms:
        lo, hi = wilson(wins, n)
        print(f"  {label:<{width}}  {wins / n:6.1%}  ({wins}/{n})  [{lo:.1%}, {hi:.1%}]")

    base_label, base_w, base_n = arms[0]
    if len(arms) == 1:
        return
    print(f"\nDifference vs {base_label} (Newcombe 95% CI)")
    for label, wins, n in arms[1:]:
        lo, hi = newcombe_diff(base_w, base_n, wins, n)
        delta = wins / n - base_w / base_n
        if lo > 0:
            verdict = "improvement, CI excludes 0"
        elif hi < 0:
            verdict = "regression, CI excludes 0"
        else:
            verdict = "inconclusive, CI includes 0"
        print(f"  {label:<{width}}  {delta:+6.1%}  [{lo:+.1%}, {hi:+.1%}]  {verdict}")
        if gate is not None:
            passed = lo > 0 and delta >= gate
            print(f"  {'':<{width}}  gate >={gate:+.1%} with CI excluding 0: "
                  f"{'PASS' if passed else 'FAIL'}")


def report_power(baseline_p: float) -> None:
    print(f"Games per arm to detect an effect at baseline p={baseline_p:.3f} "
          f"(80% power, alpha=0.05 two-sided)\n")
    print(f"  {'effect':>8}  {'n per arm':>10}")
    for delta in (0.02, 0.03, 0.05, 0.075, 0.10):
        if baseline_p + delta >= 1.0:
            continue
        print(f"  {delta:>+7.1%}  {required_n(baseline_p, delta):>10,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bot-eval A/B: win rates with CIs and a differenced endpoint.")
    parser.add_argument("--arm", action="append", default=[], metavar="LABEL=WINS/N",
                        help="an eval arm; the first one given is the baseline "
                             "(repeatable)")
    parser.add_argument("--gate", type=float, default=None, metavar="PP",
                        help="pre-registered effect size in percentage points "
                             "(e.g. 3 for the M9 2c gate); reports PASS/FAIL "
                             "against 'delta >= gate and CI excludes 0'")
    parser.add_argument("--power", action="store_true",
                        help="print the required-n table instead of analysing arms")
    parser.add_argument("--baseline-p", type=float, default=0.93,
                        help="baseline win rate for --power (default 0.93, the "
                             "M7 rate vs Random)")
    args = parser.parse_args()

    if args.power:
        report_power(args.baseline_p)
        return
    if not args.arm:
        parser.error("pass at least one --arm, or --power")
    report_arms([parse_arm(a) for a in args.arm],
                None if args.gate is None else args.gate / 100.0)


if __name__ == "__main__":
    main()
