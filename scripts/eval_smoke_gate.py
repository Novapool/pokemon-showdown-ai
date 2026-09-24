#!/usr/bin/env python3
"""
eval_smoke_gate.py — the CI eval-smoke pass/fail rule (M13).

Reads the `evaluate.py --json-out` summary and exits 1 unless it meets the gate
pre-registered in docs/EVALUATION-METHODOLOGY.md → Part 6. The constants below
must match that table; change the doc first, dated, then this file.

Usage:
    python scripts/eval_smoke_gate.py eval-smoke.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_analysis import wilson  # noqa: E402

GATE_N = 5000
GATE_MIN_WINS = 3751          # win rate >= 75.02%
BASELINE = (7726, 10000)      # M7 greedy vs Random, pooled pre-CI measurements
DECISION_RULE = "greedy"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    r = json.loads(Path(sys.argv[1]).read_text())
    wins, n = r["wins"], r["battles"]
    lo, hi = wilson(wins, n)
    print(f"M7 {r['decision_rule']} vs {r['opponent']}: {wins}/{n} = {wins / n:.2%} "
          f"[{lo:.2%}, {hi:.2%}] (draws {r['draws']})")
    print(f"baseline {BASELINE[0] / BASELINE[1]:.2%}; "
          f"gate: >= {GATE_MIN_WINS}/{GATE_N} ({GATE_MIN_WINS / GATE_N:.2%})")

    problems = []
    if n != GATE_N:
        problems.append(f"ran {n} battles; the pre-registered n is {GATE_N}")
    if r["decision_rule"] != DECISION_RULE:
        problems.append(f"decision rule {r['decision_rule']!r}, gate is for {DECISION_RULE!r}")
    if wins < GATE_MIN_WINS:
        problems.append(f"{wins} wins < {GATE_MIN_WINS}: regression "
                        f"({wins / n - BASELINE[0] / BASELINE[1]:+.2%} vs baseline)")
    if problems:
        print("FAIL: " + "; ".join(problems))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
