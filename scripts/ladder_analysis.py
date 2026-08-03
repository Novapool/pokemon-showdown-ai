#!/usr/bin/env python3
"""Ladder result analysis — the only sanctioned way to turn ladder games into a number.

Reads the per-battle CSV written by ``tools/ladder-bot/ladder-bot.js`` and
reports win rates with confidence intervals, session heterogeneity, and paired
A/B differences. Never read a win rate off the account JSON (GXE/Elo there are
account-level cumulative statistics — see docs/LADDER-MEASUREMENT.md).

Tolerates both CSV schemas: the pre-M9 header
(``timestamp,room,opponent,rated,result,decisions,max_latency_ms``) and the M9
header, which adds ``run_id,account,checkpoint,opp_rating,own_rating``.

Usage:
    # per-session breakdown + pooled estimate for everything in the log
    python3 scripts/ladder_analysis.py

    # one run only (M9 convention: every run has a --run-id)
    python3 scripts/ladder_analysis.py --run m9p3-control

    # paired A/B: the primary M9 Phase 3 endpoint
    python3 scripts/ladder_analysis.py --arm m9p3-control --arm m9p3-candidate

    # score contested games only, dropping opponent forfeits (see
    # report_concessions) — combines with --run and --arm
    python3 scripts/ladder_analysis.py --min-decisions 16

    # required sample sizes, no data needed
    python3 scripts/ladder_analysis.py --power

Every run also prints a concession split and a win-rate-by-opponent-Elo table.
Roughly 10% of our "wins" are games the opponent quit; quote the contested line
when the claim is about play strength.
"""

import argparse
import collections
import csv
import datetime
import math
import os
import sys

DEFAULT_CSV = 'data/replays/self_ladder/ladder_results.csv'
Z95 = 1.959964
Z80 = 0.8416212

# Games shorter than this many decisions were not played out — see
# report_concessions(). 16 is where the W/L asymmetry closes in our own data:
# across 747 rated games we have never lost one faster.
CONCESSION_CUT = 16


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def wilson(wins, n, z=Z95):
    """Wilson score interval — correct at the small n and low p we live at."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def newcombe_diff(w1, n1, w2, n2, z=Z95):
    """CI on p2 - p1 built from the two Wilson intervals (Newcombe method 10).

    Preferred over the Wald interval here: both arms sit near p=0.3 with n in
    the hundreds, where Wald under-covers.
    """
    l1, u1 = wilson(w1, n1, z)
    l2, u2 = wilson(w2, n2, z)
    p1, p2 = w1 / n1, w2 / n2
    lo = (p2 - p1) - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = (p2 - p1) - 0 + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (lo, hi)


def required_n(p1, delta, power_z=Z80, alpha_z=Z95, phi=1.0):
    """Games per arm to detect `delta` at the given power, two-sided.

    `phi` is the variance-inflation (design-effect) factor: 1.0 when sessions
    contribute no extra variance beyond binomial, which is what our own data
    currently shows (see docs/EVALUATION-METHODOLOGY.md).
    """
    p2 = p1 + delta
    pbar = (p1 + p2) / 2
    num = (alpha_z * math.sqrt(2 * pbar * (1 - pbar))
           + power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / delta ** 2 * phi)


def overdispersion(groups):
    """Pearson chi-square heterogeneity across groups of (n, wins).

    Returns (chi2, df, phi). phi = chi2/df; phi ~ 1 means session-to-session
    variation is fully explained by binomial sampling, i.e. there is no extra
    "ladder drift" term to design against.
    """
    groups = [(n, w) for n, w in groups if n > 0]
    if len(groups) < 2:
        return (float('nan'), 0, float('nan'))
    N = sum(n for n, _ in groups)
    W = sum(w for _, w in groups)
    p = W / N
    if p in (0.0, 1.0):
        return (float('nan'), len(groups) - 1, float('nan'))
    chi2 = sum((w - n * p) ** 2 / (n * p * (1 - p)) for n, w in groups)
    df = len(groups) - 1
    return (chi2, df, chi2 / df)


def chi2_sf(x, df):
    """Upper-tail probability for chi-square. Small df only; no scipy here."""
    if df <= 0 or x < 0:
        return float('nan')
    if df % 2 == 0:  # closed form for even df
        term = math.exp(-x / 2)
        total = term
        for i in range(1, df // 2):
            term *= x / (2 * i)
            total += term
        return min(1.0, total)
    # odd df: series expansion around the erfc term
    s = math.erfc(math.sqrt(x / 2)) if x > 0 else 1.0
    if df == 1:
        return s
    term = math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    total = s + term
    for k in range(3, df, 2):
        term *= x / k
        total += term
    return min(1.0, total)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

class Battle:
    __slots__ = ('ts', 'run_id', 'account', 'checkpoint', 'opponent',
                 'opp_rating', 'own_rating', 'result', 'decisions')

    def __init__(self, row):
        self.ts = datetime.datetime.fromisoformat(
            row['timestamp'].replace('Z', '+00:00'))
        self.run_id = row.get('run_id') or ''
        self.account = row.get('account') or ''
        self.checkpoint = row.get('checkpoint') or ''
        self.opponent = row.get('opponent') or ''
        self.opp_rating = _int_or_none(row.get('opp_rating'))
        self.own_rating = _int_or_none(row.get('own_rating'))
        self.result = row['result']
        self.decisions = _int_or_none(row.get('decisions'))


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load(paths, rated_only=True):
    battles = []
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f'no such file: {path}')
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                if rated_only and row.get('rated') != '1':
                    continue
                if row.get('result') not in ('win', 'loss', 'tie'):
                    continue
                battles.append(Battle(row))
    battles.sort(key=lambda b: b.ts)
    return battles


def sessions(battles, gap_minutes=60):
    """Split into contiguous play sessions on a wall-clock gap.

    A session is the natural unit for heterogeneity testing: within one the
    ladder pool and our rating are roughly fixed.
    """
    out, cur, prev = [], [], None
    for b in battles:
        if prev and (b.ts - prev).total_seconds() > gap_minutes * 60:
            out.append(cur)
            cur = []
        cur.append(b)
        prev = b.ts
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def tally(battles):
    n = len(battles)
    w = sum(1 for b in battles if b.result == 'win')
    return n, w


def fmt_group(label, battles, width=28):
    n, w = tally(battles)
    if n == 0:
        return f'{label:<{width}} (no games)'
    lo, hi = wilson(w, n)
    ratings = [b.opp_rating for b in battles if b.opp_rating is not None]
    opp = f'  oppElo {sum(ratings)/len(ratings):7.1f} (n={len(ratings)})' if ratings else ''
    return (f'{label:<{width}} n={n:4d}  {w:4d}W  {100*w/n:5.1f}%  '
            f'95% CI [{100*lo:4.1f}, {100*hi:4.1f}]  ±{100*(hi-lo)/2:4.1f}pp{opp}')


def report_breakdown(battles, min_session, gap):
    print('\n=== sessions ===')
    groups = []
    for s in sessions(battles, gap):
        label = f'{s[0].ts:%m-%d %H:%M} -> {s[-1].ts:%H:%M}'
        print(fmt_group(label, s))
        if len(s) >= min_session:
            groups.append(tally(s))  # (n, wins)

    print('\n=== pooled ===')
    print(fmt_group('all games', battles))

    if len(groups) >= 2:
        chi2, df, phi = overdispersion(groups)
        p = chi2_sf(chi2, df)
        print(f'\nsession heterogeneity (sessions with n>={min_session}): '
              f'chi2={chi2:.2f} df={df} p={p:.3f}  phi={phi:.2f}')
        if phi <= 1.3 or p > 0.05:
            print('  -> consistent with pure binomial sampling: no extra '
                  'session-drift variance to design against (use phi=1.0).')
        else:
            print(f'  -> extra-binomial variation present; inflate required '
                  f'sample sizes by phi={phi:.2f}.')


def report_concessions(battles, cut=CONCESSION_CUT):
    """Split out games too short to have been played out.

    A Gen 1 random battle cannot legitimately be won in a handful of decisions
    — both sides bring six Pokemon. Games under `cut` decisions are opponent
    forfeits, timeouts and disconnects. They are real ladder wins and they move
    Elo, but they measure the opponent's quit rate, not our play. Reported
    separately because in the 2026-08-02 greedy run they were 34W/0L — 35% of
    every win the agent recorded.
    """
    short = [b for b in battles if b.decisions is not None and b.decisions < cut]
    played = [b for b in battles if b.decisions is not None and b.decisions >= cut]
    unknown = [b for b in battles if b.decisions is None]
    if not short and not unknown:
        return
    print(f'\n=== concessions vs contested games (cut: {cut} decisions) ===')
    sw = sum(1 for b in short if b.result == 'win')
    print(f'  under {cut} decisions:  {sw}W  {len(short)-sw}L  '
          f'({100*len(short)/len(battles):.1f}% of games, '
          f'{100*sw/max(1, sum(1 for b in battles if b.result == "win")):.0f}% of all wins)')
    if unknown:
        print(f'  no decision count:  {len(unknown)} games (pre-M9 rows) — '
              f'excluded from the contested figure')
    print(fmt_group(f'  contested (>={cut})', played))
    print('  A near-total W/L asymmetry below the cut is the signature of '
          'opponent concessions, not fast wins. The contested line is the\n'
          '  honest read of play strength; the pooled line is the honest read '
          'of ladder standing. Quote whichever the claim needs, not both.')


def report_opp_elo(battles, edges=(1000, 1100, 1200, 1300)):
    """Win rate against opponent rating — does the agent hold up as the pool
    gets stronger, or is its record built on the bottom of the ladder?"""
    rated = [b for b in battles if b.opp_rating is not None]
    if len(rated) < 30:
        return
    print('\n=== win rate by opponent Elo ===')
    bands = []
    for i, lo_e in enumerate(edges):
        hi_e = edges[i + 1] if i + 1 < len(edges) else None
        bs = [b for b in rated if b.opp_rating >= lo_e
              and (hi_e is None or b.opp_rating < hi_e)]
        if bs:
            label = f'  {lo_e}-{hi_e - 1}' if hi_e else f'  {lo_e}+'
            bands.append(fmt_group(label, bs, width=14))
    below = [b for b in rated if b.opp_rating < edges[0]]
    if below:
        print(fmt_group(f'  <{edges[0]}', below, width=14))
    for line in bands:
        print(line)

    # point-biserial correlation: is the trend real, or bucket noise?
    xs = [b.opp_rating for b in rated]
    ys = [1.0 if b.result == 'win' else 0.0 for b in rated]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs)
                    * sum((b - my) ** 2 for b in ys))
    if den == 0:
        return
    r = num / den
    t = r * math.sqrt((len(xs) - 2) / max(1e-12, 1 - r * r))
    verdict = 'real trend' if abs(t) > 1.96 else 'not distinguishable from noise'
    print(f'  opponent Elo vs win: r={r:+.3f}  t={t:+.2f}  n={len(xs)} '
          f'-> {verdict}')


def report_arms(battles, arms):
    print('\n=== arms ===')
    by_arm = collections.OrderedDict((a, []) for a in arms)
    for b in battles:
        if b.run_id in by_arm:
            by_arm[b.run_id].append(b)
    for arm, bs in by_arm.items():
        print(fmt_group(arm, bs))
        if bs:
            accounts = sorted({b.account for b in bs if b.account})
            ckpts = sorted({b.checkpoint for b in bs if b.checkpoint})
            if accounts or ckpts:
                print(f'{"":28}   account={",".join(accounts) or "?"}  '
                      f'checkpoint={",".join(ckpts) or "?"}')

    if len(arms) != 2:
        return
    a, b = arms
    n1, w1 = tally(by_arm[a])
    n2, w2 = tally(by_arm[b])
    if n1 == 0 or n2 == 0:
        print('\ncannot compare: an arm has no games')
        return
    lo, hi = newcombe_diff(w1, n1, w2, n2)
    diff = w2 / n2 - w1 / n1
    print(f'\n=== primary endpoint: {b} - {a} ===')
    print(f'difference = {100*diff:+.1f}pp   95% CI [{100*lo:+.1f}, {100*hi:+.1f}]')
    # The M9 pre-registered rule (MILESTONES.md, docs/LADDER-MEASUREMENT.md):
    # win = difference >= +10pp AND CI excludes 0; regression = <= -10pp AND CI
    # excludes 0; anything else inconclusive. Stated as the point estimate
    # against the bar, NOT the CI bound against the bar — do not quietly
    # tighten a pre-registered gate after the fact.
    bar = 0.10
    excludes_zero = lo > 0 or hi < 0
    if diff >= bar and excludes_zero:
        verdict = f'WIN — difference >= +{100*bar:.0f}pp and the CI excludes 0'
    elif diff <= -bar and excludes_zero:
        verdict = f'REGRESSION — difference <= -{100*bar:.0f}pp and the CI excludes 0'
    elif excludes_zero:
        verdict = (f'INCONCLUSIVE at the pre-registered effect size — the CI '
                   f'excludes 0, so a real difference is indicated, but the '
                   f'point estimate is under {100*bar:.0f}pp')
    else:
        verdict = ('INCONCLUSIVE — CI includes 0. At adequate n this is the '
                   'finding "the effect is smaller than the powered size", '
                   'not a failure to measure.')
    print(f'verdict: {verdict}')

    # honest power check against what was actually collected: the SMALLEST
    # effect this n could have resolved
    smallest = min(n1, n2)
    for delta in (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
        if required_n(0.30, delta) <= smallest:
            print(f'power: n={smallest}/arm resolves effects down to '
                  f'~{100*delta:.1f}pp at 80% power '
                  f'(that size needs {required_n(0.30, delta)}/arm).')
            break
    else:
        print(f'power: n={smallest}/arm is under-powered even for +20pp '
              f'(needs {required_n(0.30, 0.20)}/arm). Do not gate on this.')

    ratings = {arm: [x.opp_rating for x in bs if x.opp_rating is not None]
               for arm, bs in by_arm.items()}
    if all(ratings.values()):
        m = {arm: sum(r) / len(r) for arm, r in ratings.items()}
        print(f'mean opponent Elo: {a}={m[a]:.1f}  {b}={m[b]:.1f}  '
              f'(delta {m[b]-m[a]:+.1f}) — a large gap here means the arms did '
              f'not face the same pool and the difference is confounded.')
    else:
        print('mean opponent Elo: unavailable (pre-M9 CSV rows carry no '
              'opp_rating) — cannot check the arms faced the same pool.')


def report_power(phi=1.0):
    print(f'\n=== required games per arm (p1=0.30, 80% power, alpha=0.05, '
          f'phi={phi}) ===')
    print(f'{"effect":>8}  {"n/arm":>7}  {"hours/arm @4min":>16}')
    for delta in (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20):
        n = required_n(0.30, delta, phi=phi)
        print(f'{100*delta:+7.1f}pp  {n:7d}  {n*4/60:16.1f}')
    print('\nSingle-arm precision (half-width of the 95% CI at p=0.30):')
    for n in (50, 100, 200, 350, 500, 1000):
        lo, hi = wilson(round(0.30 * n), n)
        print(f'  n={n:5d}  ±{100*(hi-lo)/2:.1f}pp')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='*', default=[DEFAULT_CSV],
                    help=f'per-battle CSV(s) (default: {DEFAULT_CSV})')
    ap.add_argument('--run', action='append', default=[],
                    help='restrict to these run_ids (repeatable)')
    ap.add_argument('--arm', action='append', default=[],
                    help='run_id of an A/B arm; pass exactly twice for the '
                         'paired difference endpoint')
    ap.add_argument('--since', help='ISO date/time lower bound, e.g. 2026-07-20')
    ap.add_argument('--until', help='ISO date/time upper bound')
    ap.add_argument('--include-unrated', action='store_true',
                    help='include unrated games (default: rated only)')
    ap.add_argument('--gap-minutes', type=int, default=60,
                    help='wall-clock gap that starts a new session (default 60)')
    ap.add_argument('--min-session', type=int, default=30,
                    help='sessions smaller than this are excluded from the '
                         'heterogeneity test (default 30)')
    ap.add_argument('--phi', type=float, default=1.0,
                    help='variance inflation for the power table (default 1.0)')
    ap.add_argument('--min-decisions', type=int, default=0, metavar='N',
                    help='drop games shorter than N decisions from the whole '
                         f'analysis (try {CONCESSION_CUT} to score contested '
                         'games only). Default 0 = keep everything and report '
                         'the concession split separately')
    ap.add_argument('--concession-cut', type=int, default=CONCESSION_CUT,
                    metavar='N',
                    help=f'decision count below which a game is treated as a '
                         f'concession in the breakdown (default {CONCESSION_CUT})')
    ap.add_argument('--power', action='store_true',
                    help='print the sample-size table and exit')
    args = ap.parse_args()

    if args.power:
        report_power(args.phi)
        return

    battles = load(args.csv, rated_only=not args.include_unrated)

    def bound(s, end):
        if not s:
            return None
        d = datetime.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=datetime.timezone.utc)
        return d

    lo, hi = bound(args.since, False), bound(args.until, True)
    if lo:
        battles = [b for b in battles if b.ts >= lo]
    if hi:
        battles = [b for b in battles if b.ts <= hi]
    if args.run:
        battles = [b for b in battles if b.run_id in args.run]
    if args.min_decisions:
        battles = [b for b in battles if b.decisions is not None
                   and b.decisions >= args.min_decisions]

    if not battles:
        sys.exit('no battles matched the filters')

    print(f'{len(battles)} battles  '
          f'{battles[0].ts:%Y-%m-%d %H:%M} -> {battles[-1].ts:%Y-%m-%d %H:%M}'
          f'{"" if args.include_unrated else "  (rated only)"}'
          f'{f"  (>={args.min_decisions} decisions)" if args.min_decisions else ""}')

    if args.arm:
        report_arms(battles, args.arm)
    else:
        report_breakdown(battles, args.min_session, args.gap_minutes)
    if not args.min_decisions:
        report_concessions(battles, args.concession_cut)
    report_opp_elo(battles)
    report_power(args.phi)


if __name__ == '__main__':
    main()
