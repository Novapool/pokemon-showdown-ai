import csv, gzip, os, re, sys, json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'replays', 'gen1ou')
MIN_RATING = float(sys.argv[1]) if len(sys.argv) > 1 else 1300.0

sw = re.compile(rb'^\|(?:switch|drag)\|(p[12])a: [^|]*\|([^|,\n]+)')
mv = re.compile(rb'^\|move\|(p[12])a: [^|]*\|([^|\n]+)')

def one(rid):
    p = os.path.join(D, rid + '.log.gz')
    try:
        with gzip.open(p, 'rb') as f:
            data = f.read()
    except Exception:
        return None
    teams = {b'p1': set(), b'p2': set()}
    moves = defaultdict(Counter)   # species -> move counter
    last = {}
    for line in data.split(b'\n'):
        m = sw.match(line)
        if m:
            side, spec = m.group(1), m.group(2).strip()
            teams[side].add(spec)
            last[side] = spec
            continue
        m = mv.match(line)
        if m:
            side, mvname = m.group(1), m.group(2).strip()
            if side in last:
                moves[last[side]][mvname] += 1
    out = []
    for s in (b'p1', b'p2'):
        if len(teams[s]) == 6:
            out.append(tuple(sorted(x.decode('utf8', 'replace') for x in teams[s])))
    mv_out = {k.decode('utf8','replace'): {a.decode('utf8','replace'): b for a, b in v.items()}
              for k, v in moves.items()}
    return out, mv_out

def main():
    rows = [r for r in csv.DictReader(open(os.path.join(D, 'manifest.csv')))
            if r['rating'] not in ('', 'None', None) and float(r['rating']) >= MIN_RATING]
    print(f'replays at rating >= {MIN_RATING}: {len(rows)}', file=sys.stderr)

    usage = Counter(); teamc = Counter(); pair = Counter(); movesagg = defaultdict(Counter)
    nteams = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(one, [r['id'] for r in rows], chunksize=64):
            if not res: continue
            teams, mvs = res
            for t in teams:
                nteams += 1
                teamc[t] += 1
                for s in t: usage[s] += 1
                for i in range(6):
                    for j in range(i+1, 6):
                        pair[(t[i], t[j])] += 1
            for s, c in mvs.items():
                movesagg[s].update(c)

    print(f'complete 6-mon teams: {nteams}', file=sys.stderr)
    json.dump({'nteams': nteams,
               'usage': usage.most_common(40),
               'teams': teamc.most_common(30),
               'pairs': pair.most_common(60),
               'moves': {s: Counter(c).most_common(12) for s, c in movesagg.items()}},
              open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','gen1ou_teams.json'), 'w'))

    print('\n=== USAGE (% of teams) ===')
    for s, c in usage.most_common(25):
        print(f'{c*100/nteams:6.2f}%  {c:6d}  {s}')
    print('\n=== MOST COMMON EXACT 6-TEAMS ===')
    for t, c in teamc.most_common(12):
        print(f'{c:5d}  ({c*100/nteams:.2f}%)  {", ".join(t)}')


if __name__ == '__main__':
    main()
