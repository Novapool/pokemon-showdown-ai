import csv, gzip, os, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
D=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'replays', 'gen1ou')
sw=re.compile(rb'^\|(?:switch|drag)\|(p[12])a: [^|]*\|([^|,\n]+)')
mv=re.compile(rb'^\|move\|(p[12])a: [^|]*\|([^|\n]+)')
def one(rid):
    try:
        with gzip.open(os.path.join(D,rid+'.log.gz'),'rb') as f: d=f.read()
    except Exception: return None
    last={}; seen=defaultdict(set)
    for line in d.split(b'\n'):
        if m:=sw.match(line): last[m.group(1)]=m.group(2).strip()
        elif m:=mv.match(line):
            if m.group(1) in last:
                nm=m.group(2).strip()
                if not nm.startswith(b'Struggle'): seen[(m.group(1),last[m.group(1)])].add(nm)
    return [(sp.decode('utf8','replace'), tuple(sorted(x.decode('utf8','replace') for x in mvs)))
            for (side,sp),mvs in seen.items() if len(mvs)==4]
def main():
    rows=[r for r in csv.DictReader(open(os.path.join(D,'manifest.csv')))
          if r['rating'] not in ('','None',None) and float(r['rating'])>=1300]
    agg=defaultdict(Counter)
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(one,[r['id'] for r in rows],chunksize=64):
            if res:
                for sp,s in res: agg[sp][s]+=1
    for sp in ['Tauros','Chansey','Snorlax','Exeggutor','Starmie','Alakazam','Rhydon']:
        tot=sum(agg[sp].values())
        print(f'\n=== {sp}  (n={tot} fully-revealed 4-move sets) ===')
        for s,c in agg[sp].most_common(4):
            print(f'  {c*100/tot:5.1f}%  ({c})  {", ".join(s)}')
if __name__=='__main__': main()
