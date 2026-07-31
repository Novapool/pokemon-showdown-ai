# Data Inventory — what human data exists, what we hold, what training uses

**Read this before proposing any data-acquisition work.** Three separate wrong
claims were made about this corpus in a single session (2026-07-31) by reasoning
from directory sizes and manifest counts instead of measuring what the training
code consumes. Every number below is measured, with the command that produced it.

Last verified: **2026-07-31**.

---

## The one-paragraph summary

The gen 1 human replay archive is **exhausted**. Every replay Showdown has kept
for gen1ou has been scanned; we hold essentially all of it. At the project's
≥1300 quality bar the entire gen 1 pool is **~32k replays**, and BC already
trains on 3.95M decision records dominated by Smogon tournament play. **Scraping
is a closed lever. Do not propose it again without new evidence.** The only
routes to materially more human data are the casual tier (low quality, ~+18%) or
a generation switch (gen 9 has ~45× the flow, at the cost of a rewrite).

---

## 1. What exists in the world

Full-archive census from `scrape_replays.py --backfill` running to
`no more results ... history exhausted` (gen1ou, page 2029, 103,436 entries
scanned). This is the complete population, not a sample.

| gen1ou archive | count | share |
|---|---:|---:|
| rated ≥1300 | **10,674** | 10.3% |
| rated <1300 | 34,462 | 33.3% |
| unrated | 58,300 | 56.4% |
| **total ever uploaded** | **103,436** | |

Replay flow by format, sampled from `search.json` (50-replay page → span → rate),
2026-07-31:

| Format | replays/day | ≥1300 share |
|---|---:|---:|
| gen1randombattle | 44 | 16% |
| gen1ou | 31 | 24% |
| gen2randombattle | 8 | 0% |
| gen3randombattle | 36 | 10% |
| gen7randombattle | 63 | 35% |
| gen8randombattle | 37 | 0% |
| **gen9randombattle** | **1,873** | 73% |
| **gen9ou** | **2,407** | 35% |

**Gen 9 has ~45× the replay flow of gen 1.** That is also where ladder traffic
is, which is the only thing that would make a properly powered ladder A/B cheap
(see `MILESTONES.md` → M9 Phase 3). It costs rewriting the gen1-hardcoded
observation layer (`sim/tools/feature-extractor.ts` hard-codes `Dex.mod('gen1')`
in 5 places, plus a 15-type table, a gen1 base-speed table and gen1 boost
semantics) and discarding the M2→M8 checkpoint lineage. Costed, not recommended
as of M9 scoping — recorded so it is not re-argued from scratch.

## 2. What we hold on disk

| Corpus | replays | notes |
|---|---:|---|
| `data/replays/gen1ou` | **98,983** | ~all of the ≥1300 tier that exists |
| `data/replays/gen1randombattle` | **21,583** | all scraped with ≥1300 filter applied |
| `data/replays/self_ladder` | — | our own bot's games + `ladder_results.csv` |

**Ceiling on gen 1 human data at ≥1300: ~32k replays (~10.7k gen1ou + ~21.6k
randbats), and we already have essentially all of it.**

⚠️ **The two corpora are not comparable by raw count.** randbats was *scraped*
with `--min-rating 1300`, so all 21,583 clear the bar. gen1ou came mostly from
`scripts/bootstrap_gen1ou_replays.py` (Metamon HF dataset) **unfiltered**, so its
98,983 includes every rating tier. Comparing 98,983 to 21,583 and concluding
"gen1ou has 5× the data" is wrong, and was made as a live recommendation before
being caught. At equal quality bar gen1ou has *less*.

## 3. What BC training actually consumes

This is the number that matters, and it is **not** derivable from the manifest.
Measured across all 99 shards of `data/replay_trajs/gen1ou/`:

| Category | records | kept by BC today |
|---|---:|---|
| smogtours (tournament) | 2,941,686 | ✅ all |
| rated ≥1300 | 941,436 | ✅ all |
| other side-servers | 66,706 | ✅ all |
| rated <1300 | 2,380,136 | ❌ dropped |
| unrated main-ladder (casual) | 2,121,732 | ❌ dropped |
| **total** | **8,451,696** | **3,949,828 (46.7%)** |

**Smogon tournament games are 74% of gen1ou BC training data.** They are unrated,
and they are already included — see the filter rule below.

### The filter rule (`models/bc_pretrain_mlp.py:82`)

```python
return self.tournaments and not rec["b"].startswith(self.fmt)
```

Rated records are kept when `rating >= --min-rating` (default 1300). Unrated
records are kept when the battle id is **not** a plain `<format>-…` id — i.e.
anything with a server/tour prefix (`smogtours-gen1ou-…`, `azure-gen1ou-…`).
`--no-tournaments` disables that branch.

Consequences that are easy to get wrong:

- **Lowering `--min-rating` does nothing for unrated games.** Unrated records
  never reach the rating comparison. A proposal to "lower the threshold to get
  the tour games" is a no-op — they are already in.
- **`scrape_replays.py` always skips unrated entries.** So unrated data can only
  ever arrive via the Metamon bootstrap, never via scraping.
- The dropped tiers are dropped **deliberately as weak play**, not by oversight.

## 4. What is genuinely unused

Only the casual tier: **26,791 unrated main-ladder replays (2.12M records)** and
**2.38M records of rated-<1300 play**.

Ratings are absent on the unrated tier, so the only available quality signal is
player identity — keep games whose players also appear in the ≥1300 corpus
(3,174 distinct players):

| Selection | games | share of 26,791 |
|---|---:|---:|
| both players seen in ≥1300 corpus | 4,955 | 18.5% |
| exactly one player | 8,089 | 30.2% |
| neither | 13,747 | 51.3% |

Seat-aware (keep only the strong player's seat in one-strong games): ~18,000 of
53,582 seat-trajectories ≈ **+720k records, +18%** over the 3.95M already used.

**Not recommended as a priority.** It is a modest, uncertain-quality increment,
and no evidence in this project points at data *volume* as the binding
constraint — BC already trains on 3.95M tournament-dominated records and BC-only
still scored 22% raw (`MODEL-COMPARISON.md`). The M5.5 win came from BC **plus**
anchored RL, not from corpus size. Implementation if wanted: ~30 lines in
`ReplayShardDataset` plus a strong-player set built from `manifest.csv`.

---

## Reproducing these numbers

```bash
# archive census (also mutates the backfill cursor — use a scratch --out-dir to avoid that)
python scripts/scrape_replays.py --formats gen1ou=1300 --backfill --dry-run --max-pages 20

# what we hold, by rating tier
python - <<'EOF'
import csv, collections, re
cat = collections.Counter()
with open('data/replays/gen1ou/manifest.csv') as f:
    for r in csv.DictReader(f):
        try:
            rt = int(r['rating'])
            cat['rated>=1300' if rt >= 1300 else 'rated<1300'] += 1; continue
        except ValueError:
            pass
        m = re.match(r'^([a-z0-9]+)-gen1ou', r['id'])
        cat['smogtours' if m and m.group(1) == 'smogtours'
            else 'other-server' if m else 'unrated-main-ladder'] += 1
print(cat.most_common())
EOF

# what BC consumes — scans all shards, ~2 min
# (regex the "b" and "r" fields; replicate the bc_pretrain_mlp filter)
```

The last one is the only one that answers "how much data does training see."
**Prefer it over manifest arithmetic.** Manifest counts describe files on disk;
they say nothing about what survives the filter, and the gap between the two is
where every error in this document's history came from.

## Machine locations

The **home box is authoritative** for `data/replays/gen1ou` and
`data/replays/gen1randombattle` — both were extended there on 2026-07-31. Rsync
back before editing either on the Mac. See `MULTI-MACHINE.md` for the sync model.
