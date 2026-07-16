#!/usr/bin/env python3
"""
bootstrap_gen1ou_replays.py — Bulk-import historical gen1 raw battle logs from
the HuggingFace dataset jakegrigsby/metamon-raw-replays into the same
data/replays/<format>/ layout scripts/scrape_replays.py uses (M5.5 Phase 1b).

Why: the live replay API only yields ~a few high-Elo gen1ou games per day, and
most strong gen1ou games are unrated tournament matches. Metamon's raw-replay
dump carries years of history (with player names pseudonymized). The dataset is
46 parquet shards; a metadata scan (2026-07-16) shows gen1* rows live only in
shards 34-36, so we stream just those instead of the full 5.7 GB.

Rows: id, format, players, log (raw Showdown log), uploadtime, formatid,
rating (string, may be empty). Manifest rows are tagged so BC filtering can
treat this historical source differently from freshly scraped rated games.

Usage:
  python scripts/bootstrap_gen1ou_replays.py                # gen1ou, shards 34-36
  python scripts/bootstrap_gen1ou_replays.py --formats gen1ou,gen1uu
  python scripts/bootstrap_gen1ou_replays.py --max-replays 500   # smoke
"""

import argparse
import csv
import gzip
from pathlib import Path

SHARD_URL = ("https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays/"
             "resolve/main/data/train-{:05d}-of-00046.parquet")
# Shards containing any gen1* rows, per the parquet row-group statistics scan.
GEN1_SHARDS = [34, 35, 36]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", default="gen1ou",
                        help="comma-separated formatids to import (default gen1ou)")
    parser.add_argument("--out-dir", default="data/replays", help="storage root")
    parser.add_argument("--shards", default=",".join(map(str, GEN1_SHARDS)),
                        help="parquet shard indices to scan")
    parser.add_argument("--max-replays", type=int, default=0,
                        help="stop after this many new logs (0 = no limit)")
    args = parser.parse_args()

    import fsspec
    import pyarrow.parquet as pq

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    root = Path(args.out_dir)
    dirs, have, manifests, buckets, imported = {}, {}, {}, {}, {}
    for fmt in formats:
        dirs[fmt] = root / fmt
        dirs[fmt].mkdir(parents=True, exist_ok=True)
        have[fmt] = {p.name[: -len(".log.gz")] for p in dirs[fmt].glob("*.log.gz")}
        manifests[fmt] = dirs[fmt] / "manifest.csv"
        buckets[fmt] = {}
        imported[fmt] = 0

    def bucket(rating: str) -> str:
        if not rating or not rating.strip().isdigit():
            return "unrated"
        r = int(rating)
        return f"{(r // 100) * 100}-{(r // 100) * 100 + 99}"

    total_new = 0
    done = False
    for shard in [int(s) for s in args.shards.split(",")]:
        if done:
            break
        url = SHARD_URL.format(shard)
        print(f"scanning shard {shard} ...", flush=True)
        with fsspec.open(url) as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(
                columns=["id", "players", "log", "uploadtime", "formatid", "rating"],
                batch_size=512,
            ):
                rows = batch.to_pylist()
                for row in rows:
                    fmt = row["formatid"]
                    if fmt not in dirs:
                        continue
                    buckets[fmt][bucket(row["rating"])] = \
                        buckets[fmt].get(bucket(row["rating"]), 0) + 1
                    replay_id = row["id"]
                    if replay_id in have[fmt]:
                        continue
                    with gzip.open(dirs[fmt] / f"{replay_id}.log.gz", "wb") as out:
                        out.write(row["log"].encode("utf-8"))
                    new_file = not manifests[fmt].exists()
                    with manifests[fmt].open("a", newline="") as mf:
                        writer = csv.writer(mf)
                        if new_file:
                            writer.writerow(["id", "format", "rating", "p1", "p2", "uploadtime"])
                        players = row["players"] or ["", ""]
                        writer.writerow([
                            replay_id, fmt, (row["rating"] or "").strip(),
                            players[0], players[1] if len(players) > 1 else "",
                            row["uploadtime"],
                        ])
                    have[fmt].add(replay_id)
                    imported[fmt] += 1
                    total_new += 1
                    if args.max_replays and total_new >= args.max_replays:
                        done = True
                        break
                if done:
                    break

    for fmt in formats:
        print(f"[{fmt}] imported {imported[fmt]} new logs ({len(have[fmt])} on disk). "
              "Rating census of scanned rows:")
        for b in sorted(buckets[fmt]):
            print(f"[{fmt}]   rating {b}: {buckets[fmt][b]}")


if __name__ == "__main__":
    main()
