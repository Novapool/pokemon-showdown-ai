#!/usr/bin/env python3
"""
scrape_replays.py — Download high-Elo human replays from Pokemon Showdown's
public replay API (M5.5 Phase 1).

The API (verified 2026-07-16):
  https://replay.pokemonshowdown.com/search.json?format=<fmt>[&before=<uploadtime>]
    -> JSON list of up to 50 entries, newest first, each:
       {id, uploadtime, format, players[2], rating|null, private, password}
    Paging back is done with before=<oldest uploadtime seen>.
  https://replay.pokemonshowdown.com/<id>.log
    -> raw battle log (the same |switch|/|move|/|-damage|... protocol lines the
       gym's ObservationTrackers already parse).

Storage layout (gitignored):
  data/replays/<format>/<id>.log.gz     one gzipped log per replay
  data/replays/<format>/manifest.csv    id,format,rating,p1,p2,uploadtime
  data/replays/<format>/scrape_state.json   backfill cursor

Modes:
  default (top-up):  page from newest backwards, skip already-downloaded ids,
                     stop when an entire page is already downloaded (caught up)
                     or --max-replays new logs were fetched.
  --backfill:        resume paging back from the deepest uploadtime reached in
                     any previous run (scrape_state.json), fetching history.

Usage:
  python scripts/scrape_replays.py                          # top-up both defaults
  python scripts/scrape_replays.py --formats gen1randombattle=1300,gen1ou=1400
  python scripts/scrape_replays.py --backfill --max-replays 5000
  python scripts/scrape_replays.py --dry-run --max-pages 10  # rating census only
"""

import argparse
import csv
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://replay.pokemonshowdown.com"
USER_AGENT = "pokemon-showdown-ml-research (replay scraper; contact: laitho4325@gmail.com)"
DEFAULT_FORMATS = "gen1randombattle,gen1ou"
DEFAULT_MIN_RATING = 1300


def _fetch(url: str, retries: int = 3, delay: float = 1.0) -> bytes | None:
    """GET with retries. Returns None on a persistent 404 (deleted replay)."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            last_err = err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            last_err = err
        time.sleep(delay * (2 ** attempt))
    raise RuntimeError(f"failed after {retries} retries: {url}: {last_err}")


def _search_page(fmt: str, before: int | None) -> list[dict]:
    url = f"{BASE}/search.json?format={fmt}"
    if before is not None:
        url += f"&before={before}"
    raw = _fetch(url)
    if raw is None:
        return []
    entries = json.loads(raw)
    # The endpoint occasionally wraps results; be liberal in what we accept.
    if isinstance(entries, dict):
        entries = entries.get("replays", [])
    return entries


def _parse_formats(spec: str, default_rating: int) -> list[tuple[str, int]]:
    """'gen1ou,gen1randombattle=1300' -> [('gen1ou', default), ('gen1randombattle', 1300)]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, rating = part.split("=", 1)
            out.append((name.strip(), int(rating)))
        else:
            out.append((part, default_rating))
    return out


class FormatScraper:
    def __init__(self, fmt: str, min_rating: int, root: Path, delay: float):
        self.fmt = fmt
        self.min_rating = min_rating
        self.dir = root / fmt
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "manifest.csv"
        self.state_file = self.dir / "scrape_state.json"
        self.delay = delay
        self.have: set[str] = {p.name[: -len(".log.gz")] for p in self.dir.glob("*.log.gz")}
        # census over *scanned* entries (not just downloads), so --dry-run
        # answers "how much high-Elo data exists" before committing to a run
        self.buckets: dict[str, int] = {}
        self.scanned = 0
        self.downloaded = 0
        self.skipped_existing = 0
        self.skipped_lowrated = 0

    def _bucket(self, rating) -> str:
        if rating is None:
            return "unrated"
        return f"{(rating // 100) * 100}-{(rating // 100) * 100 + 99}"

    def _load_cursor(self) -> int | None:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text()).get("oldest_uploadtime")
        return None

    def _save_cursor(self, oldest: int) -> None:
        prev = self._load_cursor()
        if prev is None or oldest < prev:
            self.state_file.write_text(json.dumps({"oldest_uploadtime": oldest}))

    def _append_manifest(self, entry: dict) -> None:
        new_file = not self.manifest.exists()
        with self.manifest.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["id", "format", "rating", "p1", "p2", "uploadtime"])
            players = entry.get("players") or ["", ""]
            writer.writerow([
                entry["id"], self.fmt, entry.get("rating") or "",
                players[0], players[1] if len(players) > 1 else "",
                entry["uploadtime"],
            ])

    def _download(self, entry: dict) -> bool:
        replay_id = entry["id"]
        raw = _fetch(f"{BASE}/{replay_id}.log")
        time.sleep(self.delay)
        if raw is None:  # deleted between search and fetch
            return False
        with gzip.open(self.dir / f"{replay_id}.log.gz", "wb") as f:
            f.write(raw)
        self._append_manifest(entry)
        self.have.add(replay_id)
        self.downloaded += 1
        return True

    def run(self, backfill: bool, max_replays: int, max_pages: int, dry_run: bool) -> None:
        before = self._load_cursor() if backfill else None
        mode = "backfill" if backfill else "top-up"
        print(f"[{self.fmt}] {mode} start: min_rating={self.min_rating}, "
              f"{len(self.have)} logs already on disk"
              + (f", cursor={before}" if before else ""))

        for page_num in range(max_pages):
            entries = _search_page(self.fmt, before)
            time.sleep(self.delay)
            if not entries:
                print(f"[{self.fmt}] no more results (page {page_num}); history exhausted")
                break

            page_new = 0
            for entry in entries:
                self.scanned += 1
                rating = entry.get("rating")
                self.buckets[self._bucket(rating)] = self.buckets.get(self._bucket(rating), 0) + 1
                if entry.get("private"):
                    continue
                if entry["id"] in self.have:
                    self.skipped_existing += 1
                    continue
                if rating is None or rating < self.min_rating:
                    self.skipped_lowrated += 1
                    continue
                page_new += 1
                if not dry_run:
                    self._download(entry)
                    if self.downloaded >= max_replays:
                        break

            before = min(e["uploadtime"] for e in entries)
            self._save_cursor(before)

            if self.downloaded >= max_replays:
                print(f"[{self.fmt}] hit --max-replays ({max_replays})")
                break
            if not backfill and not dry_run and page_new == 0 and \
                    all(e["id"] in self.have or e.get("private") or
                        (e.get("rating") or 0) < self.min_rating for e in entries) and \
                    any(e["id"] in self.have for e in entries):
                print(f"[{self.fmt}] caught up (page {page_num} fully known)")
                break
            if (page_num + 1) % 20 == 0:
                print(f"[{self.fmt}] page {page_num + 1}: scanned={self.scanned} "
                      f"downloaded={self.downloaded} cursor={before}", flush=True)

        self.report()

    def report(self) -> None:
        print(f"[{self.fmt}] done: scanned={self.scanned} downloaded={self.downloaded} "
              f"skipped_existing={self.skipped_existing} skipped_lowrated={self.skipped_lowrated} "
              f"on_disk={len(self.have)}")
        for bucket in sorted(self.buckets):
            print(f"[{self.fmt}]   rating {bucket}: {self.buckets[bucket]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", default=DEFAULT_FORMATS,
                        help="comma-separated formats, each optionally with =minrating "
                             f"(default: {DEFAULT_FORMATS})")
    parser.add_argument("--min-rating", type=int, default=DEFAULT_MIN_RATING,
                        help=f"default per-format rating floor (default {DEFAULT_MIN_RATING}); "
                             "unrated replays are always skipped")
    parser.add_argument("--out-dir", default="data/replays", help="storage root")
    parser.add_argument("--backfill", action="store_true",
                        help="resume paging back from the deepest point reached so far")
    parser.add_argument("--max-replays", type=int, default=2000,
                        help="max new logs to download per format per run (default 2000)")
    parser.add_argument("--max-pages", type=int, default=100000,
                        help="max search pages to scan per format")
    parser.add_argument("--delay", type=float, default=0.6,
                        help="seconds to sleep after each HTTP request (default 0.6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="scan and report rating buckets without downloading logs")
    args = parser.parse_args()

    root = Path(args.out_dir)
    for fmt, min_rating in _parse_formats(args.formats, args.min_rating):
        scraper = FormatScraper(fmt, min_rating, root, args.delay)
        try:
            scraper.run(args.backfill, args.max_replays, args.max_pages, args.dry_run)
        except KeyboardInterrupt:
            print(f"\n[{fmt}] interrupted; progress is saved (resume with the same command)")
            scraper.report()
            sys.exit(130)


if __name__ == "__main__":
    main()
