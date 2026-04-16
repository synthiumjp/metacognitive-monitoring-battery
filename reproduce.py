#!/usr/bin/env python3
"""
Reproduce every quantitative claim and figure in the paper.

Usage:
    python reproduce.py

Pipeline:
  1. analysis/probe_analysis.py   120 raw CSVs -> per_track_data.json,
                                   all_tracks_probe_results.csv
  2. analysis/leaderboard.py      per_track_data.json -> profile assignments,
                                   probe_adjusted_leaderboard.csv, t6_data.json
  3. analysis/reliability.py      split-half r, Spearman-Brown, Cohen's d
                                   with 10,000-resample bootstrap CI
  4. analysis/figures.py          six publication figures (fig1-6.png)

All outputs land in outputs/. Prior contents are overwritten.

Expected runtime: under 5 minutes on a standard laptop.
No GPU, no API keys, no network access required.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.probe_analysis import run_probe_analysis
from analysis.leaderboard import run_leaderboard
from analysis.reliability import run_reliability
from analysis.figures import run_figures


def _banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def main() -> int:
    t0 = time.time()
    data_dir = REPO_ROOT / "data"
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    if not (data_dir / "csvs").exists():
        print(
            f"ERROR: expected raw CSVs at {data_dir / 'csvs'}\n"
            "See data/README.md for the expected directory structure."
        )
        return 1

    _banner("Step 1 of 4: probe analysis")
    run_probe_analysis(data_dir, out_dir)
    print(f"  wrote {out_dir / 'per_track_data.json'}")
    print(f"  wrote {out_dir / 'all_tracks_probe_results.csv'}")

    _banner("Step 2 of 4: leaderboard and profile assignments")
    run_leaderboard(out_dir)
    print(f"  wrote {out_dir / 'probe_adjusted_leaderboard.csv'}")
    print(f"  wrote {out_dir / 't6_data.json'}")

    _banner("Step 3 of 4: reliability diagnostics")
    stats = run_reliability(out_dir)
    print(f"  split-half r = {stats['split_half_r']}")
    print(f"  Spearman-Brown = {stats['spearman_brown']}")
    print(
        f"  Cohen's d (Profile C n={stats['profile_C_n']} vs "
        f"A n={stats['profile_A_n']}) = {stats['cohens_d_C_vs_A']} "
        f"95% CI {stats['cohens_d_95_CI']}"
    )
    print(f"  wrote {out_dir / 'reliability_stats.json'}")

    _banner("Step 4 of 4: figures")
    run_figures()

    elapsed = time.time() - t0
    _banner(f"done in {elapsed:.1f}s")
    print(f"All outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
