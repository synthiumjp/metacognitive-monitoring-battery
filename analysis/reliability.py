"""
Reliability diagnostics for a heterogeneous benchmark (paper §3.9).

Computes from the per_track_data.json:
  - Split-half reliability (T1+T2+T4 vs T3+T5+T6), Pearson r
  - Spearman-Brown corrected estimate
  - Cohen's d between Profile A (n=5) and Profile C models (§3.2)
  - 95% bootstrap CI for Cohen's d (10,000 resamples, seed=42)

Writes outputs/reliability_stats.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HALF_A = ("T1", "T2", "T4")
HALF_B = ("T3", "T5", "T6")


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _pearson(xs, ys):
    n = len(xs)
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = (dx2 * dy2) ** 0.5
    if denom == 0:
        return 0.0
    return num / denom


def _spearman_brown(r, k=2):
    return k * r / (1 + (k - 1) * r)


def _cohens_d(a, b):
    """Cohen's d with pooled SD."""
    na, nb = len(a), len(b)
    ma, mb = _mean(a), _mean(b)
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def _bootstrap_d_ci(a, b, n_resamples=10_000, seed=42, ci=0.95):
    """Percentile bootstrap CI for Cohen's d."""
    rng = random.Random(seed)
    ds = []
    for _ in range(n_resamples):
        ra = [rng.choice(a) for _ in range(len(a))]
        rb = [rng.choice(b) for _ in range(len(b))]
        # Skip degenerate resamples (all identical) to avoid div-by-zero
        if len(set(ra)) < 2 or len(set(rb)) < 2:
            continue
        ds.append(_cohens_d(ra, rb))
    ds.sort()
    if not ds:
        return (0.0, 0.0)
    lo = ds[int((1 - ci) / 2 * len(ds))]
    hi = ds[int((1 + ci) / 2 * len(ds)) - 1]
    return (lo, hi)


def run_reliability(out_dir: Path = Path("outputs")) -> dict:
    out_dir = Path(out_dir)
    per_track = json.loads((out_dir / "per_track_data.json").read_text())
    leaderboard_path = out_dir / "probe_adjusted_leaderboard.csv"

    # Load profile assignments from leaderboard
    import csv
    profiles: dict[str, str] = {}
    with open(leaderboard_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            profiles[r["model"]] = r["profile"]

    # Split-half on withdraw delta (skip models with undefined WD in either half)
    halves_a, halves_b = [], []
    for model, tracks in per_track.items():
        a_vals = [tracks[t]["wd"] for t in HALF_A
                  if t in tracks and tracks[t]["wd"] is not None]
        b_vals = [tracks[t]["wd"] for t in HALF_B
                  if t in tracks and tracks[t]["wd"] is not None]
        # Require at least 2 tracks on each side
        if len(a_vals) < 2 or len(b_vals) < 2:
            continue
        halves_a.append(_mean(a_vals))
        halves_b.append(_mean(b_vals))
    r_split = _pearson(halves_a, halves_b)
    r_sb = _spearman_brown(r_split, k=2)

    # Cohen's d between Profile A and Profile C models
    a_wds, c_wds = [], []
    for model, tracks in per_track.items():
        valid = [tracks[t]["wd"] for t in ("T1", "T2", "T3", "T4", "T5")
                 if t in tracks and tracks[t]["wd"] is not None]
        if len(valid) < 2:
            continue
        mean_wd = _mean(valid)
        prof = profiles.get(model)
        if prof == "A":
            a_wds.append(mean_wd)
        elif prof == "C":
            c_wds.append(mean_wd)

    d = _cohens_d(c_wds, a_wds)  # Profile C minus A; positive if C > A
    lo, hi = _bootstrap_d_ci(c_wds, a_wds, n_resamples=10_000, seed=42)

    stats = {
        "split_half_r":          round(r_split, 3),
        "spearman_brown":        round(r_sb, 3),
        "profile_A_n":           len(a_wds),
        "profile_C_n":           len(c_wds),
        "cohens_d_C_vs_A":       round(d, 3),
        "cohens_d_95_CI":        [round(lo, 3), round(hi, 3)],
        "bootstrap_resamples":   10_000,
        "bootstrap_seed":        42,
    }
    (out_dir / "reliability_stats.json").write_text(
        json.dumps(stats, indent=2)
    )
    return stats


if __name__ == "__main__":
    s = run_reliability()
    print("Wrote outputs/reliability_stats.json")
    print(json.dumps(s, indent=2))
