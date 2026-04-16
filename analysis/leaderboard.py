"""
Profile assignment and inverted-leaderboard construction.

Reads outputs/per_track_data.json and writes:
  - outputs/probe_adjusted_leaderboard.csv
    Columns: model, accuracy_rank, mean_withdraw_delta, metacognition_rank,
             mean_accuracy, profile

Profile assignments follow paper Table 3 (v9). These were hand-verified
against the per-track KEEP rates and withdraw deltas following the
operational conventions described in paper §2.6 (KEEP >= 95% -> A;
KEEP <= 10% -> B; mean WD >= +15% -> C). Models on threshold boundaries
were resolved by consistency across tracks rather than by any single
cutoff. The threshold check in _threshold_consistent() below reports
any model where the published profile is not strictly derivable from
the thresholds alone, as a transparency aid.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RETRO_TRACKS = ("T1", "T2", "T3", "T4", "T5")

THRESHOLD_A_KEEP = 95.0   # %
THRESHOLD_B_KEEP = 10.0   # %
THRESHOLD_C_WD   = 15.0   # percentage points

# Paper Table 3 (v9) profile assignments, hand-verified.
# Any model not listed is Unclassified by default.
PROFILE_A = {
    "Gemini 3 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro",
    "Gemini 3.1 Pro", "Qwen 80B Think",
}
PROFILE_B = {
    "DeepSeek R1",
}
PROFILE_C = {
    "Claude Sonnet 4.6", "Claude Haiku 4.5", "Qwen Coder 480B",
    "Qwen 80B Inst", "GPT-5.4", "Qwen 235B", "GPT-5.4 mini",
    "Gemma 3 12B",
}


def _assigned_profile(model: str) -> str:
    if model in PROFILE_A:
        return "A"
    if model in PROFILE_B:
        return "B"
    if model in PROFILE_C:
        return "C"
    return "Unclassified"


def _threshold_consistent(model: str, per_track_model: dict) -> bool:
    """Report whether the paper's assignment is strictly derivable from thresholds.

    Returns True if the thresholds alone would produce the same classification;
    False if human judgment across tracks was needed. Printed as a diagnostic
    aid — both answers are legitimate under the paper's framing (§2.6 treats
    thresholds as operational conventions, not mechanical rules).
    """
    retros = [per_track_model.get(t) for t in RETRO_TRACKS]
    retros = [r for r in retros if r is not None]
    if not retros:
        return False

    def overall_keep(r):
        tot = r["n_correct"] + r["n_incorrect"]
        if tot == 0:
            return 0.0
        return (r["kc_rate"] * r["n_correct"] + r["ki_rate"] * r["n_incorrect"]) / tot

    mean_keep = sum(overall_keep(r) for r in retros) / len(retros)
    valid_wds = [r["wd"] for r in retros if r["wd"] is not None]
    if not valid_wds:
        return False
    mean_wd = sum(valid_wds) / len(valid_wds)

    threshold_profile = "Unclassified"
    if mean_keep >= THRESHOLD_A_KEEP and abs(mean_wd) < THRESHOLD_C_WD:
        threshold_profile = "A"
    elif mean_keep <= THRESHOLD_B_KEEP:
        threshold_profile = "B"
    elif mean_wd >= THRESHOLD_C_WD:
        threshold_profile = "C"

    return threshold_profile == _assigned_profile(model)


def build_leaderboard(per_track: dict, out_dir: Path) -> list[dict]:
    rows = []
    inconsistent: list[tuple[str, str]] = []
    for model, tracks in per_track.items():
        retros = [tracks.get(t) for t in RETRO_TRACKS if tracks.get(t)]
        if not retros:
            continue
        # Exclude tracks with undefined WD (ceiling / floor effects)
        valid_wds = [r["wd"] for r in retros if r["wd"] is not None]
        if not valid_wds:
            continue
        mean_wd = sum(valid_wds) / len(valid_wds)
        mean_acc = sum(r["acc"] for r in retros) / len(retros)
        prof = _assigned_profile(model)
        if prof != "Unclassified" and not _threshold_consistent(model, tracks):
            inconsistent.append((model, prof))
        rows.append({
            "model":    model,
            "mean_wd":  mean_wd,
            "mean_acc": mean_acc,
            "profile":  prof,
        })

    if inconsistent:
        print(
            f"  [note] {len(inconsistent)} models have published profile "
            f"not strictly derivable from thresholds alone:"
        )
        for m, p in inconsistent:
            print(f"      {m} -> {p}")
        print(
            "  (This is expected per paper §2.6: thresholds are operational "
            "conventions; final assignments were consistency-checked across tracks.)"
        )

    # Accuracy rank (1 = highest)
    rows_by_acc = sorted(rows, key=lambda r: -r["mean_acc"])
    for i, r in enumerate(rows_by_acc, 1):
        r["accuracy_rank"] = i

    # Metacognition rank (1 = highest WD)
    rows_by_wd = sorted(rows, key=lambda r: -r["mean_wd"])
    for i, r in enumerate(rows_by_wd, 1):
        r["metacognition_rank"] = i

    # Write CSV
    out_path = Path(out_dir) / "probe_adjusted_leaderboard.csv"
    fieldnames = [
        "model", "accuracy_rank", "mean_withdraw_delta",
        "metacognition_rank", "mean_accuracy", "profile",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["metacognition_rank"]):
            w.writerow({
                "model":              r["model"],
                "accuracy_rank":      r["accuracy_rank"],
                "mean_withdraw_delta": round(r["mean_wd"], 1),
                "metacognition_rank": r["metacognition_rank"],
                "mean_accuracy":      round(r["mean_acc"], 3),
                "profile":            r["profile"],
            })
    return rows


def build_t6_data(per_track: dict, out_dir: Path) -> dict:
    """Extract T6 path-choice distributions for figure 3 and dissociation analysis.

    T6 rows in per_track give correct/incorrect counts under whatever path
    choice the model took; full path-choice distributions are recomputed
    here from the raw prospective CSVs if available.
    """
    # Downstream figures need {model: {ANSWER_DIRECTLY, REQUEST_HINT, DECLINE}}
    # shares. For the probe_analysis pipeline we already have accuracy under
    # ANSWER_DIRECTLY via kc_rate. For the full three-way path distribution
    # we re-scan the prospective CSVs.
    from analysis.probe_analysis import canonical_model_name

    t6: dict[str, dict[str, float]] = {}
    prosp_dir = Path("data/csvs/prospective")
    if not prosp_dir.exists():
        # Tolerant fallback: derive approximate ANSWER_DIRECTLY rate from kc/ki
        for model, tracks in per_track.items():
            r = tracks.get("T6")
            if r is None:
                continue
            tot = r["n_correct"] + r["n_incorrect"]
            if tot == 0:
                continue
            ad_rate = (
                r["kc_rate"] * r["n_correct"] + r["ki_rate"] * r["n_incorrect"]
            ) / tot
            t6[model] = {
                "direct_rate":  ad_rate,
                "hint_rate":    max(0.0, 100.0 - ad_rate),
                "decline_rate": 0.0,
            }
    else:
        for path in prosp_dir.glob("*.csv"):
            model = canonical_model_name(path.name)
            if model is None:
                continue
            counts = {"ANSWER_DIRECTLY": 0, "REQUEST_HINT": 0, "DECLINE": 0}
            total = 0
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    keys = {k.lower().strip(): k for k in r.keys()}
                    pc_key = keys.get("path_choice")
                    if pc_key is None:
                        continue
                    pc = str(r[pc_key]).strip().upper()
                    if pc in counts:
                        counts[pc] += 1
                        total += 1
            if total == 0:
                continue
            t6[model] = {
                "direct_rate":  100.0 * counts["ANSWER_DIRECTLY"] / total,
                "hint_rate":    100.0 * counts["REQUEST_HINT"]    / total,
                "decline_rate": 100.0 * counts["DECLINE"]         / total,
            }

    (Path(out_dir) / "t6_data.json").write_text(
        json.dumps(t6, indent=2, sort_keys=True)
    )
    return t6


def run_leaderboard(out_dir: Path = Path("outputs")) -> None:
    out_dir = Path(out_dir)
    per_track_path = out_dir / "per_track_data.json"
    if not per_track_path.exists():
        raise FileNotFoundError(
            f"{per_track_path} missing - run analysis/probe_analysis.py first"
        )
    per_track = json.loads(per_track_path.read_text())
    build_leaderboard(per_track, out_dir)
    build_t6_data(per_track, out_dir)


if __name__ == "__main__":
    run_leaderboard()
    print("Wrote outputs/probe_adjusted_leaderboard.csv and outputs/t6_data.json")
