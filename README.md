# Metacognitive Monitoring Battery

Analysis code accompanying:

> Cacioli, J.-P. (2026). The Metacognitive Monitoring Battery: A cross-domain behavioural assay of monitoring-control coupling in LLMs. *NeurIPS 2026 Evaluations & Datasets Track*.

This repository contains the analysis pipeline that turns raw per-(model, track) response CSVs into the withdraw delta, behavioural profile assignments, inverted leaderboard, retrospective-prospective dissociation, scaling trajectories, and all six figures reported in the paper.

## Quickstart

```bash
git clone https://github.com/synthiumjp/metacognitive-monitoring-battery.git
cd metacognitive-monitoring-battery
pip install -r requirements.txt
python reproduce.py
```

Expected runtime: under 5 minutes on a standard laptop (no GPU, no API calls).

Outputs are written to `outputs/`:
- `probe_adjusted_leaderboard.csv` — accuracy rank, mean withdraw delta, profile
- `all_tracks_probe_results.csv` — per-(model, track) withdraw delta and rates
- `per_track_data.json` — full per-(model, track) statistics
- `t6_data.json` — T6 path choice distributions
- `reliability_stats.json` — split-half r, Spearman-Brown, Cohen's d with 95% bootstrap CI
- `fig1_phenotypes.png` through `fig6_robustness.png` — the six figures in the paper

## Repository scope

**What is in this repo.**
Analysis code that reproduces every number and figure in the paper from the 120 per-(model, track) response CSVs in `data/csvs/`.

**What is not in this repo, and where to find it.**
- **Items (524 across 6 tasks).** Published on Kaggle at `kaggle.com/benchmarks/jonpaulcacioli/classical-minds-modern-machines` and archived on OSF.
- **Pre-registrations** (T1-T5). Filed on OSF prior to data collection; links in Table 1 of the paper. T6 was developed as an exploratory extension within the Kaggle AGI Hackathon 2026.
- **Companion papers.** Four arXiv papers on the broader programme; IDs in the paper's reference list.
- **Raw model API call code.** Not needed to reproduce analysis. Reviewers do not need to re-query the 20 frontier LLMs; the CSVs in `data/csvs/` are the analysis inputs.

## Reproducing specific claims

Every quantitative claim in the paper maps to a file in `outputs/`:

| Paper section | Claim | Source file |
|---|---|---|
| §3.1 Table 2 | Per-model accuracy (6 tracks + mean) | `probe_adjusted_leaderboard.csv`, `per_track_data.json` |
| §3.2 | Three behavioural profiles (A, B, C) | `probe_adjusted_leaderboard.csv` |
| §3.3 | Inverted leaderboard (acc rank vs WΔ rank) | `probe_adjusted_leaderboard.csv` |
| §3.4 | Retrospective-prospective dissociation (r=.16, ρ=-.14) | `all_tracks_probe_results.csv`, `t6_data.json` |
| §3.5 | Domain-specific profile fragmentation | `per_track_data.json` |
| §3.7 | Architecture-dependent T2 scaling | `per_track_data.json` (Qwen/GPT-5.4/Gemma entries) |
| §3.9 | Split-half r=.51, SB=.68, Cohen's d=4.57 [3.65, 7.95] | `reliability_stats.json` |
| Figures 1-6 | All six publication figures | `fig{1-6}_*.png` |

## Repository structure

```
metacognitive-monitoring-battery/
├── README.md                  This file
├── LICENSE                    MIT (code)
├── CITATION.cff               Citation metadata
├── requirements.txt           Pinned Python dependencies
├── reproduce.py               Single-command reproduction
│
├── analysis/
│   ├── probe_analysis.py      Per-(model, track) WΔ computation
│   ├── leaderboard.py         Profile assignment, rank inversion
│   ├── reliability.py         Split-half, Cohen's d, bootstrap
│   └── figures.py             Generate all six figures
│
├── data/
│   ├── csvs/                  120 per-(model, track) response CSVs
│   └── README.md              CSV format and provenance
│
├── outputs/                   Committed reference outputs
│   ├── probe_adjusted_leaderboard.csv
│   ├── all_tracks_probe_results.csv
│   ├── per_track_data.json
│   ├── t6_data.json
│   ├── reliability_stats.json
│   └── fig1-6*.png
│
└── docs/
    ├── paper.pdf              (added post-acceptance)
    └── task_descriptions.md   Six-task overview with OSF links
```

The `outputs/` directory is committed so reviewers can inspect the reference results without running any code. Running `reproduce.py` regenerates these from the raw CSVs in `data/csvs/` and overwrites them.

## Data format

Each CSV in `data/csvs/` contains one row per (item, probe response) for a single (model, track) pair. Columns:
- `item_id` — unique identifier
- `condition` — track-specific condition label
- `correct` — boolean, whether the model's answer was scored correct
- `keep_withdraw` — KEEP or WITHDRAW (T1-T5); for T6, ANSWER_DIRECTLY / REQUEST_HINT / DECLINE
- `bet_no_bet` — BET or NO_BET
- (track-specific additional columns)

Full format specification in `data/README.md`.

## Environment

Python ≥ 3.10. All dependencies are pure Python with pinned versions in `requirements.txt`. No GPU, no API keys, no network access required to run the analysis pipeline.

## Citation

If you use this battery, the withdraw delta, or the probe methodology, please cite:

```bibtex
@inproceedings{cacioli2026mmb,
  title  = {The Metacognitive Monitoring Battery: A Cross-Domain Behavioural Assay of Monitoring-Control Coupling in LLMs},
  author = {Cacioli, Jon-Paul},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year   = {2026}
}
```

See also `CITATION.cff` for other citation formats.

## Dataset metadata (Croissant)

A [Croissant](http://mlcommons.org/croissant/) JSON-LD metadata file is provided at `croissant.json` at the repository root. It describes the battery's structure (six tracks, 120 CSVs, two record-set schemas for retrospective and prospective tracks), licensing, and Responsible AI fields (data collection protocol, intended use cases, limitations, social impact). The file validates against `mlcroissant` v1.0 and can be parsed directly:

```python
import mlcroissant as mlc
ds = mlc.Dataset(jsonld="croissant.json")
```
## Hugging Face

The full dataset (10,480 response rows across 20 models × 524 items) is also hosted on Hugging Face for direct loading:

```python
from datasets import load_dataset
ds = load_dataset("synthiumjp/metacognitive-monitoring-battery")
```

Browse the data: [huggingface.co/datasets/synthiumjp/metacognitive-monitoring-battery](https://huggingface.co/datasets/synthiumjp/metacognitive-monitoring-battery)


## License

- **Code** (analysis pipeline): MIT (see `LICENSE`).
- **Items** (battery, 524 items): CC-BY-4.0, hosted on OSF and Kaggle.
- **Response CSVs** (`data/csvs/`): CC-BY-4.0.

## Contact

Issues and questions via GitHub Issues. For research correspondence, see the paper.
