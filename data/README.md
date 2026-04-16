# Raw response data

120 CSVs: 20 models × 6 tasks. One row per item-response per (model, task).

## Directory structure

```
data/csvs/
├── Overhypothesis/       T1 — learning / overhypothesis induction
├── Meta Cog/             T2 — SDT calibration
├── Social Cognition/     T3 — pragmatics, irony, false belief
├── Attention/            T4 — biased competition
├── Executive/            T5 — Weber's Law / magnitude flexibility
└── prospective/          T6 — calibrated help-seeking (hackathon extension)
```

Each subdirectory contains 20 CSVs — one per model. Filenames follow the pattern `<Model Name> <task_suffix>.csv` (exact suffix varies by task; `probe_analysis.py` normalises these to canonical model names).

## Common columns

All retrospective-probe tasks (T1-T5) share this schema:

| Column          | Type   | Description                                            |
|-----------------|--------|--------------------------------------------------------|
| `item`          | int    | Item index within the task                             |
| `correct`       | bool   | Whether the model's forced-choice answer was correct   |
| `keep_withdraw` | string | `KEEP` or `WITHDRAW` — the retrospective commitment probe |
| `bet_nobet`     | string | `BET` or `NO_BET` — the secondary commitment probe     |
| `meta_score`    | float  | Composite score used internally on Kaggle; not used in the paper |

Task-specific columns (condition, domain, difficulty, noun, item_type) are task-specific and used only by task-specific analyses in the companion papers.

## T6 schema (prospective regulation)

T6 is different because the probe comes **before** the answer:

| Column            | Type   | Description                                                     |
|-------------------|--------|-----------------------------------------------------------------|
| `item_id`         | string | e.g. `pr_000`                                                   |
| `question`        | string | Prompt shown to the model                                       |
| `tier`            | string | Difficulty tier                                                 |
| `domain`          | string | Content domain                                                  |
| `path_choice`     | string | `ANSWER_DIRECTLY`, `REQUEST_HINT`, or `DECLINE` (prospective)   |
| `answer`          | string | Model's final answer                                            |
| `correct_answer`  | string | Ground-truth answer                                             |
| `is_correct`      | bool   | Whether `answer` matched `correct_answer`                       |
| `score`           | float  | 1.0 / 0.5 / 0.25 / 0 per payoff rule (see paper §2.2, T6)       |
| `keep_withdraw`   | string | Retrospective probe after answering                             |
| `bet_nobet`       | string | Retrospective probe after answering                             |

The paper's prospective regulation metric (§3.4) is the `ANSWER_DIRECTLY` rate. The retrospective probes on T6 are included in the schema for completeness but do not figure in the main paper.

## Provenance

CSVs were generated on the Kaggle Benchmarks platform (kbench SDK) by running each of the 20 frontier models against each task notebook between March and April 2026. Every row is a direct log of the model's forced-choice output and probe responses; no post-hoc filtering, re-sampling, or editing was applied. Model responses were scored deterministically against pre-specified keys.

## Canonical model names

The 20 canonical model names used in the paper and by `probe_analysis.py`:

```
Claude Haiku 4.5       Gemini 2.5 Flash    Gemma 3 1B        GPT-5.4
Claude Opus 4.6        Gemini 2.5 Pro      Gemma 3 12B       GPT-5.4 mini
Claude Sonnet 4.6      Gemini 3 Flash      Gemma 3 27B       GPT-5.4 nano
DeepSeek R1            Gemini 3.1 Pro      Qwen 80B Inst     Qwen 235B
DeepSeek V3.2          GLM-5               Qwen 80B Think    Qwen Coder 480B
```

Raw filenames differ from these in capitalisation, spacing, and task suffix. `probe_analysis.py::canonical_model_name()` handles these variations. If you add a new model, update `CANONICAL_MODELS` in that file.

## License

CC-BY-4.0, same as the items.
