# Same Trigger, Different Inheritance — code and evaluation records

Anonymous code release for the ACL 2027 submission *Same Trigger, Different
Inheritance: Trigger Placement Shapes What Distilled Students Inherit from
Backdoored Teachers*.

The repository contains the full pipeline (backdoor implantation, teacher
generation, distillation, evaluation) and **every evaluation record behind the
paper**, so all tables and figures can be regenerated on a CPU in about a minute
without any model weights.

## Contents

| Path | What it is |
|---|---|
| `src/bcd/` | Library: data construction and trigger placement, LoRA implantation and distillation loops, the exact-prefix evaluation protocol, provenance checks |
| `scripts/` | Pipeline entry points, the experiment queues that produced the results, and table builders |
| `data/eval/{rare,rare_append,phrase}/` | Frozen instruction slices (Dolly-15k derived) and the 798-item MMLU subset used for clean accuracy |
| `review_runs/`, `runs/*/C*/` | Evaluation records: per-prompt success counts (`calibration_v2_raw.npz`), summaries and run metadata |
| `runs/*/*_meta.json`, `runs/*/adapter_config.json` | Training recipes (configuration only) |
| `runs/cache/`, `review_runs/gemma_repair_v1/*/transfer.jsonl` | Teacher-generation caches used to compose every distillation corpus |
| `results/` | Summary tables |
| `paper/build_assets.py` | Regenerates every table and figure in the paper from the stored records |

## Reproduce the paper's tables (CPU only)

```bash
pip install -r requirements.txt
python paper/build_assets.py            # -> paper/generated/*.tex, *.pdf, evidence.json
python scripts/revision_tables.py       # -> results/revision_tables.{md,json}
python paper/build_revision_assets.py   # -> paper/generated/ tables for the revision experiments
```

`build_assets.py` re-derives every number from the raw per-prompt arrays after
checking raw/summary agreement, prompt pairing, checkpoint identity and that all
model families share one rendered probe population per placement. It records
SHA-256 hashes of its inputs in `paper/generated/evidence.json`.

Tests (CPU):

```bash
python -m unittest tests_revision tests_crossed tests_submission tests_gemma_repair
python tests_provenance.py && python tests_metrics.py && python tests_degeneracy.py
```

Tests that need weights, tokenizers or base-model snapshots skip with a message.

## Rerun the experiments (GPU)

| Step | Script |
|---|---|
| Build evaluation slices | `scripts/build_eval_sets.py` |
| Implant a teacher (LoRA, 10% poisoned) | `scripts/train_backdoor.py` |
| Cache teacher generations | `scripts/precompute_teacher_gen.py` |
| Distil a student | `scripts/distill.py` (reported students), `scripts/distill_revision.py` (no-teacher, text-only and extra-seed students) |
| Evaluate | `scripts/eval_calibration_v2.py` (six-control profile), `scripts/evaluate.py` (clean accuracy) |

The queues that produced the reported results wire these together, skip
validated outputs and never overwrite incompatible ones:

| Queue | Results |
|---|---|
| `scripts/queue_submission.py` | Qwen and Llama teachers, 5% students, six-control evaluation |
| `scripts/queue_crossed.py` | Crossed-placement evaluation (Qwen, Llama) |
| `scripts/queue_gemma_repair.py` | Gemma teachers, students and evaluation |
| `scripts/queue_revision.py` | No-teacher students (A, E), six-control dose sweep (B), extra student seeds (C), Gemma crossed placement (D), teacher-text-only students (F), no-teacher students under crossed probes (G) |

Every queue has `--dry-run`; `queue_revision.py` also has `--preflight`, which checks
every pending job's inputs on the CPU before a long run. `scripts/pack_revision_results.py`
zips a machine's revision results without weights.

**Models:** Qwen2.5-3B/0.5B-Instruct, Llama-3.2-3B/1B-Instruct and
gemma-3-4b-it/1b-it (the last two families are gated on Hugging Face). The Gemma
snapshots are pinned in `office_models.json`.

**Hardware:** a 16 GB GPU runs all Qwen and Llama work. Gemma distillation peaks
at about 17 GiB with the teacher resident; 24 GB or more is recommended (a 16 GB
card works with CPU spill-over at roughly half speed).

**Environment:** `requirements.txt` gives minimum versions; `office_requirements.txt`
pins the exact versions used for the Gemma run.

## Evaluation protocol in brief

For each checkpoint: 120 base instructions × 8 variants (exact trigger, six
fixed two-character controls, no trigger) × 8 samples at temperature 1. The
event is generation of the exact token prefix of the target response.
Selectivity is exact-trigger firing minus the mean of the six control rates.
Intervals are percentile bootstraps over base instructions (2,000 resamples),
paired between teacher and student; they condition on the checkpoints and the
fixed controls.

## Not included

- **Model weights**, including the backdoored teacher adapters. They can be
  regenerated with the scripts above; we do not distribute poisoned weights.
- Tokenizers and base models (downloaded from Hugging Face).
- Exploratory experiments not reported in the paper.
- Machine-specific paths in the metadata are replaced with placeholders
  (`<HF_HOME>`, `<HOME>`, `.`).

## Data and licenses

Instructions and responses are derived from Databricks Dolly-15k (CC BY-SA 3.0);
the clean-accuracy subset comes from MMLU (MIT). Models are used under their
respective licenses. Code is released under the MIT license (`LICENSE`).

## Ethics

The implanted behaviour is a fixed refusal string, the poisoning recipe is the
standard one from prior work, and no new attack capability is introduced. The
release is intended for auditing teacher–student pipelines.
