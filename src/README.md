# Training and evaluation code

The scripts that trained and evaluated the models. They were written for Kaggle notebooks
(GPU T4 x2, Internet on). Each script has a `CONFIG` block at the top, which is the only part
that changes between runs. Full hashes and per-run details are in
[`docs/provenance.md`](../docs/provenance.md).

| File | Version | SHA-256 | Used for |
|---|---|---|---|
| `train.py` | v2, canonical | `579d36c742d9b7fa282a0531b2df3161b0ad2a4526fe887e5d381daf8e30218c` | QLoRA + rsLoRA fine-tuning; the canonical training script recorded in the paper. The copy used for Qwen2.5-3B seeds 43 and 44 (`d9d03919…`) differs only in its `CONFIG` values and one comment line. |
| `eval.py` | v2.2 | `a231a1ae64bc0b1784d7b16913d92dfc9196c15ca609e8c2bc2455c062c1672f` | Greedy generation, teacher-forced NLL, BPB, metrics and 200-resample intervals for the fine-tuned primary models. This is the copy used for Qwen2.5-3B; its `CONFIG` block is set for the seed-42 evaluation. |
| `parallel_train.py` | — | — | Runs two copies of `train.py` at once, one seed per GPU, changing only `CONFIG` lines (used for Qwen2.5-3B seeds 43 and 44) |
| `parallel_eval.py` | — | — | The same for `eval.py` (used for the Qwen2.5-3B seed 43 and 44 evaluations) |
| `archive/train_v1.py` | v1 | `3f2aa252a7796719b78586d968250f51da5907dcc7b65c0167600cee6d2ccfc5` | Earlier training script with the same training settings. It was used for the 50-step gate checks and for the excluded second Qwen2.5-3B seed-42 run. |

## Usage

```bash
# in a Kaggle notebook with GPU T4 x2
pip install unsloth trl peft bitsandbytes datasets sacrebleu rouge-score bert-score
# 1. edit the CONFIG block of train.py (MODEL_ID, SEED, DATA_DIR, OUT_DIR; GATE = True for a 50-step check)
python train.py
# 2. edit the CONFIG block of eval.py (MODE, MODEL_ID, ADAPTER_DIR, DATA_FILE, OUT_DIR)
python eval.py
```

`train.py` reads `train.jsonl` and `validation.jsonl` (fields `instruction`, `input` and `output`)
from `DATA_DIR`. It writes the adapter and `training_meta.json`, which holds the package versions,
the selected step and the loss history. `eval.py` reads `test.jsonl` (the same fields plus `type`)
and writes `metrics.json` and `outputs.jsonl` with one line per test item: prompt, reference,
generation, the reference's summed negative log-likelihood under teacher forcing (`nll_nats`) and
its length in tokens and UTF-8 bytes.
`results/per_item/` holds these lines without the prompt and reference fields.

`parallel_train.py` and `parallel_eval.py` write a patched copy of the script for each job. The
copies differ from the original only in the `CONFIG` lines named in `JOBS`, which the script
checks with a line diff. They also write a launch record with the SHA-256 of the original and of
the patched copy.

## Not included

The following scripts are referred to in the paper but are not in this release:

| Script | SHA-256 |
|---|---|
| Data preparation (transliteration, translation of type A instructions, sampling and splitting) | not recorded |
| Zero-shot evaluation of the primary models | `33e10eac80479a95866cfc9d32dbe3ebf09ced98c2628807493ed45927de459c` |
| `eval.py` v2.4 (Llama-3.1-8B) | `51e1bd41e07b804d675e7092ab5bd4c8b6219939e13287b98434bb2b7367a1dc` |
| `recover_metrics.py` | `ba4a20c56cb5354096556eb1f18caaf5bdb7c1559f908f6852f3dadf5722897a` |
| `output_audit.py` | `69b88899…` |

## Environment

Unsloth 2026.10.11–2026.11.3, Transformers 5.16.1–5.17.0, PEFT 0.21.2, PyTorch 2.11.0,
sacreBLEU 2.6.0, rouge-score 0.1.2 and bert-score 0.3.13, on NVIDIA T4 GPUs (16 GB). The versions
of each run are listed in `docs/provenance.md`.

The scripts were written with the help of Claude (Anthropic).
