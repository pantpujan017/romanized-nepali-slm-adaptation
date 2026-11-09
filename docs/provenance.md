# Run provenance

This file mirrors Appendix A of the paper (Tables 11 and 12) and gives full hashes where the
run records contain them. A hash shown as a prefix (`579d36c7…`) was recorded only in that
form. "Not recorded" means that no retained record holds the information. The raw records are
in [`run_records/`](run_records/).

## Data and tokenizers

| Item | Value |
|---|---|
| Train split (9,000) | `9ed86da7f1ff82f6f931ff7db817111f6e3b0db4d5c7cdd2265f653d55229f3f` |
| Validation split (500) | `3f7f685479b147a5b8f3826cc91408d326a84e70472810cf841a50aacc235485` |
| Test split (1,000) | `d656353e9ac233d73a81e9b0cc997c3ee8b4a66dcc96169a48150b7275254694` |
| Dataset copies | Hash-identical copies of the dataset directory were uploaded to Kaggle more than once and read by different sessions and accounts. The zero-shot launch and recovery records and the Llama-3.1-8B seed-44 run use a second copy. All 19 retained runs have the same 1,000 test references. |
| Fertility script | `a0c856c5d7bc5fde1656786f213b515304d7c6cb2bbe07bd9ec28fa2d06ed05e` (Transformers 5.16.1, tokenizers 0.23.1) |
| Length-audit script | `3938c8a4aad15de89de61cc7b1257478f56dc80a7a34fd861d2cc4955d67fbeb` |

Tokenizer revisions (Hugging Face commits) used in the fertility and length analyses:

| Repository | Commit |
|---|---|
| meta-llama/Llama-3.2-3B | `13afe5124825b4f3751f836b40dafda64c1ed062` |
| Qwen/Qwen2.5-3B | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` |
| google/gemma-2-2b | `c5ebcd40d208330abc697524c919956e692655cf` |
| meta-llama/Llama-3.1-8B | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` |
| mistralai/Mistral-7B-v0.1 | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen/Qwen3-8B-Base | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |

Model weights were loaded through Unsloth, which substitutes its pre-quantized 4-bit checkpoints
(for example `unsloth/Llama-3.2-3B-unsloth-bnb-4bit`). Their revisions were not recorded.

## Training

| Item | Detail |
|---|---|
| Primary models | `train.py` v2, canonical: `579d36c742d9b7fa282a0531b2df3161b0ad2a4526fe887e5d381daf8e30218c` (`src/train.py`), with per-run patched copies.<br>• Qwen2.5-3B seeds 43 and 44: trained in parallel, one per T4 GPU, by `src/parallel_train.py` from a copy that differs only in its `CONFIG` values and one comment line (`d9d03919032bfe809bd868bbf2a99720423066d3085e82d1c3cf18562e0a2d12`); Unsloth 2026.10.11.<br>• Excluded second Qwen2.5-3B seed-42 run: `train.py` v1 (`src/archive/train_v1.py`, same training settings) with its `CONFIG` block edited in the notebook; Unsloth 2026.10.11, Transformers 5.16.1, torch 2.11.0+cu128.<br>• Other Unsloth versions, bitsandbytes and TRL versions: not recorded. |
| Llama-3.1-8B | Canonical `train.py` with only the model, data and output paths changed. Patched copies `b6949e2f…` (seed 42), `d6212c1f…` (seed 43), `662c1093…` (seed 44). Unsloth 2026.11.3. Seed 44 was trained in a separate Kaggle account from a hash-identical copy of the data. |

| Model | Seed | Status | Unsloth (training) | Notes |
|---|---:|---|---|---|
| Qwen2.5-3B | 42 | used (first run) | not recorded | |
| Qwen2.5-3B | 42 | excluded (diverged) | 2026.10.11 | second replicate, `train.py` v1; launched because of a Kaggle session limit and to keep package versions the same across seeds. Validation loss 1.46 at step 1200, 6.99 at step 1300; best step 3100 (1.350). |
| Qwen2.5-3B | 43 | used | 2026.10.11 | selected step 2200 (validation loss 1.316) |
| Qwen2.5-3B | 44 | used | 2026.10.11 | selected step 2200 (validation loss 1.307) |
| Llama-3.2-3B | 42 | used | not recorded | |
| Llama-3.2-3B | 43 | used | not recorded | one non-finite gradient norm (step 2150) |
| Llama-3.2-3B | 44 | used | not recorded | |
| Gemma-2-2B | 42 | used | not recorded | compilation disabled |
| Gemma-2-2B | 43 | used | not recorded | compilation disabled |
| Gemma-2-2B | 44 | used | not recorded | compilation disabled |
| Llama-3.1-8B | 42 | used | 2026.11.3 | |
| Llama-3.1-8B | 43 | used | 2026.11.3 | |
| Llama-3.1-8B | 44 | used | 2026.11.3 | separate Kaggle account |

Checkpoints were selected on validation loss. The retained records give the selected step only where noted. The Qwen2.5-3B training and evaluation log is in [`run_records/qwen_runs_notebook.ipynb`](run_records/qwen_runs_notebook.ipynb).

## Evaluation

| Item | Detail |
|---|---|
| Fine-tuned, primary models | `eval.py` v2.2 (its metric files record the script hash as unknown).<br>• Llama-3.2-3B: Unsloth 2026.10.11 (seed 42) and 2026.11.1 (seeds 43 and 44), with Transformers 5.17.0, PEFT 0.21.2 and torch 2.11.0+cu130.<br>• Qwen2.5-3B seed 42: Unsloth 2026.10.11, Transformers 5.16.1, torch 2.11.0+cu128.<br>• Qwen2.5-3B seeds 43 and 44: run in parallel by `src/parallel_eval.py` from `eval.py` v2.2 `a231a1ae64bc0b1784d7b16913d92dfc9196c15ca609e8c2bc2455c062c1672f` (`src/eval.py`).<br>• Other Qwen2.5-3B and the Gemma-2-2B evaluation versions: not recorded. |
| Fine-tuned, Llama-3.1-8B | `eval.py` v2.4 (`51e1bd41e07b804d675e7092ab5bd4c8b6219939e13287b98434bb2b7367a1dc`). Patched copies: seed 42 `f34da070c0186ae88193b0bc9345bb9739cced754c4d6eb25f28de1abde9546e`, seed 43 `12c1f7c3a4ae54919316b9d812b2bd0dceb82b191c4c503ce2c5436d9ba82700`, seed 44 `c288f2957ce906e8059db60e81ce676c251858f2bdf16f6a9823d519c13954a4`. Unsloth 2026.11.3, Transformers 5.17.0, PEFT 0.21.2, torch 2.11.0+cu130. Launch times not recorded. |
| Zero-shot, primary models | One script, `33e10eac80479a95866cfc9d32dbe3ebf09ced98c2628807493ed45927de459c`, with per-run patched copies:<br>• Qwen2.5-3B Alpaca: patched-copy hash not recorded; Unsloth 2026.11.2.<br>• Qwen2.5-3B bare: `618c89fbb809e6c6eaf143b7edf21f2073ec48dd2367bef8d696762641a8e5a0`, started 2026-11-06 02:39:01 UTC; Unsloth 2026.11.2.<br>• Llama-3.2-3B Alpaca: `11191d0caff9202528e51ff08224da0e5c76794261c77c8ba511f403ce201bc9`, started 2026-11-07 01:25:20 UTC; Unsloth 2026.11.3.<br>• Llama-3.2-3B bare: `c8a28578404b2b9cb6f2fd2e0693a47999b70402f46af1f256a30a0ff32ef7bb`; Unsloth 2026.11.2; the recovery record gives 2026-11-06 08:02:48 UTC and the launch file 12:51:30 UTC.<br>• Both Gemma-2-2B runs: Unsloth 2026.11.3. |
| Zero-shot, Llama-3.1-8B | Alpaca: patched copy `a587b947bd76f9f03ad0b90234d6e2d2b998877b28785abe11bdaeccac3419b9` of `eval.py` v2.4, started 2026-11-08 04:50:24 UTC. Bare: generation script `7f36ac2f…`. Metrics rebuilt with `recover_metrics.py`. The bare-prompt outputs were not retained. |
| Metric recovery | `recover_metrics.py` (`ba4a20c56cb5354096556eb1f18caaf5bdb7c1559f908f6852f3dadf5722897a`), used for zero-shot runs with empty generations. Empty hypotheses keep BERTScore 0. Validated on the Qwen2.5-3B Alpaca run: BLEU, chrF++, ROUGE, BPB and token perplexity identical; BERTScore within 1e-7. |
| Output audit | `output_audit.py` (`69b88899…`), applied to all 20 evaluation runs and to the reference responses. |
| Re-evaluation check | Qwen2.5-3B seed 43 was re-evaluated with Unsloth 2026.11.2, Transformers 5.17.0, PEFT 0.21.2 and PyTorch for CUDA 13.0. Generations, per-item NLL values and metrics were identical. |

## Metrics and re-analysis

| Item | Detail |
|---|---|
| BLEU signature | `nrefs:1\|case:mixed\|eff:no\|tok:13a\|smooth:exp\|version:2.6.0` |
| chrF++ signature | `nrefs:1\|case:mixed\|eff:yes\|nc:6\|nw:2\|space:no\|version:2.6.0` |
| ROUGE | rouge-score 0.1.2, no stemming |
| BERTScore | bert-score 0.3.13, `bert-base-multilingual-cased`, no baseline rescaling |
| Evaluation-script intervals | 200 resamples, Python `random` seed 0, re-initialized for each subset |
| Re-analysis | `analysis/run_level_analysis.py` (sacreBLEU 2.6.0, rouge-score 0.1.2, NumPy `default_rng(0)`, 2,000 resamples), applied to the per-item outputs of the 19 runs whose outputs were retained. It reproduces every reported chrF++, BLEU, ROUGE and BPB value. |
