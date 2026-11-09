# Run records

These are the launch records, metric files, recovery records, one log and one notebook written by
the training and evaluation runs. The JSON records and the log are kept as written, with one edit:
the Kaggle path of the input dataset used in the second Kaggle account is replaced by
`/kaggle/input/<dataset>/`. The notebook is unchanged.

Metric files hold the evaluation scripts' own results. Their intervals use 200 bootstrap
resamples (Python `random`, seed 0, re-initialized for each subset). The paper reports the
2,000-resample intervals of `analysis/run_level_analysis.py` instead.

| File | Run | Content |
|---|---|---|
| `qwen_runs_notebook.ipynb` | Qwen2.5-3B, fine-tuned | Kaggle notebook with saved outputs: the excluded second seed-42 training run and its validation-loss history, the evaluation of the seed-42 run that is reported, `train.py` v2 and `eval.py` v2.2 as written to disk, and the parallel training and evaluation of seeds 43 and 44 (selected step 2200 in both) |
| `ft_llama8b_s42_metrics.json` | Llama-3.1-8B, seed 42 | Evaluation metrics; script `f34da070…` |
| `ft_llama8b_s43_metrics.json` | Llama-3.1-8B, seed 43 | Evaluation metrics; script `12c1f7c3…` |
| `ft_llama8b_s44_metrics.json` | Llama-3.1-8B, seed 44 | Evaluation metrics; script `c288f295…` |
| `zs_qwen_alpaca_metrics.json` | Qwen2.5-3B, zero-shot, Alpaca prompt | Evaluation metrics (script hash not recorded) |
| `zs_qwen_alpaca_metrics_recovered.json`, `zs_qwen_alpaca_recovery.json` | Qwen2.5-3B, zero-shot, Alpaca prompt | Validation of `recover_metrics.py`: same BLEU, chrF++, ROUGE and BPB; BERTScore within 1e-7 |
| `zs_qwen_bare_launch.json` | Qwen2.5-3B, zero-shot, bare prompt | Launch record: patched script `618c89fb…` from `33e10eac…`, started 2026-11-06 02:39:01 UTC |
| `zs_qwen_bare_metrics.json` | Qwen2.5-3B, zero-shot, bare prompt | Evaluation metrics |
| `zs_qwen_bare_recovery_first_pass.json`, `zs_qwen_bare_recovery.json`, `zs_qwen_bare_metrics_recovered.json` | Qwen2.5-3B, zero-shot, bare prompt | Metric recovery (91 empty generations) |
| `zs_llama_alpaca_launch.json` | Llama-3.2-3B, zero-shot, Alpaca prompt | Launch record: patched script `11191d0c…`, started 2026-11-07 01:25:20 UTC |
| `zs_llama_bare_launch.json` | Llama-3.2-3B, zero-shot, bare prompt | Launch record: patched script `c8a28578…`, started 2026-11-06 12:51:30 UTC |
| `zs_llama_bare.log` | Llama-3.2-3B, zero-shot, bare prompt | Run log, ending with the BERTScore error that led to metric recovery |
| `zs_llama_bare_recovery.json`, `zs_llama_bare_metrics_recovered.json` | Llama-3.2-3B, zero-shot, bare prompt | Metric recovery (9 empty generations); its launch record gives 2026-11-06 08:02:48 UTC |
| `zs_llama8b_alpaca_recovery.json`, `zs_llama8b_alpaca_metrics_recovered.json` | Llama-3.1-8B, zero-shot, Alpaca prompt | Metric recovery (141 empty generations); patched script `a587b947…` from `51e1bd41…` |

The two start times recorded for the Llama-3.2-3B bare-prompt run (launch file and recovery
record) differ; both are reported in Appendix A of the paper.
