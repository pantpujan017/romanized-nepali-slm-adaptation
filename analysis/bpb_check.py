"""Bits per byte (BPB) from the released per-item outputs, without the reference texts.

Reads results/per_item/<run>.jsonl.gz (fields id, type, generation, nll_nats, resp_tokens,
resp_bytes) and recomputes, for every run, BPB = sum(nll_nats) / (ln 2 * sum(resp_bytes)), the
three-seed mean and sample SD of the fine-tuned models, and the paired item-level 95% intervals of
the BPB differences between models (2,000 resamples of the 1,000 test items, numpy default_rng
seed 0, per-item NLL averaged over seeds). The resampling is identical to run_level_analysis.py,
so the values match those in results/run_level_results.json. chrF++, BLEU and ROUGE need the
references; see scripts/restore_references.py.

Usage: python analysis/bpb_check.py results/per_item [results/run_level_results.json]
"""
import gzip, json, math, sys
from pathlib import Path
import numpy as np

PER_ITEM = Path(sys.argv[1])
CHECK = Path(sys.argv[2]) if len(sys.argv) > 2 else None
MODELS = {"qwen": "Qwen2.5-3B", "llama": "Llama-3.2-3B", "gemma": "Gemma-2-2B", "llama8b": "Llama-3.1-8B"}
SEEDS = (42, 43, 44)
PAIRS = [("llama", "qwen"), ("gemma", "qwen"), ("gemma", "llama"), ("llama8b", "llama")]
LN2 = math.log(2)


def load(run):
    with gzip.open(PER_ITEM / f"{run}.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = sorted((json.loads(l) for l in fh), key=lambda r: r["id"])
    assert [r["id"] for r in rows] == list(range(1000)), run
    return np.array([r["nll_nats"] for r in rows]), np.array([r["resp_bytes"] for r in rows])


runs = {p.name[: -len(".jsonl.gz")]: load(p.name[: -len(".jsonl.gz")]) for p in sorted(PER_ITEM.glob("*.jsonl.gz"))}
bpb = {k: float(n.sum() / (LN2 * b.sum())) for k, (n, b) in runs.items()}

rng = np.random.default_rng(0)
IDX = rng.integers(0, 1000, size=(2000, 1000))
COUNTS = np.stack([np.bincount(row, minlength=1000) for row in IDX])

summary = {}
for m, name in MODELS.items():
    v = [bpb[f"ft_{m}_s{s}"] for s in SEEDS]
    summary[name] = {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}

paired = {}
for a, b in PAIRS:
    nll_a = np.mean([runs[f"ft_{a}_s{s}"][0] for s in SEEDS], axis=0)
    nll_b = np.mean([runs[f"ft_{b}_s{s}"][0] for s in SEEDS], axis=0)
    by_a, by_b = runs[f"ft_{a}_s42"][1], runs[f"ft_{b}_s42"][1]  # each model's own scored bytes
    d = (COUNTS @ nll_a) / (LN2 * (COUNTS @ by_a)) - (COUNTS @ nll_b) / (LN2 * (COUNTS @ by_b))
    point = nll_a.sum() / (LN2 * by_a.sum()) - nll_b.sum() / (LN2 * by_b.sum())
    paired[f"{MODELS[a]} - {MODELS[b]}"] = {"diff": float(point),
                                          "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

out = {"per_run_bpb": bpb, "fine_tuned_bpb": summary, "paired_bpb": paired}
print(json.dumps(out, indent=1))

if CHECK:
    ref = json.load(open(CHECK))
    worst = max(abs(bpb[k] - ref["per_run"][k]["bpb"]) for k in bpb)
    worst = max(worst, *(abs(paired[k]["diff"] - ref["paired_fine_tuned"][k]["bpb"]["diff"]) for k in paired))
    worst = max(worst, *(abs(x - y) for k in paired for x, y in zip(paired[k]["ci"], ref["paired_fine_tuned"][k]["bpb"]["ci"])))
    print(f"largest absolute difference from {CHECK.name}: {worst:.2e}")
    sys.exit(0 if worst < 1e-9 else 1)
