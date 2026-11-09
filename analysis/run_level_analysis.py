"""Run-level re-analysis from the saved per-item outputs (generation, reference, NLL, tokens, bytes).

For every fine-tuned and zero-shot run it recomputes BPB, chrF++, BLEU and ROUGE, and then:
  * per-run 95% percentile bootstrap intervals (2,000 resamples over test items, seed 0);
  * paired item-level bootstrap of differences between models for chrF++ and BLEU, averaging the
    three seeds of each model inside every replicate (2,000 resamples, seed 0, same items for all runs);
  * paired item-level bootstrap of BPB differences with per-item NLL averaged over seeds;
  * per-type (A/B) scores, output lengths and empty outputs;
  * zero-shot BPB under both prompts;
  * seed-level Welch intervals for the same model differences;
  * gains over the zero-shot baselines (fine-tuned three-seed mean minus zero-shot run) and BPB ratios.
BERTScore is not recomputed (the mBERT weights are not available in this environment).

Input: one JSONL file per run (ft_<model>_s<seed>.jsonl, zs_<model>_<prompt>.jsonl) with the fields
id, type, reference, generation, nll_nats, resp_tokens and resp_bytes. In the code repository these
files are made by scripts/restore_references.py, which adds the references to results/per_item/.

Usage: python run_level_analysis.py <runs_dir> <out.json>
"""
import json, math, sys
from pathlib import Path
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer

RUNS_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODELS = {"qwen": "Qwen2.5-3B", "llama": "Llama-3.2-3B", "gemma": "Gemma-2-2B", "llama8b": "Llama-3.1-8B"}
ZS_PROMPTS = {"qwen": ("alpaca", "bare"), "llama": ("alpaca", "bare"), "gemma": ("alpaca", "bare"), "llama8b": ("alpaca",)}  # 8B bare outputs with NLL were not available
SEEDS = (42, 43, 44)
B, SEED = 2000, 0
LN2 = math.log(2)

def load(name):
    rows = [json.loads(l) for l in open(RUNS_DIR / name, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    return rows

runs = {}
for m in MODELS:
    for s in SEEDS:
        runs[f"ft_{m}_s{s}"] = load(f"ft_{m}_s{s}.jsonl")
    for p in ZS_PROMPTS[m]:
        runs[f"zs_{m}_{p}"] = load(f"zs_{m}_{p}.jsonl")

# alignment: same ids, types and references in every run
ref0 = runs["ft_qwen_s42"]
ids = [r["id"] for r in ref0]
assert ids == list(range(1000)), "ids are not 0..999"
for k, rows in runs.items():
    assert [r["id"] for r in rows] == ids, k
    assert all(a["reference"] == b["reference"] and a["type"] == b["type"] for a, b in zip(rows, ref0)), k
refs = [r["reference"] for r in ref0]
types = np.array([r["type"] for r in ref0])

bleu, chrf = BLEU(), CHRF(word_order=2)
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def stats(metric, hyps, rr):
    return np.array(metric._extract_corpus_statistics(hyps, [rr]), dtype=np.int64)

def score(metric, summed):
    return metric._compute_score_from_stats([int(x) for x in summed]).score

rng = np.random.default_rng(SEED)
IDX = rng.integers(0, 1000, size=(B, 1000))
COUNTS = np.zeros((B, 1000), dtype=np.int64)
for b in range(B):
    COUNTS[b] = np.bincount(IDX[b], minlength=1000)

per_run, boot = {}, {}
for k, rows in runs.items():
    hyps = [r["generation"] for r in rows]
    nll = np.array([r["nll_nats"] for r in rows])
    toks = np.array([r["resp_tokens"] for r in rows])
    byts = np.array([r["resp_bytes"] for r in rows])
    S_b, S_c = stats(bleu, hyps, refs), stats(chrf, hyps, refs)
    rouge = [scorer.score(rf, h) for rf, h in zip(refs, hyps)]
    d = {
        "bpb": float(nll.sum() / (LN2 * byts.sum())),
        "token_ppl": float(math.exp(nll.sum() / toks.sum())),
        "chrf": score(chrf, S_c.sum(0)),
        "bleu": score(bleu, S_b.sum(0)),
        "rouge1": float(np.mean([x["rouge1"].fmeasure for x in rouge])),
        "rouge2": float(np.mean([x["rouge2"].fmeasure for x in rouge])),
        "rougeL": float(np.mean([x["rougeL"].fmeasure for x in rouge])),
        "empty": int(sum(1 for h in hyps if not h.strip())),
        "gen_ref_char_ratio": float(sum(len(h) for h in hyps) / sum(len(r) for r in refs)),
        "gen_ref_word_ratio": float(sum(len(h.split()) for h in hyps) / sum(len(r.split()) for r in refs)),
        "refs_over_256_tokens": int((toks > 256).sum()),
        "refs_over_1000_tokens": int((toks > 1000).sum()),
    }
    for t in ("A", "B"):
        m = types == t
        d[f"chrf_{t}"] = score(chrf, S_c[m].sum(0))
        d[f"bleu_{t}"] = score(bleu, S_b[m].sum(0))
        d[f"bpb_{t}"] = float(nll[m].sum() / (LN2 * byts[m].sum()))
    # per-run bootstrap
    Sc_b, Sb_b = COUNTS @ S_c, COUNTS @ S_b
    bpb_b = (COUNTS @ nll) / (LN2 * (COUNTS @ byts))
    cs = np.array([score(chrf, r) for r in Sc_b]); bs = np.array([score(bleu, r) for r in Sb_b])
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    d["chrf_ci"], d["bleu_ci"], d["bpb_ci"] = q(cs), q(bs), q(bpb_b)
    per_run[k] = d
    boot[k] = {"chrf": cs, "bleu": bs, "nll": nll, "bytes": byts, "chrf_seg": S_c, "bleu_seg": S_b}

def seed_mean_sd(m, key):
    v = [per_run[f"ft_{m}_s{s}"][key] for s in SEEDS]
    return {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}

summary = {MODELS[m]: {key: seed_mean_sd(m, key) for key in
           ("chrf", "bleu", "rouge1", "rouge2", "rougeL", "bpb", "chrf_A", "chrf_B", "bpb_A", "bpb_B",
            "gen_ref_char_ratio", "gen_ref_word_ratio")} for m in MODELS}

# paired item-level bootstrap between models (seed-averaged inside each replicate)
pairs = [("llama", "qwen"), ("gemma", "qwen"), ("gemma", "llama"), ("llama8b", "llama")]
paired = {}
for a, b in pairs:
    res = {}
    for met in ("chrf", "bleu"):
        A = np.mean([boot[f"ft_{a}_s{s}"][met] for s in SEEDS], axis=0)
        Bm = np.mean([boot[f"ft_{b}_s{s}"][met] for s in SEEDS], axis=0)
        point = np.mean([per_run[f"ft_{a}_s{s}"][met] for s in SEEDS]) - np.mean([per_run[f"ft_{b}_s{s}"][met] for s in SEEDS])
        diff = A - Bm
        res[met] = {"diff": float(point), "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                    "share_replicates_a_higher": float((diff > 0).mean())}
    nll_a = np.mean([boot[f"ft_{a}_s{s}"]["nll"] for s in SEEDS], axis=0)
    nll_b = np.mean([boot[f"ft_{b}_s{s}"]["nll"] for s in SEEDS], axis=0)
    by_a, by_b = boot[f"ft_{a}_s42"]["bytes"], boot[f"ft_{b}_s42"]["bytes"]  # each model's scored bytes (item 289 differs)
    bpb_a = (COUNTS @ nll_a) / (LN2 * (COUNTS @ by_a)); bpb_b = (COUNTS @ nll_b) / (LN2 * (COUNTS @ by_b))
    pt = nll_a.sum() / (LN2 * by_a.sum()) - nll_b.sum() / (LN2 * by_b.sum())
    dd = bpb_a - bpb_b
    res["bpb"] = {"diff": float(pt), "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    paired[f"{MODELS[a]} - {MODELS[b]}"] = res

# zero-shot: paired BPB across models within a prompt
zs_bpb = {}
for p in ("alpaca", "bare"):
    for a, b in pairs:
        if f"zs_{a}_{p}" not in boot or f"zs_{b}_{p}" not in boot:
            continue
        na, nb = boot[f"zs_{a}_{p}"]["nll"], boot[f"zs_{b}_{p}"]["nll"]
        ba, bb = boot[f"zs_{a}_{p}"]["bytes"], boot[f"zs_{b}_{p}"]["bytes"]
        dd = (COUNTS @ na) / (LN2 * (COUNTS @ ba)) - (COUNTS @ nb) / (LN2 * (COUNTS @ bb))
        zs_bpb[f"{p}: {MODELS[a]} - {MODELS[b]}"] = {"diff": float(na.sum() / (LN2 * ba.sum()) - nb.sum() / (LN2 * bb.sum())),
                                                    "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}

out = {
    "signatures": {"bleu": str(bleu.get_signature()), "chrf": str(chrf.get_signature())},
    "bootstrap": {"resamples": B, "seed": SEED, "rng": "numpy default_rng"},
    "per_run": per_run, "fine_tuned_summary": summary, "paired_fine_tuned": paired, "paired_zero_shot_bpb": zs_bpb,
}
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if not kk.endswith("_ci")} for k, v in per_run.items()}, indent=0)[:6000])
print(json.dumps(paired, indent=1))
print(json.dumps(zs_bpb, indent=1))
print(out["signatures"])

# ---- seed-level Welch intervals from the exact per-seed values (appended analysis)
from scipy import stats as _st
def _welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = _st.t.ppf(0.975, df); d = a.mean() - b.mean()
    return {"diff": float(d), "ci": [float(d - t * se), float(d + t * se)], "df": float(df)}
seed_level = {}
for a, b in pairs:
    seed_level[f"{MODELS[a]} - {MODELS[b]}"] = {met: _welch([per_run[f"ft_{a}_s{s}"][met] for s in SEEDS], [per_run[f"ft_{b}_s{s}"][met] for s in SEEDS])
                                              for met in ("chrf", "bleu", "bpb")}
out["seed_level_welch"] = seed_level
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(seed_level, indent=1))

# ---- gains over the zero-shot baselines (appended analysis)
# The Llama-3.1-8B bare-prompt outputs were not retained; its zero-shot values are those of its
# metric file as reported in the paper (chrF++ 13.42, BLEU 0.40, BPB 2.082).
ZS_NOT_RETAINED = {("llama8b", "bare"): {"chrf": 13.42, "bleu": 0.40, "bpb": 2.082}}
gains = {}
for m in MODELS:
    ft = {k: float(np.mean([per_run[f"ft_{m}_s{s}"][k] for s in SEEDS])) for k in ("chrf", "bleu", "bpb")}
    g = {}
    for p in ("alpaca", "bare"):
        zs = per_run.get(f"zs_{m}_{p}") or ZS_NOT_RETAINED[(m, p)]
        g[f"chrf_gain_{p}"] = ft["chrf"] - zs["chrf"]
        g[f"bleu_gain_{p}"] = ft["bleu"] - zs["bleu"]
        g[f"bpb_ratio_{p}"] = ft["bpb"] / zs["bpb"]
    gains[MODELS[m]] = g
out["gains_over_zero_shot"] = gains
out["gains_note"] = "Llama-3.1-8B bare-prompt zero-shot values are the reported, rounded values of a run whose outputs were not retained."
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(gains, indent=1))
