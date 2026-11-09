"""Parameter counts and the precision of a three-seed standard deviation.

Outputs:
  1. The 95% interval for a true SD given a sample SD with n = 3 seeds (Section 4.6 of the paper).
  2. Total, embedding and non-embedding parameters from the published model configurations, and
     the rsLoRA (r = 32) trainable-parameter counts, checked against the counts in the paper (Table 2).
Gains over zero-shot and all model differences are computed from the per-item outputs by
analysis/run_level_analysis.py.

Usage: python analysis/derived_analyses.py [out.json]
"""
import json, math, sys
from scipy import stats

N_SEEDS = 3
out = {}

# 1. 95% interval for the true SD, n = 3: multipliers of the sample SD
df = N_SEEDS - 1
out["sd_ci_multiplier"] = [round(math.sqrt(df / stats.chi2.ppf(0.975, df)), 3), round(math.sqrt(df / stats.chi2.ppf(0.025, df)), 3)]

# 2. Parameter counts from published configurations
CFG = {
    "Qwen2.5-3B":   dict(h=2048, ff=11008, L=36, nq=16, nkv=2, hd=128, vocab_rows=151936, tied=True, qkv_bias=True, norms=2),
    "Llama-3.2-3B": dict(h=3072, ff=8192, L=28, nq=24, nkv=8, hd=128, vocab_rows=128256, tied=True, qkv_bias=False, norms=2),
    "Gemma-2-2B":   dict(h=2304, ff=9216, L=26, nq=8, nkv=4, hd=256, embed_params=590_118_912, tied=True, qkv_bias=False, norms=4),
    "Llama-3.1-8B": dict(h=4096, ff=14336, L=32, nq=32, nkv=8, hd=128, vocab_rows=128256, tied=False, qkv_bias=False, norms=2),
}
REPORTED_TRAINABLE = {"Qwen2.5-3B": 59.9e6, "Llama-3.2-3B": 48.6e6, "Gemma-2-2B": 41.5e6, "Llama-3.1-8B": 83.9e6}
R = 32
params = {}
for m, c in CFG.items():
    h, ff, q, kv = c["h"], c["ff"], c["nq"] * c["hd"], c["nkv"] * c["hd"]
    attn = h * q + 2 * h * kv + q * h + (q + 2 * kv if c["qkv_bias"] else 0)
    mlp = 3 * h * ff
    layer = attn + mlp + c["norms"] * h
    non_emb = c["L"] * layer + h  # final norm
    emb = c.get("embed_params") or c["vocab_rows"] * h * (1 if c["tied"] else 2)
    lora = c["L"] * R * ((h + q) + 2 * (h + kv) + (q + h) + 3 * (h + ff))
    params[m] = {"total": non_emb + emb, "non_embedding": non_emb, "embedding": emb,
                 "embedding_share": round(emb / (non_emb + emb), 3), "lora_r32": lora,
                 "lora_matches_reported": abs(lora - REPORTED_TRAINABLE[m]) < 0.05e6,
                 "lora_share_of_total": round(lora / (non_emb + emb), 4)}
out["parameters"] = params

if __name__ == "__main__":
    text = json.dumps(out, indent=1)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(text)
    print(text)
