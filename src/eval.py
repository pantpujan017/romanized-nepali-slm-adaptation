# eval.py v2.2 (adds scoring cap, early save of generations, per-item NLL) - one evaluation script for zero-shot and fine-tuned models (identical for every model).
# Run in a Kaggle notebook (GPU, Internet on):

# Edit CONFIG, then %run eval.py. Writes outputs.jsonl and metrics.json into OUT_DIR.
import json, math, os, random, re
import numpy as np
import torch

# ---------------- CONFIG ----------------
MODE = "finetuned"                       # "zeroshot" (bare instruction, base weights) or "finetuned"
MODEL_ID = "Qwen/Qwen2.5-3B"     # base model id (used for zeroshot)
ADAPTER_DIR = "/kaggle/input/datasets/pujanpant017/adaptaer-all/adapter_001/adapter_001"   # used when MODE == "finetuned"
DATA_FILE = "/kaggle/input/datasets/pujanpant017/dataste001/dataset_corrected/test.jsonl"
OUT_DIR = "/kaggle/working/eval_seed42"
N_ITEMS = None                           # None = all 1,000; set e.g. 500 only if the pilot says time is short
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
from unsloth import FastLanguageModel

ALPACA = ("Below is an instruction that describes a task, paired with an input that provides further context. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
ALPACA_NO_INPUT = ("Below is an instruction that describes a task. "
                   "Write a response that appropriately completes the request.\n\n"
                   "### Instruction:\n{instruction}\n\n### Response:\n")

def build_prompt(r):
    if MODE == "zeroshot":   # Rimal and Rimal: no system prompt, template or few-shot examples
        return r["instruction"] + ("\n" + r["input"] if r["input"].strip() else "")
    return (ALPACA if r["input"].strip() else ALPACA_NO_INPUT).format(**r)

rows = [json.loads(l) for l in open(DATA_FILE)]
if N_ITEMS:
    random.Random(42).shuffle(rows); rows = rows[:N_ITEMS]

load_name = ADAPTER_DIR if MODE == "finetuned" else MODEL_ID   # adapter on the 4-bit base, never merged
model, tok = FastLanguageModel.from_pretrained(model_name=load_name, max_seq_length=4096,
                                               dtype=torch.float16, load_in_4bit=True)
FastLanguageModel.for_inference(model)
print("model.max_seq_length:", getattr(model, "max_seq_length", "n/a"), flush=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

# ---- 1. greedy generation, identical for both stages ----
prompts = [build_prompt(r) for r in rows]
n_long_prompts = sum(len(tok(p, add_special_tokens=True)["input_ids"]) > 512 for p in prompts)
print("prompts longer than 512 tokens (truncated for generation):", n_long_prompts, flush=True)
gens = []
for i in range(0, len(prompts), 8):
    batch = tok(prompts[i:i+8], return_tensors="pt", padding=True, truncation=True,
                max_length=512, add_special_tokens=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=256, do_sample=False, repetition_penalty=1.1,
                             pad_token_id=tok.pad_token_id)
    gens += tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"generated {len(gens)}/{len(prompts)}", flush=True)
gens = [g.strip() for g in gens]
with open(f"{OUT_DIR}/generations_raw.jsonl", "w") as f:       # saved at once, so a later crash cannot lose them
    for i, g in enumerate(gens):
        f.write(json.dumps({"id": i, "generation": g}, ensure_ascii=False) + "\n")

# ---- 2. NLL of the reference response given the prompt (response tokens only, EOS excluded) ----
NLL_CAP = 1000   # max prompt+response tokens scored; Unsloth inference breaks above 1,024 tokens
nll_nats, ntok, nbytes, n_cut = [], [], [], 0
for r, p in zip(rows, prompts):
    p_ids = tok(p, add_special_tokens=True)["input_ids"][:512]
    r_ids = tok(r["output"], add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(r_ids) > NLL_CAP:                          # score only the part that fits; count its bytes
        r_ids = r_ids[:NLL_CAP - len(p_ids)]; n_cut += 1
        nb = len(tok.decode(r_ids, skip_special_tokens=True).encode("utf-8"))
    else:
        nb = len(r["output"].encode("utf-8"))
    ids = torch.tensor([p_ids + r_ids]).to("cuda")
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    tok_lp = lp[torch.arange(len(tgt)), tgt][len(p_ids) - 1:]      # response positions only
    nll_nats.append(-tok_lp.sum().item()); ntok.append(len(r_ids))
    nbytes.append(nb)

def lm_stats(idx):
    n = sum(nll_nats[i] for i in idx)
    return {"bits_per_byte": n / math.log(2) / sum(nbytes[i] for i in idx),
            "token_ppl": math.exp(n / sum(ntok[i] for i in idx))}

# ---- 3. generation metrics ----
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bscore
refs = [r["output"] for r in rows]
rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
rouge_items = [rs.score(a, b) for a, b in zip(refs, gens)]
_, _, f1 = bscore(gens, refs, model_type="bert-base-multilingual-cased", rescale_with_baseline=False)
bert_items = f1.numpy()

def corpus_scores(idx):
    h = [gens[i] for i in idx]; r = [refs[i] for i in idx]
    return {"bleu": sacrebleu.corpus_bleu(h, [r]).score,
            "chrfpp": sacrebleu.corpus_chrf(h, [r], word_order=2).score,
            "rouge1": float(np.mean([rouge_items[i]["rouge1"].fmeasure for i in idx])),
            "rouge2": float(np.mean([rouge_items[i]["rouge2"].fmeasure for i in idx])),
            "rougeL": float(np.mean([rouge_items[i]["rougeL"].fmeasure for i in idx])),
            "bertscore_f1": float(np.mean([bert_items[i] for i in idx])),
            **lm_stats(idx)}

def with_ci(idx, B=200):
    rng = random.Random(0)
    base = corpus_scores(idx)
    boots = [corpus_scores([rng.choice(idx) for _ in idx]) for _ in range(B)]
    return {k: {"value": v, "ci95": [float(np.percentile([b[k] for b in boots], 2.5)),
                                     float(np.percentile([b[k] for b in boots], 97.5))]}
            for k, v in base.items()}

# script check: share of outputs containing Devanagari characters
dev = [bool(re.search(r"[ऀ-ॿ]", g)) for g in gens]
groups = {"all": list(range(len(rows)))}
for t in sorted({r["type"] for r in rows}):
    groups[f"type_{t}"] = [i for i, r in enumerate(rows) if r["type"] == t]
import hashlib, importlib.metadata as im
try:
    script_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
except Exception:
    script_sha = "unknown (compute sha256sum eval.py by hand)"
metrics = {"mode": MODE, "model": MODEL_ID, "n": len(rows), "script_sha256": script_sha,
           "package_versions": {p: im.version(p) for p in ["unsloth", "transformers", "peft", "sacrebleu", "bert-score", "torch"]},
           "nll_items_truncated": n_cut, "prompts_over_512_tokens": n_long_prompts,
           "share_outputs_with_devanagari": float(np.mean(dev)),
           "scores": {g: with_ci(ix) for g, ix in groups.items()}}
json.dump(metrics, open(f"{OUT_DIR}/metrics.json", "w"), indent=1)
with open(f"{OUT_DIR}/outputs.jsonl", "w") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"id": i, "type": r["type"], "prompt": prompts[i], "reference": refs[i],
                            "generation": gens[i], "nll_nats": nll_nats[i],
                            "resp_tokens": ntok[i], "resp_bytes": nbytes[i]}, ensure_ascii=False) + "\n")
print(json.dumps({g: {k: round(v["value"], 4) for k, v in s.items()} for g, s in metrics["scores"].items()}, indent=1))
