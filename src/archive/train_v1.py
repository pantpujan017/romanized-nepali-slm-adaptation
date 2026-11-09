# train.py - QLoRA fine-tuning for the Romanized Nepali rerun (identical for every model and seed).
# Run in a Kaggle notebook (GPU T4 x2, Internet on):
#   !pip -q install unsloth trl peft bitsandbytes datasets
#   then edit the CONFIG block below and run this file (%run train.py).
# Set GATE = True first: 50 steps, prints the checks, then stop and read the output.
import json, math, os, time
import torch

# ---------------- CONFIG (the only block you edit) ----------------
MODEL_ID = "meta-llama/Llama-3.2-3B"     # google/gemma-2-2b | Qwen/Qwen2.5-3B | meta-llama/Llama-3.2-3B
SEED = 42                                # 42, 43, 44
DATA_DIR = "/kaggle/input/YOUR_DATASET"  # holds train.jsonl and validation.jsonl
OUT_DIR = "/kaggle/working/run"          # adapter, logs and training_meta.json are written here
GATE = True                              # True = 50-step check only
# ------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
from unsloth import FastLanguageModel   # import unsloth before trl/transformers
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

ALPACA = ("Below is an instruction that describes a task, paired with an input that provides further context. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
ALPACA_NO_INPUT = ("Below is an instruction that describes a task. "
                   "Write a response that appropriately completes the request.\n\n"
                   "### Instruction:\n{instruction}\n\n### Response:\n")

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, max_seq_length=512, dtype=torch.float16, load_in_4bit=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64, lora_dropout=0.0, bias="none", use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f"TRAINABLE PARAMETERS: {n_train:,} (counted on the 4-bit model; {n_all:,} total as stored)")

def fmt(ex):
    p = (ALPACA if ex["input"].strip() else ALPACA_NO_INPUT).format(**ex)
    return {"text": p + ex["output"] + tok.eos_token}

data = load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl",
                                        "validation": f"{DATA_DIR}/validation.jsonl"})
data = data.map(fmt)

# --- sanity checks that must be read in the gate run ---
first = tok(data["train"][0]["text"])["input_ids"]
print("BOS id:", tok.bos_token_id, "| first token id:", first[0],
      "| starts with BOS:", (first[0] == tok.bos_token_id) if tok.bos_token_id is not None else "n/a (no BOS)")
print("Last 3 token ids of first example:", first[-3:], "| EOS id:", tok.eos_token_id,
      "(should end with exactly one EOS)")
print("Vocab size:", len(tok), "| uniform-guess loss ln(V) = %.2f" % math.log(len(tok)))

cfg = SFTConfig(
    output_dir=OUT_DIR, dataset_text_field="text", max_seq_length=512, packing=False,
    per_device_train_batch_size=2, gradient_accumulation_steps=4, per_device_eval_batch_size=2,
    num_train_epochs=3, max_steps=50 if GATE else -1,
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=200, optim="adamw_8bit",
    fp16=True, bf16=False, seed=SEED, data_seed=SEED,
    eval_strategy="steps", eval_steps=100 if not GATE else 25,
    save_strategy="steps", save_steps=100 if not GATE else 25, save_total_limit=10,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    logging_steps=10, report_to="none")
# NOTE: in some TRL versions max_seq_length is called max_length. Rename it if you get an error.

trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=data["train"],
                     eval_dataset=data["validation"], args=cfg)

step0 = trainer.evaluate()["eval_loss"]
print(f"STEP-0 EVAL LOSS: {step0:.3f}  (must be well below ln(V) above; the old Gemma run showed 22.6)")

t0 = time.time()
trainer.train()
secs = time.time() - t0
steps = trainer.state.global_step
print(f"SECONDS PER STEP: {secs/steps:.2f}")

trainer.model.save_pretrained(f"{OUT_DIR}/adapter")
tok.save_pretrained(f"{OUT_DIR}/adapter")
meta = {"model": MODEL_ID, "seed": SEED, "gate": GATE, "trainable_params": n_train,
        "effective_batch": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
        "num_gpus_visible": torch.cuda.device_count(), "world_size": trainer.args.world_size,
        "global_steps": steps, "seconds_per_step": secs / steps, "step0_eval_loss": step0,
        "best_step": trainer.state.best_global_step, "best_eval_loss": trainer.state.best_metric,
        "log_history": trainer.state.log_history}
json.dump(meta, open(f"{OUT_DIR}/training_meta.json", "w"), indent=1)
print("DONE. Best step:", meta["best_step"], "best val loss:", meta["best_eval_loss"],
      "| steps:", steps, "| effective batch:", meta["effective_batch"], "| world size:", meta["world_size"])
