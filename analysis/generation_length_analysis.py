"""Length of the generated outputs in each model's own tokens.

For every run: the mean number of generated tokens, the number of generations with at least 250
tokens (the generation limit is 256 new tokens; an output of 250 or more tokens has almost
certainly been cut off), and the mean reference length in scored tokens (resp_tokens).
Generations are encoded without special tokens with the rebuilt tokenizers in tokenizers/
(Llama 3 BPE for Llama-3.2-3B and Llama-3.1-8B, Qwen BPE for Qwen2.5-3B, Gemma SentencePiece
for Gemma-2-2B), which reproduce the Hugging Face token counts (Appendix B of the paper).

Usage: python analysis/generation_length_analysis.py tokenizers results/per_item results/generation_length_results.json
"""
import gzip, json, sys
from pathlib import Path
import numpy as np
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import sentencepiece as spm

TOK_DIR, PER_ITEM, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LLAMA_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN_PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
llama = tiktoken.Encoding(name="llama3", pat_str=LLAMA_PAT, mergeable_ranks=load_tiktoken_bpe(str(TOK_DIR / "llama3_tokenizer.model")), special_tokens={})
qwen = tiktoken.Encoding(name="qwen", pat_str=QWEN_PAT, mergeable_ranks=load_tiktoken_bpe(str(TOK_DIR / "qwen.tiktoken")), special_tokens={})
gemma = spm.SentencePieceProcessor(model_file=str(TOK_DIR / "gemma_tokenizer.model"))
COUNT = {
    "llama": lambda s: len(llama.encode_ordinary(s)),
    "llama8b": lambda s: len(llama.encode_ordinary(s)),
    "qwen": lambda s: len(qwen.encode_ordinary(s)),
    # Gemma's SentencePiece model adds no dummy prefix (add_dummy_prefix is false)
    "gemma": lambda s: len(gemma.encode(s)),
}

out = {}
for f in sorted(PER_ITEM.glob("*.jsonl.gz")):
    run = f.name[: -len(".jsonl.gz")]
    parts = run.split("_")
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    n = np.array([COUNT[parts[1]](r["generation"]) for r in rows])
    out[run] = {"gens_ge_250_tokens": int((n >= 250).sum()),
                "mean_gen_tokens": float(n.mean()),
                "mean_ref_tokens": float(np.mean([r["resp_tokens"] for r in rows]))}
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
