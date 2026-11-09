# Data

## Source

The benchmark is built from the Nepali instruction dataset
[Saugatkafley/alpaca-nepali-sft](https://huggingface.co/datasets/Saugatkafley/alpaca-nepali-sft).
It has 52,005 examples with instruction, input and output fields in Devanagari.

Its dataset card states no licence and gives little provenance, but the data are a Nepali
translation of [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (CC BY-NC 4.0).
The English (type A) instructions of our test split, translated back from Nepali, match the
Alpaca instructions closely ([`analysis/alpaca_overlap.py`](../analysis/alpaca_overlap.py),
against `alpaca_data.json` at commit `761dc5bf`):

| Measure (500 type A test instructions) | Value |
|---|---:|
| Identical to one of the 52,002 Alpaca instructions | 29.8% |
| Similarity to the nearest Alpaca instruction ≥ 80 (rapidfuzz `fuzz.ratio`) | 92.2% |
| Similarity ≥ 60 | 100% |
| Median similarity | 95.1 |

Back-translation sometimes changes the meaning. One test instruction reads "Explain why
chloroplasts are found in plant cells and animal cells?", where Alpaca has "…in plant cells
and not animal cells?".

**We do not redistribute any text from the source dataset.** This repository releases the
model outputs and scripts, and identifies the split files by their SHA-256 hashes.

## Construction

1. Transliterate the Devanagari fields to ITRANS and lowercase them (the benchmark convention, R2).
2. Remove examples that still contain Devanagari after transliteration, and examples whose output has fewer than three words. This leaves 48,106 eligible examples. Examples with an empty input are kept.
3. Sample 10,500 examples (seed 42) and split them 9,000 / 500 / 1,000, balancing the two formats 50/50 in every split.
   - **Type A:** instruction machine-translated into English; input and output transliterated.
   - **Type B:** instruction, input and output all transliterated.

| Split | Examples | Empty input | "No input" placeholders | Zero-width spaces | SHA-256 |
|---|---:|---:|---:|---:|---|
| Train | 9,000 | 5,576 | 114 | 113 | `9ed86da7f1ff82f6f931ff7db817111f6e3b0db4d5c7cdd2265f653d55229f3f` |
| Validation | 500 | 314 | 5 | 11 | `3f7f685479b147a5b8f3826cc91408d326a84e70472810cf841a50aacc235485` |
| Test | 1,000 | 660 | 14 | 12 | `d656353e9ac233d73a81e9b0cc997c3ee8b4a66dcc96169a48150b7275254694` |

No instruction occurs in more than one split. One output is shared between train and
validation, and three between train and test, each of three or more words. They were kept.

## Length

Training sequences (Alpaca prompt plus response and EOS) were truncated at 512 tokens. Examples
longer than that, counted before truncation with each model's tokenizer
([`results/hf_tokenizer_audit/length_audit.json`](../results/hf_tokenizer_audit/length_audit.json)):

| Split | Qwen2.5-3B | Llama-3.2-3B | Gemma-2-2B |
|---|---:|---:|---:|
| Train (9,000) | 178 | 164 | 127 |
| Validation (500) | 6 | 6 | 5 |
| Test (1,000) | 19 | 17 | 14 |

No generation prompt exceeds 512 tokens. Of the 1,000 test references, 124 (Qwen2.5-3B),
117 (Llama-3.2-3B) and 88 (Gemma-2-2B) are longer than the 256-token generation limit.

## Files

| File | Content |
|---|---|
| `public/` | Public Nepali text for the fertility analysis, fetched by `scripts/fetch_public_inputs.sh` (not tracked) |

To recompute the overlap metrics from the released outputs you need the test split
(`test.jsonl`, SHA-256 above). `scripts/restore_references.py` adds its references to
`results/per_item/`; it checks the SHA-256 and the word and byte count of every reference
before writing anything.
