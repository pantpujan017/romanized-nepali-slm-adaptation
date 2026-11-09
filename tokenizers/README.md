# Tokenizer files

`scripts/fetch_public_inputs.sh` downloads the four tokenizer files used in the fertility analysis and checks their SHA-256 hashes. The files are not tracked in git.

| File | Source | SHA-256 |
|---|---|---|
| `llama3_tokenizer.model` | meta-llama/llama-models @ `0e0b8c51`, `models/llama3/tokenizer.model` | `82e9d319…` |
| `gemma_tokenizer.model` | google/gemma_pytorch @ `014acb7a`, `tokenizer/tokenizer.model` | `61a7b147…` |
| `qwen.tiktoken` | dashscope 2.0.4 wheel, `dashscope/legacy_1x/tokenizers/qwen.tiktoken` | `b2b1b8df…` |
| `mistral_tokenizer.model.v1` | mistral-common 1.13.1 wheel, `mistral_common/data/tokenizer.model.v1` | `dadfd56d…` |

The fertility values on the benchmark references (Table 3 of the paper) were measured with the models' own Hugging Face tokenizers. The files above rebuild the same tokenizers for the public-sample analysis.
