#!/usr/bin/env bash
# Fetch the public tokenizer files and Nepali text used by analysis/tokenizer_romanization_analysis.py,
# pinned to the exact versions used in the paper, and verify their SHA-256 checksums.
# Requires: curl, unzip, python -m pip.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tokenizers data/public
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

fetch() {  # url dest
  echo "fetching $2"
  curl -fsSL "$1" -o "$2"
}

# Llama 3 BPE ranks: meta-llama/llama-models @ 0e0b8c51
fetch https://raw.githubusercontent.com/meta-llama/llama-models/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/tokenizer.model \
  tokenizers/llama3_tokenizer.model
# Gemma SentencePiece model (256k vocabulary): google/gemma_pytorch @ 014acb7a
fetch https://raw.githubusercontent.com/google/gemma_pytorch/014acb7ac4563a5f77c76d7ff98f31b568c16508/tokenizer/tokenizer.model \
  tokenizers/gemma_tokenizer.model
# Qwen BPE ranks (151,643 tokens; the vocabulary shared by Qwen, Qwen2 and Qwen2.5): dashscope 2.0.4 wheel
python -m pip download --quiet --no-deps --dest "$TMP" dashscope==2.0.4
unzip -q -o -j "$TMP"/dashscope-2.0.4-*.whl dashscope/legacy_1x/tokenizers/qwen.tiktoken -d tokenizers
# Mistral v0.1 SentencePiece model (32k vocabulary): mistral-common 1.13.1 wheel
python -m pip download --quiet --no-deps --dest "$TMP" mistral-common==1.13.1
unzip -q -o -j "$TMP"/mistral_common-1.13.1-*.whl mistral_common/data/tokenizer.model.v1 -d "$TMP"
mv "$TMP"/tokenizer.model.v1 tokenizers/mistral_tokenizer.model.v1
# Mozilla Common Voice Nepali text corpus (CC0): common-voice/common-voice @ cbec2f46
fetch https://raw.githubusercontent.com/common-voice/common-voice/cbec2f469ff3bb6321349ec87172332ef9603234/server/data/ne-NP/sentence-collector.txt \
  data/public/cv_ne_sentence_collector.txt
# Universal Declaration of Human Rights in Nepali, NLTK udhr2 corpus
fetch https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr2.zip "$TMP"/udhr2.zip
unzip -q -o -j "$TMP"/udhr2.zip udhr2/nep.txt -d "$TMP"
mv "$TMP"/nep.txt data/public/udhr2_nep.txt

sha256sum -c - <<'SUMS'
82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55  tokenizers/llama3_tokenizer.model
61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2  tokenizers/gemma_tokenizer.model
b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186  tokenizers/qwen.tiktoken
dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055  tokenizers/mistral_tokenizer.model.v1
9bf231cc194355e9bd7511709dab991e9e8c3de2e7c342fde7ac23583ddc40b9  data/public/cv_ne_sentence_collector.txt
ec0a940b22ced5c52d770dd7c3f87564e9d0d173b6dbd5810cfbb5900265f948  data/public/udhr2_nep.txt
SUMS
echo "All inputs fetched and verified."
