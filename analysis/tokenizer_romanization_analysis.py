"""Usage: python analysis/tokenizer_romanization_analysis.py <repo root> <out.json>

Tokenizer fertility of four model tokenizers on Romanized Nepali under three romanization conventions.

Corpus: Mozilla Common Voice Nepali sentence collector (CC0) + UDHR in Nepali (NLTK udhr2).
Conventions: R1 ITRANS (case-preserving); R2 lowercased ITRANS (the benchmark's scheme);
R3 R2 with the word-final inherent vowel removed in polysyllabic words (a heuristic for
common informal romanization, e.g. nepala -> nepal, ghara -> ghar; monosyllables such as ma, ra, cha keep it).
Tokenizers: Llama 3 (tiktoken BPE, llama-models tokenizer.model), Qwen (qwen.tiktoken, shared by Qwen2.5),
Gemma (SentencePiece, gemma_pytorch tokenizer.model, 256k), Mistral v0.1 (SentencePiece tokenizer.model.v1, 32k).
"""
import json, random, statistics as st, sys, unicodedata, collections
from pathlib import Path
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import sentencepiece as spm

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
random.seed(0)

LLAMA_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN_PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

llama = tiktoken.Encoding(name="llama3", pat_str=LLAMA_PAT, mergeable_ranks=load_tiktoken_bpe(str(ROOT / "tokenizers/llama3_tokenizer.model")), special_tokens={})
qwen = tiktoken.Encoding(name="qwen", pat_str=QWEN_PAT, mergeable_ranks=load_tiktoken_bpe(str(ROOT / "tokenizers/qwen.tiktoken")), special_tokens={})
gemma = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizers/gemma_tokenizer.model"))
mistral = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizers/mistral_tokenizer.model.v1"))

TOK = {
    "Gemma-2": lambda s: len(gemma.encode(s)),
    "Llama-3": lambda s: len(llama.encode_ordinary(s)),
    "Qwen2.5": lambda s: len(qwen.encode_ordinary(s)),
    "Mistral-v0.1": lambda s: len(mistral.encode(s)),
}
# Word-level counts: a word inside running text is preceded by a space. Mistral's SentencePiece model
# adds its own dummy prefix, so it gets the bare word.
WORD = {
    "Gemma-2": lambda w: len(gemma.encode(" " + w)),
    "Llama-3": lambda w: len(llama.encode_ordinary(" " + w)),
    "Qwen2.5": lambda w: len(qwen.encode_ordinary(" " + w)),
    "Mistral-v0.1": lambda w: len(mistral.encode(w)),
}

def is_deva(s):
    return any("ऀ" <= ch <= "ॿ" for ch in s)

def load_lines(p):
    return [unicodedata.normalize("NFC", l.strip()) for l in open(p, encoding="utf-8") if l.strip()]

cv = [l for l in load_lines(ROOT / "data/public/cv_ne_sentence_collector.txt") if is_deva(l)]
udhr = [l for l in load_lines(ROOT / "data/public/udhr2_nep.txt") if is_deva(l)]
seen, corpus = set(), []
for src, lines in (("cv", cv), ("udhr", udhr)):
    for l in lines:
        if l not in seen:
            seen.add(l); corpus.append((src, l))

CONS = set(chr(c) for c in range(0x0915, 0x093A)) | set(chr(c) for c in range(0x0958, 0x0960))
VIRAMA, NUKTA = "्", "़"
MATRAS = set(chr(c) for c in range(0x093E, 0x094D)) | {"ॢ", "ॣ"}
IND_VOWELS = set(chr(c) for c in range(0x0904, 0x0915))

def aksharas(word):
    n = 0
    chars = [c for c in word if c != NUKTA]
    for i, c in enumerate(chars):
        if c in IND_VOWELS:
            n += 1
        elif c in CONS:
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt != VIRAMA:
                n += 1
    return n

def ends_in_inherent_vowel(word):
    core = word.rstrip("ँंः")  # trailing chandrabindu/anusvara/visarga keep vowel? treat as no
    if core != word:
        return False
    chars = [c for c in word if c != NUKTA]
    return bool(chars) and chars[-1] in CONS

PUNCT_TAIL = "।॥,.?!;:\"'()[]{}-–—"

def split_punct(tok):
    i = len(tok)
    while i > 0 and tok[i - 1] in PUNCT_TAIL:
        i -= 1
    return tok[:i], tok[i:]

def romanize(line):
    r1_words, r2_words, r3_words, info = [], [], [], []
    for w in line.split():
        core, tail = split_punct(w)
        r1_core = transliterate(core, sanscript.DEVANAGARI, sanscript.ITRANS) if core else ""
        r1_tail = transliterate(tail, sanscript.DEVANAGARI, sanscript.ITRANS) if tail else ""
        r2_core = r1_core.lower()
        inherent = bool(core) and is_deva(core) and ends_in_inherent_vowel(core)
        poly = aksharas(core) >= 2 if core else False
        r3_core = r2_core[:-1] if (inherent and poly and r2_core.endswith("a")) else r2_core
        r1_words.append(r1_core + r1_tail); r2_words.append(r2_core + r1_tail.lower()); r3_words.append(r3_core + r1_tail.lower())
        info.append({"inherent": inherent, "poly": poly, "upper": any(ch.isupper() for ch in r1_core), "r1": r1_core, "r2": r2_core})
    return " ".join(r1_words), " ".join(r2_words), " ".join(r3_words), info

rows = []
for src, line in corpus:
    r1, r2, r3, info = romanize(line)
    rows.append({"src": src, "deva": line, "R1": r1, "R2": r2, "R3": r3, "info": info})

CONVS = ["R1", "R2", "R3"]

def stats_for(conv, name):
    f = TOK[name]; wf = WORD[name]
    ratios, toks, words, chars = [], 0, 0, 0
    word_counts = []
    for r in rows:
        text = r[conv]
        n_w = len(text.split())
        if n_w == 0:
            continue
        n_t = f(text)
        ratios.append(n_t / n_w); toks += n_t; words += n_w; chars += len(text)
        word_counts.extend(wf(w) for w in text.split())
    types = collections.Counter(w for r in rows for w in r[conv].split())
    type_fert = st.mean(wf(w) for w in types)
    single = sum(1 for c in word_counts if c == 1) / len(word_counts)
    three = sum(1 for c in word_counts if c >= 3) / len(word_counts)
    # bootstrap over sentences for the mean-of-ratios fertility
    boots = []
    n = len(ratios)
    for _ in range(1000):
        sample = [ratios[random.randrange(n)] for _ in range(n)]
        boots.append(st.mean(sample))
    boots.sort()
    return {"fertility": st.mean(ratios), "ci": [boots[24], boots[974]], "pooled": toks / words,
            "tokens_per_char": toks / chars, "single_token_share": single, "three_plus_share": three,
            "type_fertility": type_fert, "n_sentences": n, "n_words": words, "n_types": len(types), "ratios": ratios}

results = {conv: {name: stats_for(conv, name) for name in TOK} for conv in CONVS}

# Ordering robustness: share of paired bootstrap resamples (same sentences) preserving Gemma < Llama < Qwen < Mistral
order = ["Gemma-2", "Llama-3", "Qwen2.5", "Mistral-v0.1"]
order_share = {}
for conv in CONVS:
    R = {name: results[conv][name]["ratios"] for name in order}
    n = len(R[order[0]]); hold = 0; rng = random.Random(1)
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        means = [st.mean(R[name][i] for i in idx) for name in order]
        hold += all(means[i] < means[i + 1] for i in range(3))
    order_share[conv] = hold / 2000

# Lowercasing ambiguity and inherent-vowel statistics (over running words)
all_info = [i for r in rows for i in r["info"] if i["r1"]]
r1_types_by_r2 = collections.defaultdict(set)
for i in all_info:
    r1_types_by_r2[i["r2"]].add(i["r1"])
colliding_r2 = {k for k, v in r1_types_by_r2.items() if len(v) > 1}
lex = {
    "running_words": len(all_info),
    "share_words_with_uppercase_itrans": sum(i["upper"] for i in all_info) / len(all_info),
    "r1_types": len({i["r1"] for i in all_info}),
    "r2_types": len(r1_types_by_r2),
    "share_r2_types_merging_distinct_r1_types": len(colliding_r2) / len(r1_types_by_r2),
    "share_running_words_in_merged_types": sum(1 for i in all_info if i["r2"] in colliding_r2) / len(all_info),
    "share_words_ending_in_inherent_vowel": sum(i["inherent"] for i in all_info) / len(all_info),
    "share_words_changed_by_R3": sum(1 for i in all_info if i["inherent"] and i["poly"]) / len(all_info),
    "examples_merged": sorted([(k, sorted(v)) for k, v in r1_types_by_r2.items() if len(v) > 1], key=lambda x: -len(x[1]))[:12],
}

# Merged character groups under lowercasing (letters and signs), derived from the scheme
units = {}
for c in sorted(CONS):
    units[c + VIRAMA] = transliterate(c + VIRAMA, sanscript.DEVANAGARI, sanscript.ITRANS)
for v in sorted(IND_VOWELS):
    units[v] = transliterate(v, sanscript.DEVANAGARI, sanscript.ITRANS)
for s in ["ँ", "ं", "ः"]:
    units["क" + s] = transliterate("क" + s, sanscript.DEVANAGARI, sanscript.ITRANS)
groups = collections.defaultdict(set)
for deva, itr in units.items():
    groups[itr.lower()].add((deva, itr))
merged_groups = sorted([(k, sorted(v)) for k, v in groups.items() if len({x[1] for x in v}) > 1])

summary = {
    "corpus": {"sentences": len(rows), "cv_sentences": sum(r["src"] == "cv" for r in rows), "udhr_lines": sum(r["src"] == "udhr" for r in rows),
               "devanagari_words": sum(len(r["deva"].split()) for r in rows)},
    "results": {conv: {name: {k: v for k, v in d.items() if k != "ratios"} for name, d in results[conv].items()} for conv in CONVS},
    "ordering_share": order_share,
    "lowercasing": lex,
    "merged_unit_groups": merged_groups,
    "examples": [{k: rows[i][k] for k in ("deva", "R1", "R2", "R3")} for i in (0, 3, 8, 12)],
}
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
for conv in CONVS:
    print(f"\n== {conv} ==")
    for name in TOK:
        d = results[conv][name]
        print(f"{name:13s} fert {d['fertility']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}] pooled {d['pooled']:.3f} tok/char {d['tokens_per_char']:.4f} single {d['single_token_share']:.3f} 3+ {d['three_plus_share']:.3f} typeF {d['type_fertility']:.3f}")
print("\nordering share", order_share)
print(json.dumps({k: v for k, v in lex.items() if k != 'examples_merged'}, indent=1))
print("merged examples", lex["examples_merged"][:8])
print("merged unit groups:")
for k, v in merged_groups:
    print(" ", k, v)
print("corpus", summary["corpus"])
for e in summary["examples"]:
    print(e)
