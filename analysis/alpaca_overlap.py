"""How closely the English (type A) test instructions match the Stanford Alpaca instructions.

Type A instructions were machine-translated back into English from the Nepali source dataset.
For each one, this script records whether it is identical to an Alpaca instruction and its
similarity to the nearest Alpaca instruction (rapidfuzz fuzz.ratio, a normalized Indel edit
similarity on a 0-100 scale; no case or whitespace normalization).

Alpaca data: tatsu-lab/stanford_alpaca @ 761dc5bfbdeeffa89b8bff5d038781a4055f796a, alpaca_data.json
(52,002 examples), e.g.
  curl -fsSLO https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json

Usage: python analysis/alpaca_overlap.py <rebuilt test.jsonl> <alpaca_data.json> <out.json>
"""
import json, statistics, sys
from rapidfuzz import fuzz, process

test_path, alpaca_path, out_path = sys.argv[1:4]
rows = [json.loads(line) for line in open(test_path, encoding="utf-8") if line.strip()]
type_a = [r["instruction"] for r in rows if r["type"] == "A"]
alpaca = [d["instruction"] for d in json.load(open(alpaca_path, encoding="utf-8"))]
alpaca_set = set(alpaca)

best = [process.extractOne(s, alpaca, scorer=fuzz.ratio)[1] for s in type_a]
out = {
    "type_a_instructions": len(type_a),
    "alpaca_instructions": len(alpaca),
    "share_identical": sum(s in alpaca_set for s in type_a) / len(type_a),
    "share_similarity_ge_80": sum(b >= 80 for b in best) / len(type_a),
    "share_similarity_ge_60": sum(b >= 60 for b in best) / len(type_a),
    "median_similarity": statistics.median(best),
    "min_similarity": min(best),
}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out, indent=1))
