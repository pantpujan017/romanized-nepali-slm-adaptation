"""Add the reference responses to the released per-item outputs.

The per-item files in results/per_item/ hold each run's generations, reference NLL and token and
byte counts, but not the prompts or references, because the source dataset states no licence
(data/README.md). After rebuilding the test split from the source dataset, this script adds the
references back so that analysis/run_level_analysis.py can recompute chrF++, BLEU and ROUGE.

Checks before writing anything:
  * the rebuilt test file has the SHA-256 of the file used in the paper;
  * every reference has the word and UTF-8 byte counts (and, if the file has a "type" field, the
    format) recorded for that item in results/hf_tokenizer_audit/fertility_per_output.csv.
Item ids are the 0-based line numbers of the test file, unless its rows carry an integer "id".

Usage:
  python scripts/restore_references.py <rebuilt test.jsonl> results/per_item <out_dir>
  python analysis/run_level_analysis.py <out_dir> results/run_level_results.json
"""
import csv, gzip, hashlib, json, sys
from pathlib import Path

TEST_SHA256 = "d656353e9ac233d73a81e9b0cc997c3ee8b4a66dcc96169a48150b7275254694"
REPO = Path(__file__).resolve().parent.parent
test_path, per_item, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
if digest != TEST_SHA256:
    sys.exit(f"{test_path} has SHA-256 {digest}, not the test split used in the paper ({TEST_SHA256}).")

rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
refs, types = {}, {}
for i, r in enumerate(rows):
    key = r["id"] if isinstance(r.get("id"), int) else i
    refs[key], types[key] = r["output"], r.get("type")
assert sorted(refs) == list(range(1000)), "expected 1,000 test items with ids 0..999"

audit = {int(r["id"]): r for r in csv.DictReader(open(REPO / "results/hf_tokenizer_audit/fertility_per_output.csv", encoding="utf-8"))}
for i, ref in refs.items():
    a = audit[i]
    ok = len(ref.encode("utf-8")) == int(a["bytes"]) and len(ref.split()) == int(a["words"])
    if types[i] is not None:
        ok = ok and types[i] == a["type"]
    if not ok:
        sys.exit(f"item {i}: the reference does not match the recorded word/byte counts; check the id order.")

out_dir.mkdir(parents=True, exist_ok=True)
for f in sorted(per_item.glob("*.jsonl.gz")):
    run = f.name[: -len(".jsonl.gz")]
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh]
    with open(out_dir / f"{run}.jsonl", "w", encoding="utf-8") as fh:
        for it in items:
            it["reference"] = refs[it["id"]]
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {out_dir / (run + '.jsonl')}")
