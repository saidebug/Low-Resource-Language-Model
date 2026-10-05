"""Step 1: clean raw (source, simple) pairs and turn them into training files.

Input  : data/seed_examples.jsonl   (fields: language, domain, source, simple)
Output : data/train.jsonl, data/val.jsonl   (chat format used by train.py)
         data/val_raw.jsonl                 (raw fields, used by evaluate.py)
"""
import argparse
import json
import random
import re
import unicodedata

from checks import check_output
from config import DATA_DIR, DOMAINS, LANGUAGES, build_prompt


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)  # one canonical form for Indic scripts
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_chat(r):
    return {
        "prompt": [{"role": "user", "content": build_prompt(r["language"], r["domain"], r["source"])}],
        "completion": [{"role": "assistant", "content": r["simple"]}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_DIR / "seed_examples.jsonl"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, seen = [], set()
    with open(args.input, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("language") not in LANGUAGES or r.get("domain") not in DOMAINS:
                print(f"line {n}: dropped (unknown language/domain)")
                continue
            r["source"], r["simple"] = clean(r["source"]), clean(r["simple"])
            if not r["source"] or not r["simple"] or r["source"] in seen:
                print(f"line {n}: dropped (empty or duplicate)")
                continue
            # Reuse the same safety checks to reject bad training pairs.
            res = check_output(r["source"], r["simple"], r["language"])
            if not res["ok"]:
                print(f"line {n}: dropped ({'; '.join(res['problems'])})")
                continue
            seen.add(r["source"])
            rows.append(r)

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, round(len(rows) * args.val_frac)) if len(rows) >= 2 else 0
    val, train = rows[:n_val], rows[n_val:]

    write_jsonl(DATA_DIR / "train.jsonl", [to_chat(r) for r in train])
    write_jsonl(DATA_DIR / "val.jsonl", [to_chat(r) for r in val])
    write_jsonl(DATA_DIR / "val_raw.jsonl", val)
    print(f"kept {len(rows)} pairs -> {len(train)} train, {len(val)} validation")
    if len(rows) < 500:
        print("NOTE: this is a smoke-test size. Real quality needs roughly 1,000-10,000 reviewed pairs.")


if __name__ == "__main__":
    main()
