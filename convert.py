import csv, json
n = 0
with open("pairs.csv", encoding="utf-8-sig") as f, \
     open("data/pairs.jsonl", "w", encoding="utf-8") as out:
    for r in csv.DictReader(f):
        if r.get("status", "approved").strip().lower() != "approved":
            continue
        row = {k: r[k].strip() for k in ("language", "domain", "source", "simple")}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
print(n, "approved pairs written")