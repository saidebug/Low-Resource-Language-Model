"""Step 5: run the model on held-out examples and write a report for human review."""
import argparse
import json

from config import DATA_DIR, OUT_DIR
from explain import DEFAULT_MODEL, explain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="local", choices=["local", "server", "auto"])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()

    with open(DATA_DIR / "val_raw.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit("val_raw.jsonl is empty. Add more data and re-run prepare_data.py.")

    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
    except ImportError:
        chrf = None

    report, passed, scores = [], 0, []
    for r in rows:
        res = explain(r["source"], r["language"], r["domain"], mode=args.mode, model_path=args.model)
        passed += res["ok"]
        if chrf:
            scores.append(chrf.sentence_score(res["answer"], [r["simple"]]).score)
        report.append({**r, "model_answer": res["answer"], "backend": res["backend"],
                       "checks_ok": res["ok"], "problems": res["problems"]})

    out = OUT_DIR / "eval_report.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for item in report:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"examples: {len(rows)} | passed automatic checks: {passed}/{len(rows)}")
    if scores:
        print(f"mean chrF vs reference: {sum(scores) / len(scores):.1f}")
    print(f"Report written to {out}. A native speaker must still review it.")


if __name__ == "__main__":
    main()
