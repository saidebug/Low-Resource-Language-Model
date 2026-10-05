"""Step 4: use the model. Runs locally, on a server, or local-first with server fallback.

Examples
  python explain.py --language hindi --domain medical --text "इस औषधि को ..."
  python explain.py --language tamil --domain govt --file notice.txt --mode auto
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path

from checks import check_output
from config import DOMAINS, LANGUAGES, OUT_DIR, build_prompt

DEFAULT_MODEL = OUT_DIR / "model-Q4_K_M.gguf"
_local_llm = None  # loaded once, reused


def run_local(prompt: str, model_path) -> str:
    global _local_llm
    if _local_llm is None:
        from llama_cpp import Llama
        _local_llm = Llama(model_path=str(model_path), n_ctx=2048,
                           n_threads=os.cpu_count() or 4, verbose=False)
    out = _local_llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, temperature=0.2, repeat_penalty=1.1)
    return out["choices"][0]["message"]["content"].strip()


def server_up(url: str) -> bool:
    try:
        urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2)
        return True
    except Exception:
        return False


def run_server(prompt: str, url: str) -> str:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 512, "temperature": 0.2}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def explain(source, language="hindi", domain="govt", mode="auto",
            model_path=DEFAULT_MODEL, server_url="http://localhost:8080") -> dict:
    prompt = build_prompt(language, domain, source)

    backends = []
    if mode in ("local", "auto") and Path(model_path).exists():
        backends.append("local")
    if mode == "server" or (mode == "auto" and (not backends or server_up(server_url))):
        backends.append("server")
    if not backends:
        raise SystemExit(f"No model found at {model_path}. Run the export step first, or use --mode server.")

    result = None
    for backend in backends:
        try:
            answer = run_local(prompt, model_path) if backend == "local" else run_server(prompt, server_url)
        except Exception as e:  # e.g. server unreachable
            result = result or {"answer": "", "backend": backend, "ok": False, "problems": [f"{backend} failed: {e}"]}
            continue
        check = check_output(source, answer, language)
        result = {"answer": answer, "backend": backend, **check}
        if check["ok"]:
            break  # good answer; no need to escalate to the next backend

    if not result["ok"]:
        result["note"] = "Automatic checks failed. Please verify against the original text."
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=LANGUAGES, default="hindi")
    ap.add_argument("--domain", choices=DOMAINS, default="govt")
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--mode", choices=["local", "server", "auto"], default="auto")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--server-url", default="http://localhost:8080")
    args = ap.parse_args()

    source = args.text or (Path(args.file).read_text(encoding="utf-8") if args.file else None)
    if not source:
        raise SystemExit("Give --text or --file")

    res = explain(source.strip(), args.language, args.domain, args.mode, args.model, args.server_url)
    print(res["answer"])
    print(f"\n[backend: {res['backend']} | checks passed: {res['ok']}]")
    for p in res["problems"]:
        print("  warning:", p)
    if res.get("note"):
        print(" ", res["note"])


if __name__ == "__main__":
    main()
