"""Shared settings.

The prompt template lives here so that training, evaluation and inference
all build *exactly* the same prompt. If they drift apart, quality drops.
"""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

BASE_MODEL = "google/gemma-3-1b-it"

# Unicode ranges used to check an answer is really in the target script.
LANGUAGES = {
    "hindi":    {"name": "Hindi",    "script": (0x0900, 0x097F)},  # Devanagari
    "tamil":    {"name": "Tamil",    "script": (0x0B80, 0x0BFF)},
    "assamese": {"name": "Assamese", "script": (0x0980, 0x09FF)},  # Bengali-Assamese block
}

DOMAINS = {
    "govt":    "government notice",
    "medical": "medical instruction",
    "legal":   "legal notice",
}


def build_prompt(language: str, domain: str, source: str) -> str:
    lang = LANGUAGES[language]["name"]
    dom = DOMAINS[domain]
    return (
        f"Rewrite the following {dom} in simple {lang} that a person with little "
        f"schooling can understand. Use short sentences and everyday words. "
        f"Keep every number, date, amount and name exactly as in the original. "
        f"Do not add any information that is not in the text.\n\n"
        f"Text:\n{source}"
    )
