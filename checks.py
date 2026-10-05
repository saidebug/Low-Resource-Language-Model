"""Cheap automatic safety checks on a model answer.

These do not prove an answer is correct. They catch the most dangerous
failures for notices and medical text: changed or invented numbers,
answers in the wrong script, and empty or runaway output.
"""
import re
import unicodedata

from config import LANGUAGES


def _ascii_digits(text: str) -> str:
    """Convert digits from any script (e.g. Tamil ௩௦, Devanagari ३०) to 0-9."""
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


def extract_numbers(text: str) -> list[str]:
    t = _ascii_digits(text)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 1,000 -> 1000
    return re.findall(r"\d+(?:\.\d+)?", t)


def script_ratio(text: str, language: str) -> float:
    """Share of letters (and combining vowel signs) in the target script."""
    lo, hi = LANGUAGES[language]["script"]
    chars = [c for c in text if unicodedata.category(c)[0] in ("L", "M")]
    if not chars:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in chars) / len(chars)


def check_output(source: str, output: str, language: str, min_script: float = 0.7) -> dict:
    out = output.strip()
    if not out:
        return {"ok": False, "problems": ["empty answer"]}

    problems = []
    src_nums, out_nums = set(extract_numbers(source)), set(extract_numbers(out))

    missing = sorted(src_nums - out_nums)
    if missing:
        problems.append("numbers missing from answer: " + ", ".join(missing))
    invented = sorted(out_nums - src_nums)
    if invented:
        problems.append("answer has numbers not in the original: " + ", ".join(invented))

    if script_ratio(out, language) < min_script:
        problems.append("answer is not mostly in the target script")

    ratio = len(out) / max(len(source), 1)
    if not 0.3 <= ratio <= 3.0:
        problems.append(f"answer length looks wrong ({ratio:.1f}x the original)")

    return {"ok": not problems, "problems": problems}
