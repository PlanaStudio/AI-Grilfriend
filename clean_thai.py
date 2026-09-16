"""Extract Thai-script text from the LINE chat export."""
from pathlib import Path
import re


SOURCE = Path("[LINE]All idiots Official🛡️.txt")
THAI_OUTPUT = Path("thai_only.txt")
ENGLISH_OUTPUT = Path("english_only.txt")


def clean_thai_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        thai_parts = re.findall(r"[\u0e00-\u0e7f]+(?:[\s\u0e00-\u0e7f]+[\u0e00-\u0e7f]+)*", line)
        cleaned = " ".join(part.strip() for part in thai_parts).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines) + ("\n" if lines else "")


def clean_english_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        english_parts = re.findall(r"[A-Za-z]+(?:[\s'-]+[A-Za-z]+)*", line)
        cleaned = " ".join(part.strip() for part in english_parts).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    THAI_OUTPUT.write_text(clean_thai_text(text), encoding="utf-8")
    ENGLISH_OUTPUT.write_text(clean_english_text(text), encoding="utf-8")
    print(f"Wrote {THAI_OUTPUT} and {ENGLISH_OUTPUT} from {SOURCE}")


if __name__ == "__main__":
    main()