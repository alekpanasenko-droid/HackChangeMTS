"""Build the root-level ZIP for the hackathon's automatic evaluation stage."""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "platform" / "solution.py"
REQUIREMENTS = ROOT / "platform" / "requirements.txt"
OUTPUT = ROOT / "output" / "energytool-platform-submission.zip"
EXPECTED = {"solution.py", "requirements.txt"}
ALLOWED = {"httpx", "numpy", "pydantic", "rank-bm25"}
MAX_ZIP_BYTES = 128 * 1024 * 1024


def build_solution() -> str:
    source = SOURCE.read_text(encoding="utf-8-sig")
    # Конкурсный промпт хранится непосредственно в solution.py.
    # Исходный диалоговый промпт продолжает использоваться веб-приложением.
    compile(source, "solution.py", "exec")
    return source


def validate_requirements(text: str) -> None:
    dependencies = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9-]+)==([0-9][A-Za-z0-9.]*)", line)
        if not match:
            raise ValueError(f"Зависимость должна иметь закреплённую версию: {line}")
        dependencies.append(match.group(1).lower())
    if set(dependencies) != ALLOWED or len(dependencies) != len(ALLOWED):
        raise ValueError("Список зависимостей отличается от разрешённого набора для сабмита")


def main() -> None:
    solution = build_solution()
    requirements = REQUIREMENTS.read_text(encoding="utf-8-sig")
    validate_requirements(requirements)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("solution.py", solution)
        archive.writestr("requirements.txt", requirements)
    with ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if set(names) != EXPECTED or len(names) != len(EXPECTED):
            raise ValueError("ZIP содержит неверный набор файлов")
        if archive.testzip() is not None:
            raise ValueError("ZIP повреждён")
        if not archive.read("solution.py").startswith(b'"""'):
            raise ValueError("Некорректный solution.py")
    if OUTPUT.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("ZIP превышает лимит 128 МБ")
    print(f"Готово: {OUTPUT} ({OUTPUT.stat().st_size} байт)")


if __name__ == "__main__":
    main()
