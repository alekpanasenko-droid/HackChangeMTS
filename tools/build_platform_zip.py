"""Build the root-level ZIP for the hackathon's automatic evaluation stage."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "platform" / "solution.py"
REQUIREMENTS = ROOT / "platform" / "requirements.txt"
PROMPT = ROOT / "system_prompt.txt"
OUTPUT = ROOT / "output" / "energytool-platform-submission.zip"
EXPECTED = {"solution.py", "requirements.txt"}
ALLOWED = {"httpx", "numpy", "pydantic", "rank-bm25"}
MAX_ZIP_BYTES = 128 * 1024 * 1024


def build_solution() -> str:
    source = SOURCE.read_text(encoding="utf-8-sig")
    prompt = PROMPT.read_text(encoding="utf-8-sig").strip()
    if not prompt:
        raise ValueError("Пустой системный промпт")
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "SYSTEM"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("В platform/solution.py должен быть ровно один SYSTEM")
    target = assignments[0]
    format_rules = ast.literal_eval(target.value)
    combined = prompt + "\n\n[Формат ответа для автоматической проверки]\n" + format_rules
    lines = source.splitlines(keepends=True)
    lines[target.lineno - 1:target.end_lineno] = ["SYSTEM = " + repr(combined) + "\n"]
    result = "".join(lines)
    compile(result, "solution.py", "exec")
    return result


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
