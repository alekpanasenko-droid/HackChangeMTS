"""Build a portable FAISS index from a directory of Markdown files."""

import argparse
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import faiss

from rag_common import (
    DIMENSION,
    INDEX_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    atomic_output,
    embed,
    load_embedder,
)

HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
CLOSING_HASHES = re.compile(r"[ \t]+#+$")
FENCE = re.compile(r"^ {0,3}(\x60{3,}|~{3,})(.*)$")
BOUNDARIES = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"(?<=[.!?…])\s+"),
    re.compile(r"\n"),
    re.compile(r"\s+"),
)


def sections(text: str):
    """Yield heading ancestry, first body line, and original Markdown."""
    headings: list[tuple[int, str]] = []
    body: list[str] = []
    first_line = 1
    fence = ""
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        raw = line.rstrip("\r\n")
        marker = FENCE.match(raw)
        if fence:
            if marker:
                closing, tail = marker.groups()
                if closing[0] == fence[0] and len(closing) >= len(fence) and not tail.strip():
                    fence = ""
            body.append(line)
            continue
        if marker:
            fence = marker[1]
            body.append(line)
            continue

        heading = HEADING.match(raw)
        if heading:
            if body:
                yield " > ".join(title for _, title in headings), first_line, "".join(body)
                body.clear()
            level = len(heading[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            title = CLOSING_HASHES.sub("", heading[2]).strip()
            headings.append((level, title))
            first_line = number + 1
        else:
            body.append(line)
    if body:
        yield " > ".join(title for _, title in headings), first_line, "".join(body)


def chunk_ranges(text: str, size: int, overlap: int):
    """Prefer paragraphs and sentences, with a hard size limit and overlap."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            lower = start + max(size * 2 // 3, overlap + 1)
            for boundary in BOUNDARIES:
                matches = list(boundary.finditer(text, lower, end))
                if matches:
                    end = matches[-1].end()
                    break
        yield start, end
        if end == len(text):
            break
        start = end - overlap


def load_chunks(docs: Path, size: int, overlap: int) -> list[dict]:
    if not docs.is_dir():
        raise FileNotFoundError(f"Каталог документов не найден: {docs}")
    chunks = []
    for path in sorted(docs.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        source = path.relative_to(docs).as_posix()
        for heading, first_line, body in sections(path.read_text(encoding="utf-8-sig")):
            for start, end in chunk_ranges(body, size, overlap):
                span = body[start:end]
                passage = span.strip()
                if not passage:
                    continue
                text_start = start + len(span) - len(span.lstrip())
                text_end = start + len(span.rstrip())
                chunks.append(
                    {
                        "source": source,
                        "heading": heading,
                        "line_start": first_line + body.count("\n", 0, text_start),
                        "line_end": first_line + body.count("\n", 0, text_end),
                        "text": passage,
                    }
                )
    if not chunks:
        raise ValueError(f"В {docs} нет непустых фрагментов Markdown")
    return chunks


def write_index(output: Path, chunks: list[dict], index: faiss.IndexFlatIP) -> None:
    metadata = {
        "version": INDEX_VERSION,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "chunks": chunks,
    }
    with atomic_output(output, "wb") as handle, ZipFile(handle, "w", ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))
        archive.writestr("index.faiss", faiss.serialize_index(index).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("knowledge-base"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/index.zip"))
    parser.add_argument("--chunk-size", type=int, default=1400, help="Символов на фрагмент")
    parser.add_argument("--overlap", type=int, default=180, help="Перекрытие в символах")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 2000 or not 0 <= args.overlap < args.chunk_size:
        parser.error("Нужно 0 <= overlap < chunk-size <= 2000")
    if args.batch_size < 1:
        parser.error("batch-size должен быть положительным")

    chunks = load_chunks(args.docs, args.chunk_size, args.overlap)
    print(f"Подготовлено фрагментов: {len(chunks)}", flush=True)
    model = load_embedder(args.device)
    index = faiss.IndexFlatIP(DIMENSION)
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        texts = [f"{chunk['heading']}\n\n{chunk['text']}".strip() for chunk in batch]
        index.add(embed(model, texts, args.batch_size))
    write_index(args.output, chunks, index)
    print(f"Индекс сохранён: {args.output}", flush=True)


if __name__ == "__main__":
    main()
