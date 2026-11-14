"""Find evidence in the supplied RAG and assess whether it fits the current question."""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD = re.compile(r"[a-zа-яё0-9]+", re.I)
STOP = {
    "какие", "какой", "какая", "можно", "нужно", "чтобы", "есть", "этот",
    "этого", "меня", "пожалуйста", "скажите", "расскажите", "просто",
    "вопрос", "помогите", "energytool", "мтс", "электроэнергии", "энергии",
}


def terms(text: str) -> set[str]:
    return {word for word in WORD.findall(text.lower()) if len(word) >= 3 and word not in STOP}


@dataclass(frozen=True)
class SearchResult:
    sources: list[dict]
    confidence: float
    sufficient: bool
    reason: str


class SearchAgent:
    def __init__(self, backend):
        self.backend = backend

    def find(self, question: str) -> SearchResult:
        candidates = self.backend.retrieve(question, top_k=6)
        if not candidates:
            return SearchResult([], 0.0, False, "В базе не найдено подходящих фрагментов")

        query_terms = terms(question[-2500:])
        matched = []
        for source in candidates:
            source_terms = terms(source.get("heading", "") + " " + source["text"])
            overlap = len(query_terms & source_terms)
            semantic = self.backend.mode.startswith("FAISS") and source.get("score", 0) >= 0.42
            if overlap or semantic:
                matched.append((overlap, source))

        if not matched:
            return SearchResult([], 0.0, False, "Совпадения по теме запроса слишком слабые")

        best_overlap = max(count for count, _ in matched)
        confidence = min(1.0, best_overlap / max(1, min(len(query_terms), 3)))
        if self.backend.mode.startswith("FAISS"):
            confidence = max(confidence, min(1.0, max(s["score"] for _, s in matched)))
        sources = [source for _, source in matched[:6]]
        return SearchResult(sources, round(confidence, 2), confidence >= 0.3,
                            "Найдены фрагменты по теме запроса" if confidence >= 0.3
                            else "Найденные фрагменты требуют уточнения задачи")
