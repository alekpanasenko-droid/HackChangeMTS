"""Coordinate retrieval, evidence gate and consultant response."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .search_agent import SearchAgent


@dataclass(frozen=True)
class Consultation:
    answer: str
    sources: list[dict]
    decision: str
    confidence: float
    reason: str
    retrieval_ms: int
    generation_ms: int


class OrchestratorAgent:
    def __init__(self, search_agent: SearchAgent, answer_model):
        self.search_agent = search_agent
        self.answer_model = answer_model

    def respond(self, history: list[dict]) -> Consultation:
        user_turns = [row["content"] for row in history if row["role"] == "user"]
        query = " ".join(user_turns[-3:])[-4500:]
        started = perf_counter()
        evidence = self.search_agent.find(query)
        retrieval_ms = round((perf_counter() - started) * 1000)
        if not evidence.sufficient:
            return Consultation(
                "Чтобы ответить по базе знаний, уточните, какую именно задачу с учётом или расходами "
                "электроэнергии вы хотите решить?",
                [], "clarify", evidence.confidence, evidence.reason, retrieval_ms, 0,
            )

        started = perf_counter()
        answer, used = self.answer_model(history, evidence.sources)
        generation_ms = round((perf_counter() - started) * 1000)
        decision = "answered" if used else "insufficient"
        return Consultation(answer, used, decision, evidence.confidence, evidence.reason,
                            retrieval_ms, generation_ms)
