import unittest

from agents.orchestrator import OrchestratorAgent
from agents.search_agent import SearchAgent
from agents.summary_agent import SummaryAgent


class Backend:
    mode = "BM25"

    def __init__(self, sources):
        self.sources = sources

    def retrieve(self, question, top_k):
        return self.sources[:top_k]


class AgentTests(unittest.TestCase):
    def test_orchestrator_clarifies_without_evidence(self):
        called = []
        agent = OrchestratorAgent(SearchAgent(Backend([])), lambda *_: called.append(True))
        result = agent.respond([{"role": "user", "content": "Как снизить потери?"}])
        self.assertEqual(result.decision, "clarify")
        self.assertEqual(result.sources, [])
        self.assertFalse(called)

    def test_orchestrator_answers_with_relevant_evidence(self):
        source = {"id": "S1", "source": "losses.md", "heading": "Потери",
                  "text": "Потери электроэнергии можно исследовать по данным учёта.", "score": 2.0}
        agent = OrchestratorAgent(SearchAgent(Backend([source])),
                                  lambda history, sources: ("Нужны данные учёта.", sources))
        result = agent.respond([{"role": "user", "content": "Как найти потери электроэнергии?"}])
        self.assertEqual(result.decision, "answered")
        self.assertEqual(result.sources[0]["id"], "S1")
        self.assertGreater(result.confidence, 0)

    def test_summary_uses_only_dialogue_facts(self):
        messages = [
            {"role": "user", "content": "Нужно найти потери на складе"},
            {"role": "assistant", "content": "Для анализа нужны данные учёта.",
             "sources": [{"source": "losses.md"}]},
            {"role": "user", "content": "Счётчики уже установлены"},
        ]
        brief = SummaryAgent().summarize(messages)
        self.assertIn("складе", brief.client_request)
        self.assertIn("Счётчики", brief.details)
        self.assertEqual(brief.sources, ("losses.md",))


if __name__ == "__main__":
    unittest.main()
