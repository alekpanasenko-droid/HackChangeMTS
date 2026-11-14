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
    def test_general_product_question_uses_product_map(self):
        class ProductBackend(Backend):
            def retrieve(self, question, top_k):
                if question == "Карта продуктов МТС EnergyTool":
                    return [{"id": "S2", "source": "00-reference/00-product-map.md",
                             "heading": "Карта продуктов МТС EnergyTool",
                             "text": "Задачи клиента и подходящие направления", "score": 1.0}]
                return [{"id": "S1", "source": "10-energy/10-market.md",
                         "heading": "Рынок электроэнергии",
                         "text": "За что вы платите", "score": 1.0}]

        result = SearchAgent(ProductBackend([])).find("Что вы предлагаете?")
        self.assertTrue(result.sufficient)
        self.assertEqual(result.sources[0]["source"], "00-reference/00-product-map.md")

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
