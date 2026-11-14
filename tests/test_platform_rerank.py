"""Offline checks against the API contract supplied by the organizers."""
import importlib.util
import os
from pathlib import Path
import sys
from time import monotonic
import unittest
from unittest.mock import patch

import httpx

spec = importlib.util.spec_from_file_location(
    "platform_solution", Path(__file__).resolve().parents[1] / "platform" / "solution.py")
solution = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = solution
spec.loader.exec_module(solution)


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "LLM_BASE_URL": "https://proxy.example/v1",
            "LLM_API_KEY": "test-generation",
            "EMBEDDER_API_KEY": "test-embedding",
            "RERANKER_API_KEY": "test-reranking",
            "EMB_MODEL": "platform/embedding-alias",
            "RERANKER_MODEL": "platform/reranker-alias",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def proxy(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return solution.Proxy(client, monotonic() + 100)

    def test_service_credentials_and_model_identifiers(self):
        import json
        seen = []
        def handler(request):
            seen.append((request.url.path, request.headers["Authorization"], json.loads(request.content)))
            return httpx.Response(200, json={})
        proxy = self.proxy(handler)
        proxy.post("embeddings", {"model": proxy.embed_model}, timeout=1)
        proxy.post("rerank", {"model": proxy.rerank_model}, timeout=1)
        proxy.post("chat/completions", {}, timeout=1)
        self.assertEqual([r[1] for r in seen], ["Bearer test-embedding", "Bearer test-reranking",
                                               "Bearer test-generation"])
        self.assertEqual(seen[0][2]["model"], "platform/embedding-alias")
        self.assertEqual(seen[1][2]["model"], "platform/reranker-alias")

    def test_shared_key_fallback(self):
        os.environ.pop("RERANKER_API_KEY")
        def handler(request):
            self.assertEqual(request.headers["Authorization"], "Bearer test-generation")
            return httpx.Response(200, json={"results": [{"index": 0}]})
        passages = [solution.Passage("a.md", "Title", "Text")]
        self.assertEqual(self.proxy(handler).rerank("question", passages, 6), passages)

    def test_reranker_order_and_supported_response_shapes(self):
        import json
        passages = [solution.Passage(str(i), "Heading", "Text") for i in range(5)]
        for key in ("results", "data"):
            def handler(request):
                body = json.loads(request.content)
                self.assertEqual(request.url.path, "/v1/rerank")
                self.assertEqual(body["top_n"], 2)
                self.assertEqual(len(body["documents"]), 5)
                return httpx.Response(200, json={key: [{"index": 4}, {"index": 1}]})
            self.assertEqual(self.proxy(handler).rerank("question", passages, 2),
                             [passages[4], passages[1]])

    def test_reject_invalid_indices(self):
        passages = [solution.Passage("a.md", "", "Text") for _ in range(3)]
        for indices in ([0, 0], [0, 99], [True, 1], [0], ["0", 1]):
            proxy = self.proxy(lambda _: httpx.Response(200, json={
                "results": [{"index": i} for i in indices]}))
            with self.assertRaises(ValueError):
                proxy.rerank("question", passages, 2)

    def test_missing_endpoint_is_not_retried_for_each_question(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(404)
        proxy = self.proxy(handler)
        passages = [solution.Passage("a.md", "", "Text")]
        with self.assertRaises(httpx.HTTPStatusError):
            proxy.rerank("question", passages, 1)
        with self.assertRaises(TimeoutError):
            proxy.rerank("question", passages, 1)
        self.assertEqual(len(calls), 1)

    def test_expanded_pool_is_reduced_and_failure_keeps_lexical_result(self):
        class SearchProxy:
            fail = False
            def embeddings(self, texts):
                raise RuntimeError("offline")
            def rerank(self, query, passages, top_n):
                if self.fail:
                    raise RuntimeError("unavailable")
                self.pool = passages
                return passages[-top_n:]
        passages = [solution.Passage(str(i), "", f"alpha item{i}") for i in range(40)]
        proxy = SearchProxy()
        retriever = solution.Retriever(passages, proxy)
        selected = retriever.search("alpha")
        self.assertEqual(len(proxy.pool), 30)
        self.assertEqual(len(selected), 6)
        self.assertEqual(selected, proxy.pool[-6:])
        proxy.fail = True
        self.assertEqual(retriever.search("alpha"), passages[:6])


if __name__ == "__main__":
    unittest.main()
