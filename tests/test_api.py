import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app as service


class ApiFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_db = service.DB
        self.original_pdf_dir = service.LEAD_PDF_DIR
        self.original_answer = service.orchestrator.answer_model
        root = Path(self.temp.name)
        service.DB = root / "test.sqlite3"
        service.LEAD_PDF_DIR = root / "leads"
        service.init_db()
        service.orchestrator.answer_model = lambda history, sources: ("Для анализа нужны данные учёта.", sources[:1])
        self.client = TestClient(service.app)

    def tearDown(self):
        service.orchestrator.answer_model = self.original_answer
        service.DB = self.original_db
        service.LEAD_PDF_DIR = self.original_pdf_dir
        self.client.close()
        self.temp.cleanup()

    def test_chat_to_pdf_and_new_session(self):
        session = self.client.post("/api/sessions").json()["id"]
        response = self.client.post("/api/chat", json={
            "session_id": session, "message": "Как найти потери электроэнергии на предприятии?"
        })
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()
        self.assertEqual(answer["trace"]["decision"], "answered")
        self.assertTrue(answer["sources"])

        lead = self.client.post("/api/leads", json={
            "session_id": session, "full_name": "Иванов Иван Иванович",
            "phone": "+7 900 000-00-00", "company": "ООО Тест",
        })
        self.assertEqual(lead.status_code, 200, lead.text)
        pdf = self.client.get(lead.json()["pdf_url"])
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.content.startswith(b"%PDF-"))
        saved = self.client.get(f"/api/sessions/{session}").json()
        self.assertEqual(saved["lead"]["id"], lead.json()["id"])
        self.assertEqual(saved["messages"][1]["trace"]["decision"], "answered")

        fresh = self.client.post("/api/sessions").json()["id"]
        self.assertEqual(self.client.get(f"/api/sessions/{fresh}").json()["messages"], [])


if __name__ == "__main__":
    unittest.main()
