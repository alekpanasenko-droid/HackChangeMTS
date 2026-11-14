import tempfile
import unittest
from pathlib import Path

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

    def tearDown(self):
        service.orchestrator.answer_model = self.original_answer
        service.DB = self.original_db
        service.LEAD_PDF_DIR = self.original_pdf_dir
        self.temp.cleanup()

    def test_chat_to_pdf_and_new_session(self):
        session = service.create_session()["id"]
        answer = service.chat(service.ChatInput(
            session_id=session, message="Как найти потери электроэнергии на предприятии?"
        ))
        self.assertEqual(answer["trace"]["decision"], "answered")
        self.assertTrue(answer["sources"])

        lead = service.create_lead(service.LeadInput(
            session_id=session, full_name="Иванов Иван Иванович",
            phone="+7 900 000-00-00", company="ООО Тест",
        ))
        pdf = service.download_lead_pdf(lead["id"])
        self.assertTrue(Path(pdf.path).read_bytes().startswith(b"%PDF-"))
        saved = service.get_session(session)
        self.assertEqual(saved["lead"]["id"], lead["id"])
        self.assertEqual(saved["messages"][1]["trace"]["decision"], "answered")

        fresh = service.create_session()["id"]
        self.assertEqual(service.get_session(fresh)["messages"], [])


if __name__ == "__main__":
    unittest.main()
