"""Produce a factual manager brief and a Cyrillic PDF from one chat session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RED = colors.HexColor("#e30613")
INK = colors.HexColor("#202027")
MUTED = colors.HexColor("#6f6d76")


@dataclass(frozen=True)
class ManagerBrief:
    client_request: str
    details: str
    consultation: str
    questions: str
    sources: tuple[str, ...]

    def as_text(self) -> str:
        return (f"Запрос: {self.client_request}\n"
                f"Детали: {self.details}\n"
                f"Консультация: {self.consultation}\n"
                f"Что уточнить: {self.questions}")


class SummaryAgent:
    def summarize(self, messages: list[dict]) -> ManagerBrief:
        user = [row["content"].strip() for row in messages if row["role"] == "user"]
        assistant = [row for row in messages if row["role"] == "assistant"
                     and not row["content"].startswith("Заявка сформирована")]
        cited = []
        for row in assistant:
            for source in row.get("sources", []):
                title = source.get("source", "")
                if title and title not in cited:
                    cited.append(title)
        return ManagerBrief(
            client_request=(user[0] if user else "Не указан")[:300],
            details=(" / ".join(user[1:5]) if len(user) > 1 else "Дополнительные данные пока не сообщены")[:500],
            consultation=(" / ".join(row["content"] for row in assistant[-2:])
                          if assistant else "Консультация пока не состоялась")[:500],
            questions=("Уточнить параметры задачи и обсудить следующий шаг с клиентом"),
            sources=tuple(cited[:5]),
        )

    @staticmethod
    def _font() -> str:
        if "EnergytoolSans" in pdfmetrics.getRegisteredFontNames():
            return "EnergytoolSans"
        paths = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        for path in paths:
            if path.is_file():
                pdfmetrics.registerFont(TTFont("EnergytoolSans", str(path)))
                return "EnergytoolSans"
        raise RuntimeError("Для PDF нужен шрифт DejaVu Sans или Arial с кириллицей")

    def render_pdf(self, lead_id: int, full_name: str, phone: str, company: str,
                   created_at: str, brief: ManagerBrief) -> bytes:
        font = self._font()
        output = BytesIO()
        doc = SimpleDocTemplate(output, pagesize=(210 * mm, 297 * mm),
                                leftMargin=22 * mm, rightMargin=22 * mm,
                                topMargin=22 * mm, bottomMargin=20 * mm,
                                title=f"Заявка Energytool №{lead_id}")
        title = ParagraphStyle("title", fontName=font, fontSize=21, leading=27,
                               textColor=INK, spaceAfter=7 * mm)
        kicker = ParagraphStyle("kicker", fontName=font, fontSize=9, leading=13,
                                textColor=RED, spaceAfter=4 * mm)
        section = ParagraphStyle("section", fontName=font, fontSize=11, leading=16,
                                 textColor=RED, spaceBefore=5 * mm, spaceAfter=2 * mm)
        body = ParagraphStyle("body", fontName=font, fontSize=9.5, leading=15,
                              textColor=INK, alignment=TA_LEFT)
        muted = ParagraphStyle("muted", fontName=font, fontSize=8, leading=12,
                               textColor=MUTED)
        safe = lambda value: escape(str(value)).replace("\n", "<br/>")

        story = [Paragraph("МТС ENERGYTOOL  /  ЗАЯВКА МЕНЕДЖЕРУ", kicker),
                 Paragraph("Резюме консультации", title)]
        try:
            date = datetime.fromisoformat(created_at).strftime("%d.%m.%Y %H:%M")
        except ValueError:
            date = created_at
        story.append(Paragraph(f"Заявка №{lead_id} · {safe(date)}", muted))
        story.append(Spacer(1, 8 * mm))
        details = [
            [Paragraph("КЛИЕНТ", muted), Paragraph(safe(full_name), body)],
            [Paragraph("КОМПАНИЯ", muted), Paragraph(safe(company), body)],
            [Paragraph("ТЕЛЕФОН", muted), Paragraph(safe(phone), body)],
        ]
        table = Table(details, colWidths=[37 * mm, 124 * mm], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f7f9")),
            ("BOX", (0, 0), (-1, -1), .4, colors.HexColor("#e8e7eb")),
            ("LINEBELOW", (0, 0), (-1, 1), .4, colors.HexColor("#e8e7eb")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(table)
        sections = [
            ("Задача клиента", brief.client_request),
            ("Дополнительные сведения", brief.details),
            ("Что обсудили", brief.consultation),
            ("Что уточнить менеджеру", brief.questions),
        ]
        for heading, value in sections:
            story.append(KeepTogether([Paragraph(heading, section), Paragraph(safe(value), body)]))
        if brief.sources:
            story.append(Paragraph("Документы базы знаний", section))
            for source in brief.sources:
                story.append(Paragraph("• " + safe(source), muted))
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("Сформировано автоматически по сохранённому диалогу. Факты о клиенте требуют проверки менеджером.", muted))
        doc.build(story)
        return output.getvalue()
