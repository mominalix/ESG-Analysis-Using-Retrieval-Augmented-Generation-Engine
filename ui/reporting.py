"""Report export helpers for the Streamlit application."""

from __future__ import annotations

import io
import re
from html import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def markdown_to_pdf(content: str, title: str) -> bytes:
    """Render a conservative subset of Markdown to a downloadable PDF."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 16)]

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        if line.startswith("### "):
            style, text = styles["Heading3"], line[4:]
        elif line.startswith("## "):
            style, text = styles["Heading2"], line[3:]
        elif line.startswith("# "):
            style, text = styles["Heading1"], line[2:]
        elif re.match(r"^[-*]\s+", line):
            style, text = styles["BodyText"], f"• {line[2:]}"
        else:
            style, text = styles["BodyText"], line
        story.append(Paragraph(escape(text), style))

    document.build(story)
    return buffer.getvalue()
