from __future__ import annotations
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_pdf_report(review: Optional[Dict[str, Any]], testgen: Optional[Dict[str, Any]]) -> Optional[bytes]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Paragraph,
            Spacer,
            SimpleDocTemplate,
            Table,
            TableStyle,
            Preformatted,
        )
    except Exception:
        return None

    font_name = _register_cjk_font(pdfmetrics, TTFont)
    base_font = font_name or "Helvetica"

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = base_font
    styles["Title"].fontName = base_font
    styles["Heading1"].fontName = base_font
    styles["Heading2"].fontName = base_font
    styles["Heading3"].fontName = base_font

    styles["Title"].fontSize = 18
    styles["Heading1"].fontSize = 14
    styles["Heading2"].fontSize = 12
    styles["Heading3"].fontSize = 11

    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10,
        leading=14,
        wordWrap="CJK",
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=9,
        leading=12,
        wordWrap="CJK",
    )
    code = ParagraphStyle(
        "Code",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=8,
        leading=10,
        wordWrap="CJK",
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    elements: List[Any] = []

    elements.append(Paragraph("CodeAssistant 报告", styles["Title"]))
    elements.append(Spacer(1, 12))

    if review:
        findings = review.get("findings", []) or []
        elements.append(Paragraph("1. 代码审查（Review）", styles["Heading1"]))
        elements.append(Spacer(1, 6))

        sev = Counter([f.get("severity") for f in findings])
        tool = Counter([f.get("tool") for f in findings])
        ds = Counter([f.get("rule") for f in findings if f.get("tool") == "ds-rule"])

        elements.append(Paragraph("概览", styles["Heading2"]))
        overview_rows = [
            ["问题总数", str(len(findings))],
            ["高/中/低", f"{sev.get('high', 0)}/{sev.get('medium', 0)}/{sev.get('low', 0)}"],
            ["工具数", str(len(tool))],
            ["DS 规则数", str(sum(ds.values()))],
        ]
        elements.append(_make_table([["指标", "值"]] + overview_rows, base_font))
        elements.append(Spacer(1, 8))

        if sev:
            elements.append(Paragraph("严重性分布", styles["Heading2"]))
            elements.append(_make_table(_counter_rows("严重性", sev), base_font))
            elements.append(Spacer(1, 8))

        if tool:
            elements.append(Paragraph("工具分布", styles["Heading2"]))
            elements.append(_make_table(_counter_rows("工具", tool), base_font))
            elements.append(Spacer(1, 8))

        if ds:
            elements.append(Paragraph("DS 规则分布", styles["Heading2"]))
            elements.append(_make_table(_counter_rows("规则", ds), base_font))
            elements.append(Spacer(1, 8))

        if findings:
            elements.append(Paragraph("Top 20 问题", styles["Heading2"]))
            headers = ["严重性", "工具", "规则", "位置", "说明"]
            rows = [headers]
            for f in findings[:20]:
                loc = _format_loc(f)
                msg = _truncate(f.get("message", ""), 180)
                rows.append(
                    [
                        _para(f.get("severity", ""), small),
                        _para(f.get("tool", ""), small),
                        _para(f.get("rule", ""), small),
                        _para(loc, small),
                        _para(msg, small),
                    ]
                )
            elements.append(_make_table(rows, base_font, col_widths=[45, 55, 60, 160, 210], header=True))
            elements.append(Spacer(1, 8))

        raw = review.get("tool_raw") or {}
        radon = raw.get("complexity_radon") or {}
        if radon.get("stdout"):
            elements.append(Paragraph("复杂度摘要（Radon）", styles["Heading2"]))
            elements.append(Preformatted(radon.get("stdout")[:6000], code))
            elements.append(Spacer(1, 8))

    if testgen:
        elements.append(Paragraph("2. 测试生成（TestGen）", styles["Heading1"]))
        elements.append(Spacer(1, 6))
        rows = [
            ["写入测试文件数", str(testgen.get("written_files", 0))],
            ["覆盖函数数", str(testgen.get("function_count", 0))],
            ["输出目录", str(testgen.get("output_dir", ""))],
        ]
        elements.append(_make_table([["指标", "值"]] + rows, base_font))
        elements.append(Spacer(1, 8))

        cov = testgen.get("coverage") or {}
        report = cov.get("report") or {}
        if report.get("stdout"):
            elements.append(Paragraph("覆盖率报告（coverage report -m）", styles["Heading2"]))
            elements.append(Preformatted(report.get("stdout")[:6000], code))
            elements.append(Spacer(1, 8))

    doc.build(elements)
    return buf.getvalue()


def _make_table(rows: List[List[Any]], font_name: str, col_widths=None, header: bool = True):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    tbl = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
    ]
    if header and rows:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#10A37F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
        ]
    tbl.setStyle(TableStyle(style))
    return tbl


def _counter_rows(label: str, counter: Counter) -> List[List[str]]:
    rows = [[label, "数量"]]
    for k, v in counter.most_common():
        rows.append([str(k), str(v)])
    return rows


def _format_loc(finding: Dict[str, Any]) -> str:
    file = finding.get("file") or ""
    line = finding.get("line") or ""
    if file and line:
        return f"{file}:{line}"
    return file or ""


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _para(text: str, style):
    from reportlab.platypus import Paragraph

    return Paragraph(_escape(text), style)


def _register_cjk_font(pdfmetrics, TTFont) -> Optional[str]:
    candidates = [
        ("CJK", "C:/Windows/Fonts/msyh.ttc", 0),
        ("CJK", "C:/Windows/Fonts/msyh.ttf", None),
        ("CJK", "C:/Windows/Fonts/simsun.ttc", 0),
        ("CJK", "C:/Windows/Fonts/simsun.ttf", None),
        ("CJK", "C:/Windows/Fonts/simhei.ttf", None),
        ("CJK", "/System/Library/Fonts/PingFang.ttc", 0),
        ("CJK", "/System/Library/Fonts/STHeiti Light.ttc", 0),
        ("CJK", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
        ("CJK", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ]
    for name, path, idx in candidates:
        if not Path(path).exists():
            continue
        try:
            if idx is None:
                pdfmetrics.registerFont(TTFont(name, path))
            else:
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
            return name
        except Exception:
            continue
    return None
