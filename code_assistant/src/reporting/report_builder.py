from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional


def _truncate(text: str, limit: int = 160) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _md_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _counter_rows(counter: Counter) -> List[List[str]]:
    return [[str(k), str(v)] for k, v in counter.most_common()]


def _format_loc(finding: Dict[str, Any]) -> str:
    file = finding.get("file") or ""
    line = finding.get("line") or ""
    if file and line:
        return f"{file}:{line}"
    return file or ""


def build_markdown_report(review: Optional[Dict[str, Any]], testgen: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = ["# CodeAssistant 报告", ""]

    if review:
        findings = review.get("findings", []) or []
        lines += ["## 1. 代码审查（Review）", ""]

        sev = Counter([f.get("severity") for f in findings])
        tool = Counter([f.get("tool") for f in findings])
        ds = Counter([f.get("rule") for f in findings if f.get("tool") == "ds-rule"])

        lines += ["### 概览", ""]
        overview = [
            ["问题总数", str(len(findings))],
            ["高/中/低", f"{sev.get('high', 0)}/{sev.get('medium', 0)}/{sev.get('low', 0)}"],
            ["工具数", str(len(tool))],
            ["DS 规则数", str(sum(ds.values()))],
        ]
        lines += _md_table(["指标", "值"], overview)
        lines.append("")

        if sev:
            lines += ["### 严重性分布", ""]
            lines += _md_table(["严重性", "数量"], _counter_rows(sev))
            lines.append("")

        if tool:
            lines += ["### 工具分布", ""]
            lines += _md_table(["工具", "数量"], _counter_rows(tool))
            lines.append("")

        if ds:
            lines += ["### DS 规则分布", ""]
            lines += _md_table(["规则", "数量"], _counter_rows(ds))
            lines.append("")

        if findings:
            lines += ["### Top 20 问题", ""]
            rows = []
            for f in findings[:20]:
                rows.append(
                    [
                        str(f.get("severity") or ""),
                        str(f.get("tool") or ""),
                        str(f.get("rule") or ""),
                        _format_loc(f),
                        _truncate(f.get("message", ""), 180),
                    ]
                )
            lines += _md_table(["严重性", "工具", "规则", "位置", "说明"], rows)
            lines.append("")

        raw = review.get("tool_raw") or {}
        radon = raw.get("complexity_radon") or {}
        if radon.get("stdout"):
            lines += ["### 复杂度摘要（Radon）", "", "```", (radon.get("stdout") or "")[:8000], "```", ""]

    if testgen:
        lines += ["## 2. 测试生成（TestGen）", ""]
        rows = [
            ["写入测试文件数", str(testgen.get("written_files", 0))],
            ["覆盖函数数", str(testgen.get("function_count", 0))],
            ["输出目录", str(testgen.get("output_dir", ""))],
        ]
        lines += _md_table(["指标", "值"], rows)
        lines.append("")

        cov = testgen.get("coverage") or {}
        report = cov.get("report") or {}
        if report.get("stdout"):
            lines += ["### 覆盖率报告（coverage report -m）", "", "```", (report.get("stdout") or "")[:8000], "```", ""]

    if not review and not testgen:
        lines.append("_No results._")

    return "\n".join(lines)
