from __future__ import annotations
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def build_latex_report(review: Optional[Dict[str, Any]], testgen: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.extend(_latex_preamble())
    lines.append("\\begin{document}")
    lines.append("")
    lines.extend(
        [
            "\\begin{center}",
            "{\\LARGE \\textbf{CodeAssistant 报告}}\\\\[4pt]",
            "{\\normalsize （由自动审查与测试生成系统输出）}",
            "\\end{center}",
            "",
            "\\vspace{0.5em}",
            "\\tableofcontents",
            "\\vspace{0.8em}",
            "\\hrule",
            "\\vspace{1.2em}",
            "",
        ]
    )

    if review:
        findings = review.get("findings", []) or []
        sev = Counter([f.get("severity") for f in findings])
        tool = Counter([f.get("tool") for f in findings])
        ds = Counter([f.get("rule") for f in findings if f.get("tool") == "ds-rule"])

        lines.append("\\section{代码审查（Review）}")
        lines.append("")

        lines.append("\\subsection{概览}")
        lines.append("\\begin{itemize}")
        lines.append(f"  \\item 问题总数：{len(findings)}")
        lines.append(
            f"  \\item 高/中/低：{sev.get('high', 0)} / {sev.get('medium', 0)} / {sev.get('low', 0)}"
        )
        lines.append(f"  \\item 工具数：{len(tool)}")
        lines.append(f"  \\item DS 规则命中总数：{sum(ds.values())}")
        lines.append("\\end{itemize}")
        lines.append("")

        if sev:
            lines.append("\\subsection{严重性分布}")
            lines.append("\\begin{itemize}")
            for k, v in sev.most_common():
                lines.append(f"  \\item {latex_escape(str(k))}：{v}")
            lines.append("\\end{itemize}")
            lines.append("")

        if tool:
            lines.append("\\subsection{工具分布}")
            lines.append("\\begin{itemize}")
            for k, v in tool.most_common():
                lines.append(f"  \\item {latex_escape(str(k))}：{v}")
            lines.append("\\end{itemize}")
            lines.append("")

        if ds:
            lines.append("\\subsection{DS 规则分布}")
            lines.append("\\begin{itemize}")
            for k, v in ds.most_common():
                lines.append(f"  \\item {latex_escape(str(k))}：{v}")
            lines.append("\\end{itemize}")
            lines.append("")

        if findings:
            lines.append("\\subsection{Top 20 问题}")
            lines.append("\\begin{enumerate}")
            for f in findings[:20]:
                sev_text = latex_escape(str(f.get("severity", "")))
                tool_text = latex_escape(str(f.get("tool", "")))
                rule_text = latex_escape(str(f.get("rule", "")))
                msg = latex_escape(str(f.get("message", "")))
                loc = format_loc(f)
                lines.append(f"  \\item \\textbf{{[{sev_text}] {tool_text} {rule_text}}} \\\\")
                if loc:
                    lines.append(f"  位置：{latex_path(loc)} \\\\")
                lines.append(f"  说明：{msg}")
                lines.append("")
            lines.append("\\end{enumerate}")
            lines.append("")

        radon = (review.get("tool_raw") or {}).get("complexity_radon") or {}
        if radon.get("stdout"):
            lines.append("\\subsection{复杂度摘要}")
            lines.append("\\begin{itemize}")
            lines.append("  \\item 来源：Radon Cyclomatic Complexity（CC）")
            lines.append("  \\item 说明：等级通常为 A（简单）到 F（复杂），括号内为复杂度分数")
            lines.append("\\end{itemize}")
            lines.append("")
            rows = parse_radon_rows(radon.get("stdout") or "")
            if not rows:
                rows = [("N/A", "-", "N/A", "-", "-")]
            lines.append("\\small")
            lines.append("\\setlength{\\emergencystretch}{2em}")
            lines.append("\\begin{longtable}{T{6.6cm} L{1.1cm} T{5.2cm} L{0.9cm} r}")
            lines.append("\\toprule")
            lines.append("文件 & 类型 & 符号（函数/方法） & 等级 & 分数 \\\\")
            lines.append("\\midrule")
            lines.append("\\endfirsthead")
            lines.append("\\toprule")
            lines.append("文件 & 类型 & 符号（函数/方法） & 等级 & 分数 \\\\")
            lines.append("\\midrule")
            lines.append("\\endhead")
            lines.append("\\midrule")
            lines.append("\\multicolumn{5}{r}{（续下页）}\\\\")
            lines.append("\\endfoot")
            lines.append("\\bottomrule")
            lines.append("\\endlastfoot")
            for row in rows:
                fpath, ftype, symbol, grade, score = row
                # 文件路径用 latex_path 处理（已包含转义），符号和等级用 latex_escape
                lines.append(
                    f"{latex_path(fpath)} & {latex_escape(ftype)} & {latex_escape(symbol)} & "
                    f"{latex_escape(grade)} & {latex_escape(score)} \\\\"
                )
            lines.append("\\end{longtable}")
            lines.append("\\normalsize")
            lines.append("")

    if testgen:
        lines.append("\\section{测试生成（TestGen）}")
        lines.append("")
        lines.append("\\subsection{指标}")
        lines.append("\\begin{itemize}")
        lines.append(f"  \\item 写入测试文件数：{testgen.get('written_files', 0)}")
        lines.append(f"  \\item 覆盖函数数：{testgen.get('function_count', 0)}")
        out_dir = testgen.get("output_dir", "")
        if out_dir:
            lines.append(f"  \\item 输出目录：{latex_path(str(out_dir))}")
        lines.append("\\end{itemize}")
        lines.append("")

        cov = testgen.get("coverage") or {}
        report = cov.get("report") or {}
        if report.get("stdout"):
            lines.append("\\subsection{覆盖率报告}")
            rows = parse_coverage_rows(report.get("stdout") or "")
            if not rows:
                rows = [("N/A", "0", "0", "0%", "")]
            lines.append("\\small")
            lines.append("\\setlength{\\emergencystretch}{2em}")
            lines.append("\\begin{longtable}{T{7.2cm} r r r T{5.2cm}}")
            lines.append("\\toprule")
            lines.append("Name & Stmts & Miss & Cover & Missing \\\\")
            lines.append("\\midrule")
            lines.append("\\endfirsthead")
            lines.append("\\toprule")
            lines.append("Name & Stmts & Miss & Cover & Missing \\\\")
            lines.append("\\midrule")
            lines.append("\\endhead")
            lines.append("\\midrule")
            lines.append("\\multicolumn{5}{r}{（续下页）}\\\\")
            lines.append("\\endfoot")
            lines.append("\\bottomrule")
            lines.append("\\endlastfoot")
            for name, stmts, miss, cover, missing in rows:
                if str(name).strip().upper() == "TOTAL":
                    name_cell = "\\textbf{TOTAL}"
                else:
                    name_cell = latex_path(name)
                miss_cell = latex_escape(missing) if missing else ""
                lines.append(
                    f"{name_cell} & {latex_escape(stmts)} & {latex_escape(miss)} & "
                    f"{latex_escape(cover)} & {miss_cell} \\\\"
                )
            lines.append("\\end{longtable}")
            lines.append("\\normalsize")
            lines.append("")

    if not review and not testgen:
        lines.append("\\section*{No results}")

    lines.append("\\end{document}")
    return "\n".join(lines)


def _latex_preamble() -> List[str]:
    return [
        "% !TeX program = xelatex",
        "% !TeX encoding = UTF-8",
        "\\documentclass[11pt,a4paper]{article}",
        "\\usepackage{array}",
        "\\newcolumntype{L}[1]{>{\\raggedright\\arraybackslash}p{#1}}",
        "\\usepackage{xurl}",
        "\\usepackage{longtable}",
        "\\usepackage{booktabs}",
        "\\newcolumntype{T}[1]{>{\\ttfamily\\raggedright\\arraybackslash}p{#1}}",
        "\\newcommand{\\codepath}[1]{\\path{#1}}",
        "",
        "% ===== 中文与字体（支持UTF-8和中文路径） =====",
        "\\usepackage{xeCJK}",
        "\\setCJKmainfont{SimSun}",
        "\\setmonofont{Consolas}",
        "\\setCJKmonofont{SimSun}",
        "\\usepackage{fontspec}",
        "",
        "% ===== 页面与排版 =====",
        "\\usepackage[a4paper,margin=2.2cm]{geometry}",
        "\\usepackage{setspace}",
        "\\setstretch{1.12}",
        "\\usepackage{microtype}",
        "",
        "% ===== 数学/符号 =====",
        "\\usepackage{amsmath,amssymb}",
        "",
        "% ===== 链接 =====",
        "\\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue]{hyperref}",
        "",
        "% ===== 表格 =====",
        "\\usepackage{tabularx}",
        "\\usepackage{multirow}",
        "\\newcolumntype{Y}{>{\\raggedright\\arraybackslash}X}",
        "\\newcolumntype{C}{>{\\centering\\arraybackslash}X}",
        "",
        "% ===== 代码块 =====",
        "\\usepackage{xcolor}",
        "\\usepackage{listings}",
        "\\lstset{",
        "  basicstyle=\\ttfamily\\small,",
        "  columns=fullflexible,",
        "  breaklines=true,",
        "  breakatwhitespace=true,",
        "  showstringspaces=false,",
        "  frame=single,",
        "  framerule=0.3pt,",
        "  rulecolor=\\color{black!25},",
        "  xleftmargin=1.2em,",
        "  xrightmargin=0.6em,",
        "  aboveskip=0.8em,",
        "  belowskip=0.8em",
        "}",
        "",
        "% ===== 标题格式 =====",
        "\\usepackage{titlesec}",
        "\\titleformat{\\section}{\\Large\\bfseries}{\\thesection}{0.6em}{}",
        "\\titleformat{\\subsection}{\\large\\bfseries}{\\thesubsection}{0.6em}{}",
        "\\titleformat{\\subsubsection}{\\normalsize\\bfseries}{\\thesubsubsection}{0.6em}{}",
        "",
    ]


def latex_escape(text: str) -> str:
    text = str(text or "")
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def latex_path(path: str) -> str:
    """处理路径，支持中文字符
    
    注意：\\path{} 命令（通过 \\codepath 别名）会自动处理特殊字符，
    包括下划线、井号等，因此不需要手动转义这些字符。
    只需要移除可能破坏 LaTeX 语法的字符如花括号。
    """
    safe = str(path or "").replace("\\", "/")
    # 移除可能破坏 LaTeX 语法的字符
    safe = safe.replace("{", "").replace("}", "")
    # 不需要调用 latex_escape，因为 \path{} 命令会自动处理特殊字符
    return f"\\codepath{{{safe}}}"


def format_loc(finding: Dict[str, Any]) -> str:
    file = finding.get("file") or ""
    line = finding.get("line") or ""
    if file and line:
        return f"{file}:{line}"
    return file or ""


def parse_radon_rows(stdout: str) -> List[Tuple[str, str, str, str, str]]:
    rows: List[Tuple[str, str, str, str, str]] = []
    current_file = ""
    for line in stdout.splitlines():
        if not line.strip():
            continue
        if not line[:1].isspace():
            current_file = line.strip()
            continue
        m = re.match(r"\s*([A-Z])\s+\d+:\d+\s+(.+?)\s+-\s+([A-F])\s+\((\d+)\)", line)
        if m and current_file:
            ftype, symbol, grade, score = m.group(1), m.group(2), m.group(3), m.group(4)
            rows.append((current_file, ftype, symbol, grade, score))
    return rows


def parse_coverage_rows(stdout: str) -> List[Tuple[str, str, str, str, str]]:
    rows: List[Tuple[str, str, str, str, str]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        if line.startswith("Name") or set(line.strip()) == {"-"}:
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 4:
            continue
        name, stmts, miss, cover = parts[:4]
        missing = parts[4] if len(parts) > 4 else ""
        rows.append((name.strip(), stmts, miss, cover, missing.strip()))
    return rows

