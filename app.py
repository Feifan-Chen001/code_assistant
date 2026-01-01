from __future__ import annotations
import base64
import io
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st

from src.core.config import load_config
from src.core.orchestrator import Orchestrator
from src.core.subproc import run_cmd
from src.reporting.report_builder import build_markdown_report
from src.reporting.pdf_builder import build_pdf_report
from src.reporting.latex_builder import build_latex_report


def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap");

:root {
  --bg: #f7f7f8;
  --panel: #ffffff;
  --ink: #0f172a;
  --muted: #64748b;
  --accent: #10a37f;
  --accent-strong: #0c8b6b;
  --border: #e2e8f0;
  --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}

.stApp {
  background:
    radial-gradient(1100px 540px at 10% -10%, #e6f6f1 0%, transparent 60%),
    radial-gradient(900px 500px at 90% 0%, #fff4e5 0%, transparent 55%),
    var(--bg);
  color: var(--ink);
  font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
}

div.block-container {
  padding-top: 2rem;
  padding-bottom: 4rem;
  max-width: 1180px;
  animation: rise 0.6s ease-out;
}

section[data-testid="stSidebar"] {
  background: #f0f2f4;
  border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
  color: var(--ink);
}

section[data-testid="stSidebar"] .stTextInput input {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.55rem 0.75rem;
}

.stButton button {
  width: 100%;
  border-radius: 999px;
  border: 1px solid transparent;
  background: linear-gradient(135deg, var(--accent), #22c28f);
  color: #ffffff;
  font-weight: 600;
  padding: 0.6rem 1.2rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

.stButton button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(16, 163, 127, 0.2);
  background: linear-gradient(135deg, var(--accent-strong), #1aa37a);
}

h1, h2, h3 {
  letter-spacing: -0.02em;
}

.hero {
  padding: 1.5rem 1.75rem;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
  margin-bottom: 1.5rem;
}

.hero-title {
  font-size: 2.1rem;
  font-weight: 700;
}

.hero-subtitle {
  margin-top: 0.35rem;
  color: var(--muted);
  font-size: 1rem;
}

div[data-testid="column"] > div {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem;
  box-shadow: var(--shadow);
}

div[data-testid="column"] {
  animation: rise 0.6s ease-out;
}

div[data-testid="column"]:nth-child(1) { animation-delay: 0.05s; }
div[data-testid="column"]:nth-child(2) { animation-delay: 0.1s; }
div[data-testid="column"]:nth-child(3) { animation-delay: 0.15s; }

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.75rem;
  box-shadow: var(--shadow);
}

code, pre {
  font-family: "JetBrains Mono", "Fira Code", monospace;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _ensure_dirs(out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def _parse_repo_inputs(text: str) -> List[str]:
    items = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        items.append(raw)
    return items


def _is_repo_root(path: Path) -> bool:
    markers = [".git", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"]
    return any((path / m).exists() for m in markers)


def _expand_local_repos(path: Path) -> List[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    subrepos = [p for p in path.iterdir() if p.is_dir() and _is_repo_root(p)]
    if len(subrepos) >= 2:
        return sorted(subrepos, key=lambda p: p.name.lower())
    if _is_repo_root(path):
        return [path]
    if subrepos:
        return sorted(subrepos, key=lambda p: p.name.lower())
    return [path]


def _unique_name(name: str, used: Dict[str, int]) -> str:
    base = name or "repo"
    if base not in used:
        used[base] = 1
        return base
    used[base] += 1
    return f"{base}-{used[base]}"


def _is_github_url(value: str) -> bool:
    value = value.strip().lower()
    return value.startswith("https://github.com/") or value.startswith("http://github.com/") or value.startswith(
        "git@github.com:"
    )


def _github_slug(url: str) -> Optional[str]:
    url = url.strip()
    if url.startswith("git@github.com:"):
        path = url.split("git@github.com:", 1)[1]
    else:
        parsed = urlparse(url)
        if parsed.netloc not in {"github.com", "www.github.com"}:
            return None
        path = parsed.path.lstrip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    repo = repo.replace(".git", "")
    return f"{owner}__{repo}"


def _resolve_repo_input(repo_input: str, cache_dir: str) -> Optional[str]:
    repo_input = repo_input.strip()
    if not repo_input:
        st.error("Repository path or URL is empty.")
        return None
    if not _is_github_url(repo_input):
        return repo_input
    slug = _github_slug(repo_input)
    if not slug:
        st.error("Invalid GitHub repository URL.")
        return None
    cache_root = Path(cache_dir).expanduser().resolve()
    dest = cache_root / slug
    if dest.exists():
        st.info(f"Using cached repo: {dest}")
        return str(dest)
    cache_root.mkdir(parents=True, exist_ok=True)
    st.info(f"Cloning {repo_input} ...")
    res = run_cmd(["git", "clone", "--depth", "1", repo_input, str(dest)])
    if not res["ok"]:
        st.error(f"Git clone failed: {res['stderr'] or res['stdout']}")
        return None
    return str(dest)


def _prepare_cfg(cfg: Dict[str, Any], test_out: Optional[Path]) -> Dict[str, Any]:
    cfg = dict(cfg)
    if test_out is not None:
        cfg.setdefault("testgen", {})
        cfg["testgen"] = dict(cfg["testgen"])
        cfg["testgen"]["output_dir"] = str(test_out)
    return cfg


def _resolve_repo_inputs(repo_text: str, cache_dir: str) -> List[Dict[str, str]]:
    items = _parse_repo_inputs(repo_text)
    repos: List[Dict[str, str]] = []
    used: Dict[str, int] = {}
    for raw in items:
        if _is_github_url(raw):
            resolved = _resolve_repo_input(raw, cache_dir)
            if not resolved:
                continue
            slug = _github_slug(raw) or Path(resolved).name
            name = _unique_name(slug, used)
            repos.append({"name": name, "path": resolved, "source": raw})
            continue
        p = Path(raw).expanduser()
        if not p.exists():
            st.error(f"Repository path not found: {raw}")
            continue
        for local in _expand_local_repos(p):
            name = _unique_name(local.name or "repo", used)
            repos.append({"name": name, "path": str(local.resolve()), "source": raw})
    return repos


def _markdown_to_text(md_text: str) -> str:
    lines = []
    for line in md_text.splitlines():
        if line.startswith("#"):
            line = line.lstrip("#").strip().upper()
        line = line.replace("**", "").replace("`", "")
        lines.append(line)
    return "\n".join(lines)


def _build_pdf_from_markdown(md_text: str) -> Optional[bytes]:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except Exception:
        return None
    text = _markdown_to_text(md_text)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter
    margin = 0.75 * inch
    x = margin
    y = height - margin
    line_height = 12
    wrap_width = 110
    for line in text.splitlines():
        chunks = textwrap.wrap(line, width=wrap_width, replace_whitespace=False, drop_whitespace=False) or [""]
        for chunk in chunks:
            if y <= margin:
                c.showPage()
                y = height - margin
            c.drawString(x, y, chunk)
            y -= line_height
    c.save()
    return buf.getvalue()


def _render_pdf_preview(pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    html = (
        f'<iframe src="data:application/pdf;base64,{b64}" '
        'width="100%" height="720" style="border: 1px solid #e2e8f0; border-radius: 14px;"></iframe>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _compile_latex(tex_path: Path) -> Optional[Path]:
    pdf_path = tex_path.with_suffix(".pdf")
    res = run_cmd(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=str(tex_path.parent),
    )
    if res["ok"] and pdf_path.exists():
        return pdf_path
    res = run_cmd(["tectonic", tex_path.name], cwd=str(tex_path.parent))
    if res["ok"] and pdf_path.exists():
        return pdf_path
    return None


def _load_report_sources(out_dir: str, state):
    review = state.get("last_review")
    testgen = state.get("last_testgen")
    review_path = Path(out_dir) / "review.json"
    testgen_path = Path(out_dir) / "testgen.json"
    if review is None and review_path.exists():
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except Exception:
            review = None
    if testgen is None and testgen_path.exists():
        try:
            testgen = json.loads(testgen_path.read_text(encoding="utf-8"))
        except Exception:
            testgen = None
    return review, testgen


def _make_pdf_bytes(review, testgen, md_text: str) -> Optional[bytes]:
    pdf_bytes = build_pdf_report(review, testgen)
    if pdf_bytes is not None:
        return pdf_bytes
    return _build_pdf_from_markdown(md_text)


def _write_report(out_dir: str, review, testgen) -> tuple[Path, Optional[Path]]:
    _ensure_dirs(out_dir)
    md = build_markdown_report(review, testgen)
    report_path = Path(out_dir) / "report.md"
    report_path.write_text(md, encoding="utf-8")
    tex_path = Path(out_dir) / "report.tex"
    tex_path.write_text(build_latex_report(review, testgen), encoding="utf-8")
    pdf_path = _compile_latex(tex_path)
    if pdf_path:
        return report_path, pdf_path
    pdf_bytes = _make_pdf_bytes(review, testgen, md)
    if pdf_bytes:
        fallback_path = Path(out_dir) / "report.pdf"
        fallback_path.write_bytes(pdf_bytes)
        return report_path, fallback_path
    return report_path, None


def _existing_report_path(out_dir: str, state) -> Optional[Path]:
    last = state.get("last_report_path")
    if last and Path(last).exists():
        return Path(last)
    candidate = Path(out_dir) / "report.md"
    if candidate.exists():
        return candidate
    return None


def _plot_counts(values: List[str], title: str) -> None:
    values = [v for v in values if v is not None]
    if not values:
        st.info("No data to display.")
        return
    counts = Counter(values)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    x = [k for k, _ in items]
    y = [v for _, v in items]
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Bar(x=x, y=y)])
        fig.update_layout(title=title, yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
        return
    except Exception:
        pass
    st.markdown(f"**{title}**")
    max_count = max(y) if y else 0
    lines = []
    for k, v in items:
        bar_len = 0 if max_count == 0 else int(v / max_count * 20)
        bar = "#" * bar_len
        lines.append(f"{k}: {bar} ({v})")
    st.text("\n".join(lines))


def _show_findings_table(rows) -> None:
    try:
        import pandas as pd

        if getattr(pd, "DataFrame", None) is None or getattr(pd, "__version__", None) is None:
            raise RuntimeError("pandas not usable")
    except Exception:
        st.json(rows)
        return
    st.dataframe(rows, width="stretch")


def main() -> None:
    st.set_page_config(page_title="CodeAssistant", layout="wide")
    _inject_css()

    st.markdown(
        """
<div class="hero">
  <div class="hero-title">CodeAssistant</div>
  <div class="hero-subtitle">Data science review, test generation, and reporting in one flow.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Workspace")
        cfg_path = st.text_input("Config file", value="config.yaml")
        repo_input = st.text_area("Repository paths or GitHub URLs (one per line)", value="my_repo", height=120)
        cache_dir = st.text_input("GitHub cache folder", value="Git_repo")
        st.caption("One repo per line. Local folders with multiple repos will be expanded automatically.")
        out_dir = st.text_input("Output folder", value="reports")
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown("- Review\n- Test generation\n- Reports")
        st.caption("Run full workflows from the main panel.")

    cfg = load_config(cfg_path)
    st.markdown("### Actions")
    col_a, col_b, col_c = st.columns(3, gap="large")
    with col_a:
        st.markdown("**Review**")
        st.caption("Static analysis, security, and DS rules.")
        run_review = st.button("Run Review", width="stretch")
    with col_b:
        st.markdown("**Test Generation**")
        st.caption("Generate unit tests and coverage report.")
        run_testgen = st.button("Run TestGen", width="stretch")
    with col_c:
        st.markdown("**Full Pipeline**")
        st.caption("Review + tests + report in one run.")
        run_all = st.button("Run All", width="stretch")

    st.divider()

    state = st.session_state
    state.setdefault("last_review", None)
    state.setdefault("last_testgen", None)
    state.setdefault("last_report_path", None)
    state.setdefault("last_report_pdf", None)
    state.setdefault("batch_results", [])

    repo_jobs: List[Dict[str, str]] = []
    if run_review or run_testgen or run_all:
        repo_inputs = _parse_repo_inputs(repo_input)
        repo_jobs = _resolve_repo_inputs(repo_input, cache_dir)
        if not repo_jobs:
            st.stop()
        if len(repo_jobs) > 1:
            st.info(f"Batch mode: {len(repo_jobs)} repositories.")
        elif repo_inputs and repo_jobs[0]["path"] != repo_inputs[0]:
            st.caption(f"Resolved repository: {repo_jobs[0]['path']}")

        state["batch_results"] = []
        batch_mode = len(repo_jobs) > 1
        if batch_mode:
            state["last_review"] = None
            state["last_testgen"] = None
            state["last_report_path"] = None
            state["last_report_pdf"] = None

        for job in repo_jobs:
            repo_out = Path(out_dir) / job["name"] if batch_mode else Path(out_dir)
            _ensure_dirs(str(repo_out))

            test_out = None
            if batch_mode and (run_testgen or run_all):
                test_out = repo_out / "generated_tests"

            cfg_run = _prepare_cfg(cfg, test_out)
            orch = Orchestrator(cfg_run)

            review = None
            if run_review or run_all:
                with st.spinner(f"Running review: {job['name']}"):
                    review = orch.run_review(repo_path=job["path"])
                (repo_out / "review.json").write_text(
                    json.dumps(review, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.success(f"[{job['name']}] Review complete: {len(review['findings'])} findings.")

            testgen = None
            if run_testgen or run_all:
                with st.spinner(f"Generating tests: {job['name']}"):
                    testgen = orch.run_testgen(repo_path=job["path"])
                (repo_out / "testgen.json").write_text(
                    json.dumps(testgen, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.success(
                    f"[{job['name']}] Test generation complete: {testgen['written_files']} files written to "
                    f"{testgen['output_dir']}."
                )

            report_path = None
            pdf_path = None
            if run_all:
                with st.spinner(f"Building report: {job['name']}"):
                    report_path, pdf_path = _write_report(str(repo_out), review, testgen)
                st.success(f"[{job['name']}] Report saved: {report_path}")

            if not batch_mode:
                if review is not None:
                    state["last_review"] = review
                if testgen is not None:
                    state["last_testgen"] = testgen
                if report_path:
                    state["last_report_path"] = str(report_path)
                    state["last_report_pdf"] = str(pdf_path) if pdf_path else None

            state["batch_results"].append(
                {
                    "name": job["name"],
                    "path": job["path"],
                    "out_dir": str(repo_out),
                    "review_count": len(review["findings"]) if review else None,
                    "testgen_written": testgen.get("written_files") if testgen else None,
                    "report_path": str(report_path) if report_path else None,
                    "pdf_path": str(pdf_path) if pdf_path else None,
                }
            )

    batch_results = state.get("batch_results") or []
    batch_mode = len(batch_results) > 1
    active_out_dir = out_dir
    active_review = state.get("last_review")
    active_testgen = state.get("last_testgen")
    active_item = None

    if batch_mode:
        st.markdown("### Batch results")
        for item in batch_results:
            review_part = (
                f"{item['review_count']} findings" if item.get("review_count") is not None else "review: n/a"
            )
            test_part = (
                f"{item['testgen_written']} tests" if item.get("testgen_written") is not None else "testgen: n/a"
            )
            report_part = "report ready" if item.get("report_path") else "no report"
            st.markdown(f"- **{item['name']}** · {review_part} · {test_part} · {report_part}")
        labels = [item["name"] for item in batch_results]
        active_name = st.selectbox("Inspect repository", labels, key="batch_select")
        active_item = next((i for i in batch_results if i["name"] == active_name), None)
        if active_item:
            active_out_dir = active_item["out_dir"]
            active_review, active_testgen = _load_report_sources(active_out_dir, {})

    review = active_review
    if review:
        title = "### Review Board"
        if batch_mode and active_item:
            title = f"### Review Board ({active_item['name']})"
        findings = review["findings"]
        st.markdown(title)
        if len(findings) == 0:
            st.info("No findings (or scanners disabled).")
        else:
            ds_rules = [f.get("rule") for f in findings if f.get("tool") == "ds-rule"]
            metric_cols = st.columns(3, gap="large")
            metric_cols[0].metric("Findings", len(findings))
            metric_cols[1].metric("DS findings", len(ds_rules))
            metric_cols[2].metric("Tools", len({f.get('tool') for f in findings}))

            sev = [f["severity"] for f in findings]
            _plot_counts(sev, "Severity distribution")

            tools = [f["tool"] for f in findings]
            _plot_counts(tools, "Tool distribution")

            if ds_rules:
                _plot_counts(ds_rules, "DS Rule Distribution")

            st.markdown("#### Top 20 findings")
            _show_findings_table(findings[:20])

    testgen = active_testgen
    if testgen:
        title = "### Test Generation"
        if batch_mode and active_item:
            title = f"### Test Generation ({active_item['name']})"
        st.markdown(title)
        st.json({k: v for k, v in testgen.items() if k != "generated"})

    report_title = "### Report"
    if batch_mode and active_item:
        report_title = f"### Report ({active_item['name']})"
    st.markdown(report_title)

    rp = None
    if batch_mode and active_item:
        candidate = Path(active_out_dir) / "report.md"
        if candidate.exists():
            rp = candidate
    else:
        rp = _existing_report_path(out_dir, state)

    if not rp:
        if active_review or active_testgen:
            button_label = "Build report for selected repo" if batch_mode else "Build report from latest run"
            if st.button(button_label, key="build_report"):
                report_path, pdf_path = _write_report(active_out_dir, active_review, active_testgen)
                if batch_mode and active_item:
                    active_item["report_path"] = str(report_path)
                    active_item["pdf_path"] = str(pdf_path) if pdf_path else None
                else:
                    state["last_report_path"] = str(report_path)
                    state["last_report_pdf"] = str(pdf_path) if pdf_path else None
                rp = report_path
        else:
            st.info("Run Review/TestGen/All to generate a report.")

    if rp and rp.exists():
        source_state = {} if batch_mode else state
        review_src, testgen_src = _load_report_sources(active_out_dir, source_state)
        if batch_mode and active_item:
            pdf_path = active_item.get("pdf_path")
        else:
            pdf_path = state.get("last_report_pdf")
        pdf_bytes = None
        if pdf_path and Path(pdf_path).exists():
            pdf_bytes = Path(pdf_path).read_bytes()
        else:
            md_text = rp.read_text(encoding="utf-8")
            tex_path = rp.with_suffix(".tex")
            if not tex_path.exists():
                tex_path.write_text(build_latex_report(review_src, testgen_src), encoding="utf-8")
            compiled = _compile_latex(tex_path)
            if compiled:
                pdf_path = compiled
                pdf_bytes = compiled.read_bytes()
                if batch_mode and active_item:
                    active_item["pdf_path"] = str(pdf_path)
                else:
                    state["last_report_pdf"] = str(pdf_path)
            else:
                pdf_bytes = _make_pdf_bytes(review_src, testgen_src, md_text)
                if pdf_bytes:
                    pdf_path = rp.with_suffix(".pdf")
                    pdf_path.write_bytes(pdf_bytes)
                    if batch_mode and active_item:
                        active_item["pdf_path"] = str(pdf_path)
                    else:
                        state["last_report_pdf"] = str(pdf_path)
        if pdf_bytes:
            btn_col, info_col = st.columns([1, 3], gap="large")
            file_prefix = f"{active_item['name']}_" if batch_mode and active_item else ""
            with btn_col:
                st.download_button(
                    "Download report.pdf",
                    data=pdf_bytes,
                    file_name=f"{file_prefix}report.pdf",
                    mime="application/pdf",
                )
                st.download_button(
                    "Download report.md",
                    data=rp.read_bytes(),
                    file_name=f"{file_prefix}report.md",
                )
            with info_col:
                st.caption("PDF preview")
            _render_pdf_preview(pdf_bytes)
        else:
            st.info("PDF preview unavailable. Install reportlab to enable PDF export.")


if __name__ == "__main__":
    main()














