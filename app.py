from __future__ import annotations

import base64
import io
import json
import re
import textwrap
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

from src.core.config import load_config
from src.core.config_validator import validate_config, CodeAssistantConfig
from src.core.llm_client import llm_chat, build_llm_config
from src.core.logger import setup_logger
from src.core.orchestrator import Orchestrator
from src.core.subproc import run_cmd
from src.features.review.notebook import extract_code_cells
from src.features.review.rule_plugin import get_registry
from src.reporting.latex_builder import build_latex_report
from src.reporting.pdf_builder import build_pdf_report
from src.reporting.report_builder import build_markdown_report

def _inject_css() -> None:
    # 加载背景图片并转换为 base64
    bg_image_base64 = ""
    bg_image_path = Path("background.png")
    if bg_image_path.exists():
        try:
            bg_image_base64 = base64.b64encode(bg_image_path.read_bytes()).decode()
        except Exception:
            pass  # 如果加载失败，使用默认背景
    
    # 根据是否有图片生成不同的背景样式
    if bg_image_base64:
        hero_background = f"""
  background: linear-gradient(var(--hero-overlay), var(--hero-overlay)),
              url('data:image/png;base64,{bg_image_base64}') center/cover no-repeat;
"""
    else:
        hero_background = """
  background: var(--panel);
"""
    
    st.markdown(
        f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap");


:root {{
  --bg: #f7f7f8;
  --panel: #ffffff;
  --ink: #0f172a;
  --muted: #64748b;
  --accent: #10a37f;
  --accent-strong: #0c8b6b;
  --accent-soft: #22c28f;
  --accent-glow: rgba(16, 163, 127, 0.2);
  --border: #e2e8f0;
  --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
  --sidebar-bg: #fafafa;
  --input-bg: #ffffff;
  --input-border: #dbe3ee;
  --code-bg: #f1f5f9;
  --code-border: #e2e8f0;
  --hero-overlay: rgba(255, 255, 255, 0.3);
  --hero-text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8);
  --hero-subtext-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
  --app-glow-1: radial-gradient(1000px 500px at 10% 0%, rgba(16, 163, 127, 0.18) 0%, transparent 55%);
  --app-glow-2: radial-gradient(900px 450px at 90% 0%, rgba(52, 211, 153, 0.18) 0%, transparent 55%);
  --app-glow-3: radial-gradient(700px 350px at 50% 100%, rgba(26, 188, 156, 0.14) 0%, transparent 50%);
  --sidebar-glow-1: radial-gradient(420px 420px at 0% 70%, rgba(255, 220, 120, 0.28) 0%, transparent 60%);
  --sidebar-glow-2: radial-gradient(320px 320px at 0% 0%, rgba(255, 235, 59, 0.22) 0%, transparent 50%);
}}

html, body {{
  color-scheme: light;
}}



.stApp {{
  background: var(--app-glow-1), var(--app-glow-2), var(--app-glow-3), var(--bg);
  color: var(--ink);
  font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
}}

div.block-container {{
  padding-top: 2rem;
  padding-bottom: 4rem;
  max-width: 1180px;
  animation: rise 0.6s ease-out;
}}

section[data-testid="stSidebar"] {{
  background: var(--sidebar-glow-1), var(--sidebar-glow-2), var(--sidebar-bg);
  border-right: 1px solid var(--border);
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {{
  color: var(--ink);
}}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea {{
  background: var(--input-bg);
  border: 1px solid var(--input-border);
  color: var(--ink);
  border-radius: 12px;
  padding: 0.55rem 0.75rem;
}}

section[data-testid="stSidebar"] .stTextInput input::placeholder,
section[data-testid="stSidebar"] .stTextArea textarea::placeholder {{
  color: var(--muted);
}}

.stButton button {{
  width: 100%;
  border-radius: 999px;
  border: 1px solid transparent;
  background: linear-gradient(135deg, var(--accent), var(--accent-soft));
  color: #ffffff;
  font-weight: 600;
  padding: 0.6rem 1.2rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}}

.stButton button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 10px 22px var(--accent-glow);
  background: linear-gradient(135deg, var(--accent-strong), var(--accent));
}}

.stDownloadButton button {{
  width: 100%;
  border-radius: 999px;
  border: 1px solid transparent;
  background: linear-gradient(135deg, var(--accent), var(--accent-soft)) !important;
  color: #ffffff !important;
  font-weight: 600;
  padding: 0.6rem 1.2rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}}

.stDownloadButton button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 10px 22px var(--accent-glow);
  background: linear-gradient(135deg, var(--accent-strong), var(--accent)) !important;
}}



div[data-testid="stSegmentedControl"] {{
  margin-bottom: 0.75rem;
}}

div[data-testid="stSegmentedControl"] div[role="radiogroup"] {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.2rem;
  box-shadow: var(--shadow);
}}

div[data-testid="stSegmentedControl"] button[role="radio"] {{
  border-radius: 999px;
  color: var(--muted);
  background: transparent;
}}

div[data-testid="stSegmentedControl"] button[role="radio"]:hover {{
  background: rgba(16, 163, 127, 0.12);
  color: var(--ink);
}}

div[data-testid="stSegmentedControl"] button[role="radio"][aria-checked="true"] {{
  background: linear-gradient(135deg, var(--accent), var(--accent-soft));
  color: #ffffff;
  box-shadow: 0 8px 18px var(--accent-glow);
}}
h1, h2, h3 {{
  letter-spacing: -0.02em;
}}

.hero {{
  padding: 1.5rem 1.75rem;{hero_background}
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
  margin-bottom: 1.5rem;
  position: relative;
}}

.hero-title {{
  font-size: 2.1rem;
  font-weight: 700;
  position: relative;
  z-index: 1;
  text-shadow: var(--hero-text-shadow);
}}

.hero-subtitle {{
  margin-top: 0.35rem;
  color: var(--muted);
  font-size: 1rem;
  position: relative;
  z-index: 1;
  text-shadow: var(--hero-subtext-shadow);
}}

div[data-testid="column"] > div {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem;
  box-shadow: var(--shadow);
}}

div[data-testid="column"] {{
  animation: rise 0.6s ease-out;
}}

div[data-testid="column"]:nth-child(1) {{ animation-delay: 0.05s; }}
div[data-testid="column"]:nth-child(2) {{ animation-delay: 0.1s; }}
div[data-testid="column"]:nth-child(3) {{ animation-delay: 0.15s; }}

div[data-testid="stMetric"] {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.75rem;
  box-shadow: var(--shadow);
}}

code, pre {{
  font-family: "JetBrains Mono", "Fira Code", monospace;
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 10px;
}}

@keyframes rise {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def _hide_theme_picker() -> None:
    components.html(
        """
<script>
(function () {
  function hideThemeRow() {
    const labels = window.parent.document.querySelectorAll("label");
    labels.forEach((label) => {
      const text = (label.textContent || "").replace(/\s+/g, " ").trim();
      if (text === "Choose app theme" || text === "??????") {
        const row = label.parentElement;
        if (row) {
          row.style.display = "none";
        }
      }
    });
  }

  const observer = new MutationObserver(hideThemeRow);
  observer.observe(window.parent.document.body, { childList: true, subtree: true });
  hideThemeRow();
})();
</script>
""",
        height=0,
        width=0,
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
        st.error("仓库路径或 URL 为空。")
        return None
    if not _is_github_url(repo_input):
        return repo_input
    slug = _github_slug(repo_input)
    if not slug:
        st.error("无效的 GitHub 仓库 URL。")
        return None
    cache_root = Path(cache_dir).expanduser().resolve()
    dest = cache_root / slug
    if dest.exists():
        st.info(f"使用缓存仓库：{dest}")
        return str(dest)
    cache_root.mkdir(parents=True, exist_ok=True)
    st.info(f"正在克隆 {repo_input} ...")
    res = run_cmd(["git", "clone", "--depth", "1", repo_input, str(dest)])
    if not res["ok"]:
        st.error(f"Git 克隆失败：{res['stderr'] or res['stdout']}")
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
        res2 = run_cmd(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(tex_path.parent),
        )
        if pdf_path.exists():
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
        st.info("没有数据可显示。")
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

def _truncate_text(text: str, limit: int = 12000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _normalize_plan(raw: Any) -> List[Dict[str, str]]:
    plan: List[Dict[str, str]] = []
    if isinstance(raw, dict):
        raw = raw.get("plan") or raw.get("steps") or raw.get("changes") or []
    if isinstance(raw, list):
        for idx, item in enumerate(raw, start=1):
            if isinstance(item, str):
                plan.append({"id": str(idx), "file": "", "summary": item})
            elif isinstance(item, dict):
                summary = (
                    item.get("summary")
                    or item.get("change")
                    or item.get("desc")
                    or item.get("title")
                    or ""
                )
                file = item.get("file") or item.get("path") or ""
                plan.append({"id": str(item.get("id", idx)), "file": str(file), "summary": str(summary)})
    return plan


def _fallback_plan(text: str) -> List[Dict[str, str]]:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0].isdigit() or line.startswith("-"):
            line = line.lstrip("- ")
            line = line.lstrip("0123456789.")
            line = line.strip()
        if line:
            lines.append(line)
    return [{"id": str(i + 1), "file": "", "summary": line} for i, line in enumerate(lines[:10])]


def _normalize_recommendations(raw: Any) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if isinstance(raw, dict):
        raw = raw.get("projects") or raw.get("recommendations") or raw.get("items") or []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            url = str(item.get("url") or "").strip()
            why = str(item.get("why") or item.get("pros") or item.get("summary") or "").strip()
            if name or url or why:
                items.append({"name": name, "url": url, "why": why})
    return items


def _format_file_context(files: List[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for item in files:
        path = item.get("path") or ""
        content = item.get("content") or ""
        blocks.append(f"FILE: {path}\n```python\n{content}\n```")
    return "\n\n".join(blocks)


def _collect_context_files(
    repo_root: Path,
    review: Optional[Dict[str, Any]],
    max_files: int = 6,
    max_chars: int = 4000,
) -> List[Dict[str, str]]:
    if not review:
        return []
    findings = review.get("findings", []) or []
    seen: set[Path] = set()
    out: List[Dict[str, str]] = []
    for finding in findings:
        file_path = finding.get("file")
        if not file_path:
            continue
        file_path = str(file_path).split("#", 1)[0]
        p = Path(file_path)
        if not p.is_absolute():
            p = repo_root / file_path
        try:
            p = p.resolve()
        except Exception:
            continue
        if repo_root not in p.parents and p != repo_root:
            continue
        if not p.exists() or p in seen:
            continue
        if p.suffix == ".ipynb":
            cells = extract_code_cells(p)
            content = "\n\n".join([f"# cell {idx}\n{code}" for idx, code in cells])
        else:
            content = p.read_text(encoding="utf-8", errors="ignore")
        content = content.strip()
        if not content:
            continue
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]..."
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        out.append({"path": rel, "content": content})
        seen.add(p)
        if len(out) >= max_files:
            break
    return out


def _llm_ready(cfg: Dict[str, Any]) -> tuple[bool, str]:
    llm_cfg = build_llm_config(cfg)
    if not llm_cfg.get("api_key") and not llm_cfg.get("allow_empty_key"):
        env_name = llm_cfg.get("api_key_env") or "OPENAI_API_KEY"
        return False, f"Missing API key. Set {env_name} or llm.api_key."
    return True, ""


def _apply_llm_changes(
    repo_root: Path,
    files: List[Dict[str, Any]],
    allow_new: bool = False,
) -> tuple[List[Path], List[str]]:
    allowed_exts = {".py", ".md", ".yaml", ".yml", ".txt", ".json", ".toml", ".ini", ".cfg", ".ipynb"}
    changed: List[Path] = []
    skipped: List[str] = []
    for item in files:
        path_val = str(item.get("path") or "").strip()
        content = item.get("content")
        if not path_val or content is None:
            skipped.append(path_val or "<empty>")
            continue
        p = Path(path_val)
        if p.is_absolute():
            try:
                rel = p.relative_to(repo_root)
            except Exception:
                skipped.append(path_val)
                continue
        else:
            rel = p
        target = (repo_root / rel).resolve()
        if repo_root not in target.parents and target != repo_root:
            skipped.append(path_val)
            continue
        if target.suffix not in allowed_exts:
            skipped.append(path_val)
            continue
        if not target.exists() and not allow_new:
            skipped.append(path_val)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        new_text = str(content)
        if target.exists():
            try:
                old_text = target.read_text(encoding="utf-8", errors="ignore")
                if old_text == new_text:
                    continue
            except Exception:
                pass
        target.write_text(new_text, encoding="utf-8")
        changed.append(target)
    return changed, skipped


def _build_changes_zip(files: List[Path], repo_root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            try:
                arc = str(path.relative_to(repo_root)).replace("\\", "/")
                data = path.read_text(encoding="utf-8", errors="ignore")
                zf.writestr(arc, data)
            except Exception:
                continue
    return buf.getvalue()

def _llm_generate_plan(report_text: str, files_context: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    system_msg = (
        "You are a senior Python engineer. Using the report and file context, "
        "propose a minimal fix plan. Only refer to the provided files. "
        "Return strict JSON: {\"plan\":[{\"id\":1,\"file\":\"path\",\"summary\":\"...\"}],\"notes\":\"...\"}."
    )
    user_msg = f"REPORT:\n{report_text}\n\nFILES:\n{files_context}\n"
    res = llm_chat([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ], cfg)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or "LLM error"}
    raw = res.get("text", "")
    data = _extract_json_block(raw)
    plan = _normalize_plan(data) if data else []
    if not plan:
        plan = _fallback_plan(raw)
    return {"ok": True, "plan": plan, "raw": raw}


def _llm_generate_changes(
    report_text: str,
    plan: List[Dict[str, str]],
    files_context: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    plan_json = json.dumps(plan, ensure_ascii=True, indent=2)
    system_msg = (
        "You are a senior Python engineer. Apply the plan using the provided files. "
        "Return strict JSON: {\"files\":[{\"path\":\"path\",\"content\":\"<full file text>\"}],\"notes\":\"...\"}. "
        "Only include files you change; keep everything else untouched."
    )
    user_msg = (
        f"PLAN:\n{plan_json}\n\nREPORT:\n{report_text}\n\nFILES:\n{files_context}\n"
    )
    res = llm_chat([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ], cfg)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or "LLM error"}
    raw = res.get("text", "")
    data = _extract_json_block(raw)
    files = []
    if isinstance(data, dict):
        files = data.get("files") or data.get("changes") or []
    if not isinstance(files, list):
        files = []
    return {"ok": True, "files": files, "raw": raw}


def _llm_generate_recommendations(report_text: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    system_msg = (
        "You are a software advisor. Based on the report, recommend 3-5 relevant projects. "
        "Return strict JSON: {\"projects\":[{\"name\":\"...\",\"url\":\"https://...\",\"why\":\"...\"}]}."
    )
    user_msg = f"REPORT:\n{report_text}\n"
    res = llm_chat([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ], cfg)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or "LLM error"}
    raw = res.get("text", "")
    data = _extract_json_block(raw)
    recs = _normalize_recommendations(data) if data else []
    if not recs:
        recs = _normalize_recommendations(_extract_json_block(raw) or {})
    return {"ok": True, "projects": recs, "raw": raw}

def main() -> None:
    st.set_page_config(page_title="代码助手", layout="wide")
    _inject_css()
    _hide_theme_picker()

    st.markdown(
        """
<div class="hero">
  <div class="hero-title">代码助手 </div>
  <div class="hero-subtitle">数据科学审查、测试生成和报告一站式完成</div>
</div>
""",
        unsafe_allow_html=True,
    )
    
    # 添加页面选择器
    nav_options = ["主工作区", "规则文档", "配置管理"]
    segmented = getattr(st, "segmented_control", None)
    if segmented:
        page = segmented("导航", nav_options, default=nav_options[0], label_visibility="collapsed")
    else:
        page = st.radio("导航", nav_options, horizontal=True, label_visibility="collapsed")

    # 规则文档页面
    if page == "规则文档":
        st.markdown("## 📚 可用规则文档")
        
        # DS 基础规则
        with st.expander("🎯 Data Science Basic Rules (11 个)", expanded=True):
            st.markdown("""
            #### DS_RANDOM_SEED
            **严重性**: High  
            **说明**: 检测随机数使用但未显式设置 seed，可能导致结果不可复现。  
            **示例**:
            ```python
            # bad
            import random
            x = random.random()

            # good
            import random
            random.seed(42)
            x = random.random()
            ```

            #### DS_SKLEARN_RANDOM_STATE
            **严重性**: High  
            **说明**: sklearn 的随机组件未设置 `random_state`。  
            **示例**:
            ```python
            # bad
            clf = RandomForestClassifier(n_estimators=100)

            # good
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            ```

            #### DS_TORCH_SEED
            **严重性**: Medium  
            **说明**: 使用 PyTorch 随机数但未设置 `torch.manual_seed`。  
            **示例**:
            ```python
            # bad
            import torch
            x = torch.rand(3)

            # good
            import torch
            torch.manual_seed(42)
            x = torch.rand(3)
            ```

            #### DS_TF_SEED
            **严重性**: Medium  
            **说明**: 使用 TensorFlow 随机数但未设置 `tf.random.set_seed`。  
            **示例**:
            ```python
            # bad
            import tensorflow as tf
            x = tf.random.uniform([3])

            # good
            import tensorflow as tf
            tf.random.set_seed(42)
            x = tf.random.uniform([3])
            ```

            #### DS_LEAKAGE_FIT_BEFORE_SPLIT
            **严重性**: High  
            **说明**: 在 `train_test_split` 之前调用 `fit_transform`，可能造成数据泄漏。  
            **示例**:
            ```python
            # bad
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

            # good
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            ```

            #### DS_PIPELINE_SUGGEST
            **严重性**: Medium  
            **说明**: 预处理与模型分离 fit/transform，建议使用 Pipeline。  
            **示例**:
            ```python
            # bad
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            clf.fit(X, y)

            # good
            pipe = make_pipeline(StandardScaler(), LogisticRegression())
            pipe.fit(X, y)
            ```

            #### DS_PANDAS_ITERROWS
            **严重性**: Medium  
            **说明**: `iterrows()` 逐行遍历性能低。  
            **示例**:
            ```python
            # bad
            for _, row in df.iterrows():
                total += row["value"]

            # good
            total = df["value"].sum()
            ```

            #### DS_PANDAS_APPLY_AXIS1
            **严重性**: Medium  
            **说明**: `apply(axis=1)` 行级 apply 性能较差。  
            **示例**:
            ```python
            # bad
            df["z"] = df.apply(lambda r: r["x"] + r["y"], axis=1)

            # good
            df["z"] = df["x"] + df["y"]
            ```

            #### DS_PANDAS_SETTINGWITHCOPY
            **严重性**: High  
            **说明**: `SettingWithCopy` 可能导致修改丢失或意外写入。  
            **示例**:
            ```python
            # bad
            df[df.a > 0]["b"] = 1

            # good
            df.loc[df.a > 0, "b"] = 1
            ```

            #### DS_MODEL_PICKLE_UNSAFE
            **严重性**: High  
            **说明**: 直接反序列化未验证的 pickle 可能带来安全风险。  
            **示例**:
            ```python
            # bad
            import pickle
            model = pickle.load(open("model.pkl", "rb"))

            # good
            # 仅从可信来源加载，或使用更安全的格式/签名校验
            ```

            #### DS_HYPERPARAMS_HARDCODED
            **严重性**: Low  
            **说明**: 超参数硬编码，降低可复现性与可调参性。  
            **示例**:
            ```python
            # bad
            clf = RandomForestClassifier(n_estimators=200, max_depth=8)

            # good
            params = {"n_estimators": 200, "max_depth": 8}
            clf = RandomForestClassifier(**params)
            ```
            """)
        
        # DS 高级规则
        with st.expander("🚀 Data Science Advanced Rules (7 个)", expanded=True):
            st.markdown("""
            #### DS_FEATURE_SELECTION_NO_NESTED_CV
            **严重性**: Medium  
            **说明**: 特征选择未嵌套在交叉验证中，可能引入泄漏。  
            **建议**: 使用 Pipeline + 交叉验证或嵌套 CV。

            #### DS_IMBALANCE_NOT_IN_PIPELINE
            **严重性**: High  
            **说明**: 处理不平衡（如 SMOTE）未放入 Pipeline，可能在 CV 中泄漏。  
            **建议**: 使用 `imblearn.pipeline.Pipeline` 组合采样与模型。

            #### DS_IMBALANCE_UNHANDLED
            **严重性**: Low  
            **说明**: 检测到类别不平衡但未进行处理。  
            **建议**: class_weight/重采样/阈值调节等。

            #### DS_CV_NOT_STRATIFIED
            **严重性**: Medium  
            **说明**: 分类任务使用非分层 CV（如 KFold/ShuffleSplit）。  
            **建议**: 使用 StratifiedKFold/StratifiedShuffleSplit。

            #### DS_NO_VALIDATION_SPLIT
            **严重性**: Low  
            **说明**: 未进行 train/valid 划分或未提供 validation_split/validation_data。  
            **建议**: 保留验证集用于调参与早停。

            #### DS_EVAL_ON_TRAIN
            **严重性**: Medium  
            **说明**: 在训练集上评估指标，结果易过拟合。  
            **建议**: 在验证集或测试集上评估。

            #### DS_EVALUATION_INCOMPLETE
            **严重性**: Medium  
            **说明**: 评价指标不完整，仅有单一指标。  
            **建议**: 补充 Precision/Recall/F1/AUC 等。
            """)
        
        # 插件规则
        with st.expander("🔌 Plugin Rules (4 个)", expanded=True):
            st.markdown("""
            #### PY_MUTABLE_DEFAULT_ARG
            **类别**: Data Science  
            **严重性**: Medium  
            **描述**: 可变默认参数会被所有函数调用共享  
            ```python
            # ❌ 错误
            def func(items=[]):
                items.append(1)
                return items
            
            # ✅ 正确
            def func(items=None):
                if items is None:
                    items = []
                items.append(1)
                return items
            ```
            
            #### PY_GLOBAL_VARIABLE
            **类别**: Data Science  
            **严重性**: Low  
            **描述**: 过度使用全局变量  
            
            #### PY_RESOURCE_LEAK
            **类别**: Security  
            **严重性**: High  
            **描述**: 文件未在 with 语句中使用  
            ```python
            # ❌ 错误
            f = open('file.txt')
            data = f.read()
            f.close()
            
            # ✅ 正确
            with open('file.txt') as f:
                data = f.read()
            ```
            
            #### PY_LOOP_INVARIANT
            **类别**: Performance  
            **严重性**: Low  
            **描述**: 循环内的不变表达式应提取到循环外  
            """)
        
        st.markdown("---")
        st.info("💡 提示: 这些规则可在侧边栏的 Advanced Settings 中启用/禁用")
        return
    
    # 配置页面
    if page == "配置管理":
        st.markdown("## ⚙️ 配置管理")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 当前配置")
            cfg_path = st.text_input("配置文件路径", value="config.yaml")
            if Path(cfg_path).exists():
                cfg = load_config(cfg_path, validate=False)
                
                # 验证配置
                try:
                    validated = validate_config(cfg)
                    st.success("✅ 配置验证通过")
                    
                    # 显示配置摘要
                    st.json({
                        "assistant": {
                            "max_files": cfg.get("assistant", {}).get("max_files"),
                            "include_globs": cfg.get("assistant", {}).get("include_globs"),
                        },
                        "review": {
                            "enable_ds_rules": cfg.get("review", {}).get("enable_ds_rules"),
                            "enable_ds_rules_advanced": cfg.get("review", {}).get("enable_ds_rules_advanced"),
                        }
                    })
                except ValueError as e:
                    st.error(f"❌ 配置验证失败:\n{e}")
            else:
                st.warning(f"配置文件不存在: {cfg_path}")
        
        with col2:
            st.markdown("### Pydantic 数据模型")
            st.code("""
# 配置数据模型
class AssistantConfig(BaseModel):
    max_files: int = Field(2000, ge=1)
    include_globs: List[str]
    exclude_globs: List[str]

class ReviewConfig(BaseModel):
    enable_ruff: bool = True
    enable_ds_rules: bool = True
    enable_ds_rules_advanced: bool = True
    # ...更多配置项

class CodeAssistantConfig(BaseModel):
    assistant: AssistantConfig
    review: ReviewConfig
    testgen: TestGenConfig
    coverage: CoverageConfig
            """, language="python")
        
        st.markdown("---")
        st.info("💡 配置使用 Pydantic 进行类型验证和约束检查")
        return

    with st.sidebar:
        st.markdown("### 📂 工作区")
        cfg_path = st.text_input("配置文件", value="config.yaml")
        repo_input = st.text_area("仓库路径或 GitHub 链接（每行一个）", value="my_repo", height=120)
        cache_dir = st.text_input("GitHub 缓存文件夹", value="Git_repo")
        st.caption("每行一个仓库。本地文件夹中有多个仓库时会自动展开。")
        out_dir = st.text_input("输出文件夹", value="reports")
        
        st.markdown("---")
        st.markdown("### 🔧 Advanced Settings")
        
        # 高级规则开关
        with st.expander("🎯 DS 规则（数据科学）", expanded=False):
            enable_ds_basic = st.checkbox("启用基础 DS 规则", value=True, 
                help="随机种子、数据泄漏、Pipeline 建议等")
            enable_ds_advanced = st.checkbox("启用高级 DS 规则", value=False,
                help="⚠️ 特征选择、不平衡处理、评估指标等 (较慢，建议仅用于小型仓库)")
            
            if enable_ds_advanced:
                st.warning("⚠️ 高级规则会显著增加扫描时间，特别是对于大型代码库。如果扫描卡顿，请禁用此选项。")
            
            if enable_ds_basic or enable_ds_advanced:
                st.caption("✓ 复现性检测\n✓ 数据泄漏检测\n✓ 模型序列化安全\n✓ 超参数硬编码")
        
        # 规则插件系统
        with st.expander("🔌 规则插件", expanded=False):
            # 动态获取注册的规则
            try:
                from src.features.review import builtin_rules
                registry = get_registry()
                all_rules = registry.get_all()
                categories = registry.get_categories()
                
                if all_rules:
                    st.caption(f"已加载 {len(all_rules)} 个插件规则")
                    for cat in sorted(categories):
                        cat_rules = registry.get_all(category=cat)
                        st.markdown(f"**{cat}** ({len(cat_rules)})")
                        for rule in cat_rules:
                            st.checkbox(
                                f"{rule.rule_id}",
                                value=True,
                                key=f"rule_{rule.rule_id}",
                                help=rule.description
                            )
                else:
                    st.info("未加载任何插件规则")
            except Exception as e:
                st.warning(f"规则插件系统不可用: {e}")
        
        # 其他工具开关
        with st.expander("🛠️ 工具", expanded=False):
            enable_ruff = st.checkbox("启用 Ruff", value=True, help="快速 Python Linter")
            enable_bandit = st.checkbox("启用 Bandit", value=True, help="安全漏洞扫描")
            enable_radon = st.checkbox("启用 Radon", value=True, help="代码复杂度分析")
            enable_mypy = st.checkbox("启用 MyPy", value=True, help="静态类型检查")
        
        # 日志配置
        with st.expander("📋 日志设置", expanded=False):
            log_level = st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
            use_color_logs = st.checkbox("彩色输出", value=True)
            log_to_file = st.checkbox("保存到文件", value=False)
            if log_to_file:
                log_file_path = st.text_input("日志文件", value="logs/app.log")
        
        st.markdown("---")
        st.markdown("### 💡 快速指南")
        st.caption("**工作流程：**")
        st.markdown("""
        1. 🔍 **代码审查** - 检测问题
        2. 🧪 **测试生成** - 创建单元测试  
        3. 📄 **报告生成** - 导出结果
        """)
        st.caption("💻 在主面板选择仓库并点击「全部运行」即可开始。")

    # 加载和验证配置
    cfg = load_config(cfg_path, validate=False)
    
    # 显示配置验证状态
    try:
        validated_cfg = validate_config(cfg)
        st.sidebar.success("✅ 配置验证成功")
    except ValueError as e:
        st.sidebar.error(f"❌ 配置验证失败: {str(e)[:100]}")
        st.stop()
    llm_cfg = cfg.get("llm", {})
    with st.sidebar:
        st.markdown("---")
        st.markdown("### LLM Settings")
        llm_enabled = st.checkbox("Enable LLM actions", value=bool(llm_cfg.get("enabled", True)))
        llm_model = st.text_input("Model", value=str(llm_cfg.get("model", "gpt-4o-mini")))
        llm_base_url = st.text_input(
            "Base URL",
            value=str(llm_cfg.get("base_url", "https://api.openai.com/v1")),
        )
        llm_api_key_env = st.text_input(
            "API key env",
            value=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
        )
        llm_api_key = st.text_input("API key (optional)", type="password", value="")
        llm_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(llm_cfg.get("temperature", 0.2)),
            step=0.1,
        )
        llm_max_tokens = st.number_input(
            "Max tokens",
            min_value=256,
            max_value=8192,
            value=int(llm_cfg.get("max_tokens", 1200)),
            step=128,
        )

    cfg.setdefault("llm", {})
    cfg["llm"]["enabled"] = llm_enabled
    cfg["llm"]["model"] = llm_model
    cfg["llm"]["base_url"] = llm_base_url
    cfg["llm"]["api_key_env"] = llm_api_key_env
    cfg["llm"]["temperature"] = llm_temperature
    cfg["llm"]["max_tokens"] = llm_max_tokens
    if llm_api_key:
        cfg["llm"]["api_key"] = llm_api_key
    # 应用 GUI 设置到配置
    cfg.setdefault("review", {})
    cfg["review"]["enable_ds_rules"] = enable_ds_basic
    cfg["review"]["enable_ds_rules_advanced"] = enable_ds_advanced
    cfg["review"]["enable_ruff"] = enable_ruff
    cfg["review"]["enable_bandit"] = enable_bandit
    cfg["review"]["enable_radon"] = enable_radon
    cfg["review"]["enable_mypy"] = enable_mypy
    
    # 配置日志系统
    if 'logger_configured' not in st.session_state:
        import logging
        level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, 
                     "WARNING": logging.WARNING, "ERROR": logging.ERROR}
        setup_logger(
            "code-assistant",
            level=level_map[log_level],
            use_color=use_color_logs,
            log_file=log_file_path if log_to_file else None
        )
        st.session_state['logger_configured'] = True
    
    st.markdown("---")
    st.markdown("### 🚀 执行操作")
    col_a, col_b, col_c = st.columns(3, gap="medium")
    
    with col_a:
        st.markdown("**🔍 代码审查**")
        st.caption("静态分析、安全检查和数据科学规则")
        run_review = st.button("开始审查", use_container_width=True)
    with col_b:
        st.markdown("**🧪 测试生成**")
        st.caption("生成单元测试和覆盖率报告")
        run_testgen = st.button("生成测试", use_container_width=True)
    with col_c:
        st.markdown("**⚡ 完整流程**")
        st.caption("审查 + 测试 + 报告一步完成")
        run_all = st.button("全部运行", type="primary", use_container_width=True, help="一键执行审查、测试生成和报告生成")

    st.divider()

    state = st.session_state
    state.setdefault("last_review", None)
    state.setdefault("last_testgen", None)
    state.setdefault("last_report_path", None)
    state.setdefault("last_report_pdf", None)
    state.setdefault("batch_results", [])
    state.setdefault("last_repo_path", None)
    state.setdefault("llm_plan", None)
    state.setdefault("llm_plan_source", None)
    state.setdefault("llm_changes", None)
    state.setdefault("llm_recommendations", None)

    repo_jobs: List[Dict[str, str]] = []
    if run_review or run_testgen or run_all:
        repo_inputs = _parse_repo_inputs(repo_input)
        repo_jobs = _resolve_repo_inputs(repo_input, cache_dir)
        if not repo_jobs:
            st.stop()
        if len(repo_jobs) > 1:
            st.info(f"批量模式：{len(repo_jobs)} 个仓库。")
        elif repo_inputs and repo_jobs[0]["path"] != repo_inputs[0]:
            st.caption(f"已解析仓库：{repo_jobs[0]['path']}")

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
                st.info(f"📊 正在扫描 {job['name']} 的问题...（这可能需要一些时间）")
                with st.spinner(f"正在执行审查：{job['name']}"):
                    review = orch.run_review(repo_path=job["path"])
                (repo_out / "review.json").write_text(
                    json.dumps(review, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.success(f"✅ [{job['name']}] 审查完成：发现 {len(review['findings'])} 个问题。")

            testgen = None
            if run_testgen or run_all:
                st.info(f"🧪 为 {job['name']} 生成测试...")
                with st.spinner(f"正在生成测试：{job['name']}"):
                    testgen = orch.run_testgen(repo_path=job["path"])
                (repo_out / "testgen.json").write_text(
                    json.dumps(testgen, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                st.success(
                    f"✅ [{job['name']}] 测试生成完成：{testgen['written_files']} 个文件已写入"
                    f"{testgen['output_dir']}。"
                )
                st.success(
                    f"[{job['name']}] Test generation complete: {testgen['written_files']} files written to "
                    f"{testgen['output_dir']}."
                )

            report_path = None
            pdf_path = None
            if run_all:
                st.info(f"📄 为 {job['name']} 生成报告...")
                with st.spinner(f"正在生成报告：{job['name']}"):
                    report_path, pdf_path = _write_report(str(repo_out), review, testgen)
                st.success(f"✅ [{job['name']}] 报告已保存：{report_path}")

            if not batch_mode:
                state['last_repo_path'] = job['path']
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
        st.markdown("---")
        st.markdown("### 📦 批量处理结果")
        st.caption("多个仓库的分析结果汇总")
        
        # 使用表格形式展示结果
        for item in batch_results:
            review_part = (
                f"✓ {item['review_count']} 个问题" if item.get("review_count") is not None else "⊘ 未审查"
            )
            test_part = (
                f"✓ {item['testgen_written']} 个测试" if item.get("testgen_written") is not None else "⊘ 未生成"
            )
            report_part = "✓ 报告已生成" if item.get("report_path") else "⊘ 无报告"
            
            st.markdown(f"**📁 {item['name']}** · {review_part} · {test_part} · {report_part}")
        
        st.markdown("---")
        st.markdown("#### 🔍 选择仓库查看详情")
        labels = [item["name"] for item in batch_results]
        active_name = st.selectbox(
            "选择要检查的仓库",
            labels,
            key="batch_select",
            help="选择一个仓库以查看其详细的审查结果和测试报告"
        )
        active_item = next((i for i in batch_results if i["name"] == active_name), None)
        if active_item:
            active_out_dir = active_item["out_dir"]
            active_review, active_testgen = _load_report_sources(active_out_dir, {})

    review = active_review
    if review:
        title = "### 📋 审查报告"
        if batch_mode and active_item:
            title = f"### 📋 审查报告（{active_item['name']}）"
        findings = review["findings"]
        st.markdown(title)
        
        if len(findings) == 0:
            st.success("🎉 未发现问题！代码质量优秀（或扫描器已禁用）。")
        else:
            # 分类统计
            ds_rules = [f for f in findings if f.get("tool") in ["ds-rule", "ds-rule-advanced"]]
            plugin_rules = [f for f in findings if f.get("tool") == "rule-plugin"]
            other_findings = [f for f in findings if f.get("tool") not in ["ds-rule", "ds-rule-advanced", "rule-plugin"]]
            
            # 主要指标 - 使用卡片式布局
            st.markdown("#### 📊 问题概览")
            metric_cols = st.columns(4, gap="medium")
            metric_cols[0].metric("🔍 发现问题总数", len(findings))
            metric_cols[1].metric("🎯 DS 规则", len(ds_rules))
            metric_cols[2].metric("🔌 插件规则", len(plugin_rules))
            metric_cols[3].metric("🛠️ 其他工具", len(other_findings))
            
            st.markdown("")  # 添加间距

            # 展开新规则的详细信息
            if ds_rules:
                with st.expander("🎯 数据科学规则详情", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**规则分布**")
                        ds_rule_names = [f.get("rule") for f in ds_rules]
                        _plot_counts(ds_rule_names, "DS 规则类型")
                    
                    with col2:
                        st.markdown("**严重性分布**")
                        ds_severities = [f.get("severity") for f in ds_rules]
                        _plot_counts(ds_severities, "DS 严重性")
                    
                    # 显示关键 DS 问题
                    st.markdown("**关键问题**")
                    ds_high = [f for f in ds_rules if f.get("severity") == "high"]
                    if ds_high:
                        for f in ds_high[:5]:
                            st.warning(f"⚠️ {f.get('rule')}：{f.get('message')}（第 {f.get('line', 'N/A')} 行）")
                    else:
                        st.success("✅ 未发现高危 DS 问题")
            
            if plugin_rules:
                with st.expander("🔌 插件规则详情", expanded=True):
                    plugin_rule_names = [f.get("rule") for f in plugin_rules]
                    _plot_counts(plugin_rule_names, "插件规则类型")
                    
                    # 按类别分组
                    plugin_by_rule = {}
                    for f in plugin_rules:
                        rule = f.get("rule", "unknown")
                        plugin_by_rule.setdefault(rule, []).append(f)
                    
                    for rule, items in plugin_by_rule.items():
                        st.markdown(f"**{rule}** ({len(items)} 个)")
                        for item in items[:3]:
                            st.caption(f"- {item.get('file')}:{item.get('line')} - {item.get('message')}")

            # 总览图表 - 使用分隔线和更好的布局
            st.markdown("---")
            st.markdown("#### 📊 数据可视化")
            chart_cols = st.columns(3, gap="large")
            
            with chart_cols[0]:
                st.markdown("**🎨 严重性分布**")
                sev = [f["severity"] for f in findings]
                _plot_counts(sev, "")
            
            with chart_cols[1]:
                st.markdown("**🔧 工具分布**")
                tools = [f["tool"] for f in findings]
                _plot_counts(tools, "")
            
            with chart_cols[2]:
                st.markdown("**📁 问题文件 Top 10**")
                # 文件分布
                files = [f.get("file", "unknown") for f in findings]
                file_counts = Counter(files).most_common(10)
                _plot_counts(file_counts, "")

            # 详细问题列表
            st.markdown("---")
            st.markdown("#### 🔍 详细问题列表（Top 20）")
            st.caption("按严重性和工具排序的前 20 个问题")
            _show_findings_table(findings[:20])

    # 测试生成结果
    testgen = active_testgen
    if testgen:
        st.markdown("---")
        title = "### 🧪 测试生成结果"
        if batch_mode and active_item:
            title = f"### 🧪 测试生成结果（{active_item['name']}）"
        st.markdown(title)
        
        # 显示关键指标
        col1, col2, col3 = st.columns(3)
        col1.metric("📝 生成文件数", testgen.get("written_files", 0))
        col2.metric("📂 输出目录", testgen.get("output_dir", "N/A"))
        col3.metric("✅ 状态", "完成" if testgen.get("written_files", 0) > 0 else "无")
        out_dir_val = str(testgen.get("output_dir", "N/A"))
        copy_key = f"testgen_outdir_copy_{active_item['name']}" if batch_mode and active_item else "testgen_outdir_copy"
        st.text_input("Output dir (copyable)", value=out_dir_val, key=copy_key)
        
        # 详细信息（折叠）
        with st.expander("📋 查看详细信息", expanded=False):
            st.json({k: v for k, v in testgen.items() if k != "generated"})

    # 报告下载和预览
    st.markdown("---")
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

        st.markdown("---")
        st.markdown("### LLM Actions")
        llm_cfg = cfg.get("llm", {})
        if not llm_cfg.get("enabled", True):
            st.info("LLM actions are disabled. Enable them in the LLM settings.")
        else:
            ready, reason = _llm_ready(cfg)
            if not ready:
                st.warning(f"LLM not configured: {reason}")
            else:
                repo_root = None
                if review_src and review_src.get("repo"):
                    repo_root = Path(review_src.get("repo"))
                elif testgen_src and testgen_src.get("repo"):
                    repo_root = Path(testgen_src.get("repo"))
                elif batch_mode and active_item:
                    repo_root = Path(active_item.get("path", ""))
                elif state.get("last_repo_path"):
                    repo_root = Path(state.get("last_repo_path"))

                if not repo_root or not repo_root.exists():
                    st.warning("Repo path not found for LLM actions.")
                else:
                    repo_key = str(repo_root)
                    meta = state.get("llm_plan_source") or {}
                    if isinstance(meta, dict) and meta.get("repo") != repo_key:
                        state["llm_plan"] = None
                        state["llm_changes"] = None

                    report_text = _truncate_text(build_markdown_report(review_src, testgen_src), 12000)
                    files_ctx = _collect_context_files(repo_root, review_src)
                    ctx_text = _format_file_context(files_ctx) if files_ctx else ""
                    if not ctx_text:
                        st.caption("No file context extracted from the report; LLM output may be limited.")

                    plan_col, rec_col = st.columns(2, gap="medium")
                    with plan_col:
                        if st.button("Generate fix plan", key="llm_plan_btn"):
                            plan_res = _llm_generate_plan(report_text, ctx_text, cfg)
                            if plan_res.get("ok"):
                                state["llm_plan"] = plan_res.get("plan")
                                state["llm_plan_source"] = {"repo": repo_key, "raw": plan_res.get("raw", "")}
                                state["llm_changes"] = None
                                st.success("Plan generated.")
                            else:
                                st.error(plan_res.get("error") or "LLM plan failed.")
                    with rec_col:
                        if st.button("Get recommendations", key="llm_rec_btn"):
                            rec_res = _llm_generate_recommendations(report_text, cfg)
                            if rec_res.get("ok"):
                                state["llm_recommendations"] = rec_res.get("projects")
                                st.success("Recommendations ready.")
                            else:
                                st.error(rec_res.get("error") or "LLM recommendations failed.")

                    plan = state.get("llm_plan") or []
                    if plan:
                        st.markdown("#### Proposed changes")
                        for item in plan:
                            file_part = f" ({item.get('file')})" if item.get("file") else ""
                            st.markdown(f"- {item.get('summary')}{file_part}")

                        if st.button("Apply changes", key="llm_apply_btn"):
                            apply_res = _llm_generate_changes(report_text, plan, ctx_text, cfg)
                            if apply_res.get("ok"):
                                changed, skipped = _apply_llm_changes(
                                    repo_root,
                                    apply_res.get("files") or [],
                                    allow_new=bool(llm_cfg.get("allow_new_files", False)),
                                )
                                state["llm_changes"] = {
                                    "changed": [str(p) for p in changed],
                                    "skipped": skipped,
                                }
                                if changed:
                                    zip_bytes = _build_changes_zip(changed, repo_root)
                                    st.download_button(
                                        "Download modified files",
                                        data=zip_bytes,
                                        file_name="llm_changes.zip",
                                    )
                                    st.success(f"Applied changes to {len(changed)} files.")
                                if skipped:
                                    st.warning(f"Skipped {len(skipped)} file entries.")
                            else:
                                st.error(apply_res.get("error") or "LLM apply failed.")

                    recs = state.get("llm_recommendations") or []
                    if recs:
                        st.markdown("#### Recommendations")
                        for rec in recs:
                            name = rec.get("name") or "(no name)"
                            url = rec.get("url") or ""
                            why = rec.get("why") or ""
                            if url:
                                st.markdown(f"- [{name}]({url}) - {why}")
                            else:
                                st.markdown(f"- {name} - {why}")


if __name__ == "__main__":
    main()


























