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
from src.core.config_validator import validate_config, CodeAssistantConfig
from src.core.orchestrator import Orchestrator
from src.core.subproc import run_cmd
from src.core.logger import setup_logger
from src.features.review.rule_plugin import get_registry
from src.reporting.report_builder import build_markdown_report
from src.reporting.pdf_builder import build_pdf_report
from src.reporting.latex_builder import build_latex_report


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
  background: linear-gradient(rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.3)), 
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
  --border: #e2e8f0;
  --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}}

.stApp {{
  background:
    radial-gradient(1000px 500px at 10% 0%, rgba(16, 163, 127, 0.2) 0%, transparent 50%),
    radial-gradient(800px 400px at 90% 0%, rgba(52, 211, 153, 0.2) 0%, transparent 45%),
    radial-gradient(600px 300px at 50% 100%, rgba(26, 188, 156, 0.2) 0%, transparent 40%),
    #ffffff;
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
  background:
    radial-gradient(400px 400px at 0% 70%, rgba(255, 220, 120, 0.25) 0%, transparent 60%),
    radial-gradient(300px 300px at 0% 0%, rgba(255, 235, 59, 0.20) 0%, transparent 50%),
    #fafafa;
  border-right: 1px solid var(--border);
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {{
  color: var(--ink);
}}

section[data-testid="stSidebar"] .stTextInput input {{
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.55rem 0.75rem;
}}

.stButton button {{
  width: 100%;
  border-radius: 999px;
  border: 1px solid transparent;
  background: linear-gradient(135deg, var(--accent), #22c28f);
  color: #ffffff;
  font-weight: 600;
  padding: 0.6rem 1.2rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}}

.stButton button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(16, 163, 127, 0.2);
  background: linear-gradient(135deg, var(--accent-strong), #1aa37a);
}}

.stDownloadButton button {{
  width: 100%;
  border-radius: 999px;
  border: 1px solid transparent;
  background: linear-gradient(135deg, var(--accent), #22c28f) !important;
  color: #ffffff !important;
  font-weight: 600;
  padding: 0.6rem 1.2rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}}

.stDownloadButton button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(16, 163, 127, 0.2);
  background: linear-gradient(135deg, var(--accent-strong), #1aa37a) !important;
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
  text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8);
}}

.hero-subtitle {{
  margin-top: 0.35rem;
  color: var(--muted);
  font-size: 1rem;
  position: relative;
  z-index: 1;
  text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
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
}}

@keyframes rise {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
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


def main() -> None:
    st.set_page_config(page_title="代码助手", layout="wide")
    _inject_css()

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
    page = st.radio(
        "导航",
        ["🏠 主工作区", "📚 规则文档", "⚙️ 配置管理"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # 规则文档页面
    if page == "📚 规则文档":
        st.markdown("## 📚 可用规则文档")
        
        # DS 基础规则
        with st.expander("🎯 Data Science Basic Rules (7 个)", expanded=True):
            st.markdown("""
            #### DS_RANDOM_SEED
            **严重性**: Medium  
            **描述**: 检测使用随机性但未设置种子  
            **示例**:
            ```python
            # ❌ 错误
            import random
            x = random.random()
            
            # ✅ 正确
            import random
            random.seed(42)
            x = random.random()
            ```
            
            #### DS_SKLEARN_RANDOM_STATE
            **严重性**: Medium  
            **描述**: sklearn 随机组件缺少 random_state 参数  
            **示例**:
            ```python
            # ❌ 错误
            clf = RandomForestClassifier(n_estimators=100)
            
            # ✅ 正确
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            ```
            
            #### DS_LEAKAGE_FIT_BEFORE_SPLIT
            **严重性**: High  
            **描述**: fit_transform 在 train_test_split 之前可能导致数据泄漏  
            **示例**:
            ```python
            # ❌ 错误
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test = train_test_split(X_scaled)
            
            # ✅ 正确
            X_train, X_test = train_test_split(X)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            ```
            
            #### DS_PIPELINE_SUGGEST
            **严重性**: Medium  
            **描述**: 缩放器未在 Pipeline 中使用  
            
            #### DS_MODEL_PICKLE_UNSAFE
            **严重性**: High  
            **描述**: 使用 pickle 序列化模型不安全  
            **建议**: 使用 joblib.dump() 或 ONNX 导出
            
            #### DS_HYPERPARAMS_HARDCODED
            **严重性**: Low  
            **描述**: 模型超参数硬编码  
            **建议**: 使用 GridSearchCV 或配置文件
            
            #### DS_PANDAS_ITERROWS / DS_PANDAS_APPLY_AXIS1
            **严重性**: Low  
            **描述**: pandas 低效操作  
            **建议**: 使用向量化操作
            """)
        
        # DS 高级规则
        with st.expander("🚀 Data Science Advanced Rules (5 个)", expanded=True):
            st.markdown("""
            #### DS_FEATURE_SELECTION_NO_NESTED_CV
            **严重性**: Medium  
            **描述**: 特征选择后未使用嵌套交叉验证可能导致过拟合  
            
            #### DS_IMBALANCE_NOT_IN_PIPELINE
            **严重性**: High  
            **描述**: 采样方法（SMOTE）未在 Pipeline 中可能导致数据泄漏  
            
            #### DS_IMBALANCE_UNHANDLED
            **严重性**: Low  
            **描述**: 模型训练未处理数据不平衡  
            **建议**: 使用 class_weight、SMOTE 或分层 CV
            
            #### DS_EVALUATION_INCOMPLETE
            **严重性**: Low  
            **描述**: 评估指标不足  
            **建议**: 使用多个指标（accuracy, precision, recall, F1, ROC-AUC）
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
    if page == "⚙️ 配置管理":
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
            enable_mypy = st.checkbox("启用 MyPy", value=False, help="静态类型检查")
        
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
        
        # 详细信息（折叠）
        with st.expander("📋 查看详细信息", expanded=False):
            st.json({k: v for k, v in testgen.items() if k != "generated"})

    # 报告下载和预览
    st.markdown("---")
    report_title = "### 📄 综合报告"
    if batch_mode and active_item:
        report_title = f"### 📄 综合报告（{active_item['name']}）"
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
            st.info("📝 报告尚未生成，点击下方按钮创建综合报告。")
            button_label = "🔨 为选定的仓库生成报告" if batch_mode else "🔨 生成综合报告"
            
            # 居中的按钮
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(button_label, key="build_report", use_container_width=True):
                    with st.spinner("正在生成报告..."):
                        report_path, pdf_path = _write_report(active_out_dir, active_review, active_testgen)
                        if batch_mode and active_item:
                            active_item["report_path"] = str(report_path)
                            active_item["pdf_path"] = str(pdf_path) if pdf_path else None
                        else:
                            state["last_report_path"] = str(report_path)
                            state["last_report_pdf"] = str(pdf_path) if pdf_path else None
                        rp = report_path
                    st.success("✅ 报告生成成功！")
                    st.rerun()
        else:
            st.warning("⚠️ 请先运行审查或测试生成以获取数据。")
            st.caption("💡 提示：点击上方的「开始审查」、「生成测试」或「全部运行」按钮。")

    if rp and rp.exists():
        source_state = {} if batch_mode else state
        review_src, testgen_src = _load_report_sources(active_out_dir, source_state)
        
        # 始终从最新的 .md 报告文件生成 PDF（确保预览最新内容）
        md_text = rp.read_text(encoding="utf-8")
        pdf_bytes = None
        pdf_path = None
        
        # 尝试从 LaTeX 编译生成 PDF（高质量）
        tex_path = rp.with_suffix(".tex")
        if not tex_path.exists():
            tex_path.write_text(build_latex_report(review_src, testgen_src), encoding="utf-8")
        compiled = _compile_latex(tex_path)
        if compiled:
            pdf_path = compiled
            pdf_bytes = compiled.read_bytes()
        else:
            # 降级到 reportlab 生成 PDF
            pdf_bytes = _make_pdf_bytes(review_src, testgen_src, md_text)
            if pdf_bytes:
                pdf_path = rp.with_suffix(".pdf")
                pdf_path.write_bytes(pdf_bytes)
        
        # 显示下载按钮和 PDF 预览
        if pdf_bytes:
            file_prefix = f"{active_item['name']}_" if batch_mode and active_item else ""
            
            # 下载按钮区域 - 美化设计
            st.markdown("---")
            st.markdown("### 📥 报告下载")
            st.caption("💾 可下载 PDF 或 Markdown 格式的完整报告")
            
            col1, col2, col3 = st.columns([1, 1, 2], gap="medium")
            with col1:
                st.download_button(
                    "📄 下载 PDF",
                    data=pdf_bytes,
                    file_name=f"{file_prefix}report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📝 下载 Markdown",
                    data=rp.read_bytes(),
                    file_name=f"{file_prefix}report.md",
                    use_container_width=True
                )
            with col3:
                st.info("💡 提示：PDF 格式更适合打印和分享")
            
            # PDF 预览区域 - 美化设计
            st.markdown("---")
            st.markdown("### 👁️ PDF 预览")
            st.caption("📖 最新生成的报告内容（实时更新）")
            
            # 添加预览容器
            with st.container():
                _render_pdf_preview(pdf_bytes)
        else:
            st.warning("⚠️ PDF 预览不可用")
            st.caption("请确保已安装 reportlab 或 xelatex/tectonic 以生成 PDF 报告。")


if __name__ == "__main__":
    main()














