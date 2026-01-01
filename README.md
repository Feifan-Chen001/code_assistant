# CodeAssistant 智能代码助手（数据科学专项）

一个面向数据科学/机器学习项目的“代码审查 + 自动测例生成 + 报告输出”一体化工具。支持 Python 仓库与 Notebook 扫描，内置数据科学专项规则（复现性/泄漏/Pipeline/性能坑），提供可视化看板与批量实验 CLI。

## 目录
- [项目定位与目标](#项目定位与目标)
- [系统流程](#系统流程)
- [功能清单](#功能清单)
- [安装与环境准备](#安装与环境准备)
- [快速开始（GUI）](#快速开始gui)
- [快速开始（CLI）](#快速开始cli)
- [编程接口（Python 调用）](#编程接口python-调用)
- [配置说明（config.yaml）](#配置说明configyaml)
- [数据科学专项规则（DS Rules）](#数据科学专项规则ds-rules)
- [Notebook 支持](#notebook-支持)
- [覆盖率评估](#覆盖率评估)
- [报告与产物](#报告与产物)
- [批处理（多仓库实验）](#批处理多仓库实验)
- [可视化界面说明](#可视化界面说明)
- [目录结构与函数索引（逐文件/逐函数）](#目录结构与函数索引逐文件逐函数)
- [常见问题](#常见问题)
- [扩展与二次开发](#扩展与二次开发)

---

## 项目定位与目标
- **面向数据科学/ML 仓库的“静态审查 + 测例生成 + 报告输出”**：聚焦复现性、数据泄漏、Pipeline 使用建议、pandas 性能坑位。
- **支持 Notebook 扫描**：从 `.ipynb` 中抽取 code cell 作为“虚拟文件”扫描并可定位到 cell。
- **一键全流程与批量实验**：GUI 一键跑全流程，CLI 支持多仓库批处理，便于实验统计。

---

## 系统流程
```
Repo Path / GitHub URL
        │
        ▼
  文件扫描（include/exclude + max_files）
        │
        ├─ Review Pipeline
        │   ├─ AST 规则（禁用 eval/exec、裸 except）
        │   ├─ DS 规则（复现性/泄漏/Pipeline/性能）
        │   ├─ ruff / bandit / pip-audit
        │   └─ radon / mypy（可选）
        │
        ├─ TestGen Pipeline
        │   ├─ AST 提取公开函数
        │   ├─ 生成 pytest 模板
        │   └─ coverage report -m（可选）
        │
        ▼
  报告生成（report.md / report.tex / report.pdf）
```

---

## 功能清单
### Review（代码审查）
- AST 规则：`eval/exec` 禁用、裸 `except` 等
- Lint/安全/依赖：`ruff`、`bandit`、`pip-audit`、`radon`（可选 `mypy`）
- **DS Rules**：详见 [数据科学专项规则（DS Rules）](#数据科学专项规则ds-rules)
- Notebook 支持：`.ipynb` code cell 解析扫描

### TestGen（测试生成）
- 基于 AST 提取公开函数并生成 pytest 模板
- 可选 Hypothesis 模板
- 自动执行 `coverage report -m` 写入报告（可关闭）

### 报告与可视化
- `report.md`：结构化总结、Top 问题列表、覆盖率/复杂度摘要
- `report.tex`：LaTeX 格式
- `report.pdf`：可预览/下载的 PDF 报告
- `review.json` / `testgen.json`：机器可读输出

---

## 安装与环境准备
推荐 Python 3.9+。

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

依赖工具包含：
- 代码审计：`ruff`、`bandit`、`pip-audit`、`radon`（可选 `mypy`）
- 测试/覆盖率：`pytest`、`coverage`、`hypothesis`
- UI：`streamlit`
- PDF 导出：`reportlab`（fallback），`tectonic`（LaTeX 引擎）

可选但强烈建议：
- **XeLaTeX**（TeX Live / MiKTeX）以获得最佳排版
- **Git**：当 UI 输入 GitHub 链接时用于 clone

---

## 快速开始（GUI）
```bash
streamlit run app.py
```

界面里选择：
- 仓库路径或 GitHub 链接（支持多行，每行一个仓库）
- GitHub 缓存目录（默认 `Git_repo`，自动 clone 到该目录）
- 输出目录（默认 `reports`）
- 执行 Review / TestGen / All

批量说明：
- 多行输入会进入批量模式，按顺序处理每个仓库。
- 如果输入的是本地目录且内部含多个仓库（含 `.git` / `pyproject.toml` / `setup.py` / `setup.cfg` / `requirements.txt`），会自动展开为多个仓库。
- 批量输出默认写入 `reports/<repo_name>/`。

> 若使用 GitHub 链接，请确保本机已安装 `git` 且可联网访问 GitHub。

---

## 快速开始（CLI）
```bash
python -m src.cli review  --repo /path/to/repo --out reports
python -m src.cli testgen --repo /path/to/repo --out generated_tests
python -m src.cli all     --repo /path/to/repo --out reports
python -m src.cli batch   --repos repos.txt --mode all --out reports_batch
```

CLI 默认输出：
- `review.json` / `testgen.json` / `report.md`

如需生成 `report.tex`/`report.pdf`，请使用 GUI 或在代码中调用 `build_latex_report()` 并用 LaTeX 编译。

---

## 编程接口（Python 调用）
如果你希望在其他脚本中调用：

```python
from src.core.config import load_config
from src.core.orchestrator import Orchestrator

cfg = load_config("config.yaml")
orch = Orchestrator(cfg)

review = orch.run_review("D:/code/repo")
testgen = orch.run_testgen("D:/code/repo")
```

生成报告：
```python
from src.reporting.report_builder import build_markdown_report
from src.reporting.latex_builder import build_latex_report

md = build_markdown_report(review, testgen)
tex = build_latex_report(review, testgen)
```

---

## 配置说明（config.yaml）
`config.yaml` 是所有模块的统一入口。关键字段：

```yaml
assistant:
  max_files: 2000
  include_globs:
    - "**/*.py"
    - "**/*.ipynb"
  exclude_globs:
    - "**/.venv/**"
    - "**/venv/**"
    - "**/__pycache__/**"
    - "**/build/**"
    - "**/dist/**"
    - "**/.git/**"

review:
  enable_ruff: true
  enable_mypy: false
  enable_bandit: true
  enable_pip_audit: true
  enable_radon: true
  enable_ds_rules: true
  enable_notebook: true
  ruff_args: ["check", "--format", "json"]
  mypy_args: ["--show-error-codes", "--no-error-summary"]
  bandit_args: ["-r", "-f", "json"]
  pip_audit_args: ["-f", "json"]

testgen:
  output_dir: "generated_tests"
  use_hypothesis: true
  max_functions: 200

coverage:
  enable: true
  pytest_args: ["-q"]
```

说明：
- `include_globs` 已支持 `.ipynb`，用于扫描 Notebook。
- 若不需要 DS 规则或 Notebook 扫描，可关闭 `enable_ds_rules` / `enable_notebook`。
- `max_files`/`max_functions` 用于控制扫描规模与生成测试数量。

---

## 数据科学专项规则（DS Rules）
规则基于 **AST + 启发式**，目标是抓出可复现性、数据泄漏与性能坑位。

### 1) 复现性
- **随机性无显式 seed**：检测 `random` / `numpy.random` 使用但没有 `seed` 或随机对象种子。
- **sklearn 随机组件缺 `random_state`**：例如 `train_test_split`、`KFold`、`RandomForest`、`KMeans` 等。

### 2) 数据泄漏（启发式）
- **在 `train_test_split` 之前出现 `fit_transform`**：可能导致数据泄漏（先整体拟合再分割）。

### 3) Pipeline 建议
- **`StandardScaler` / `OneHotEncoder` 单独 `fit/transform`** 但未使用 `Pipeline`：提示使用 `sklearn.pipeline.Pipeline` 或 `make_pipeline`。

### 4) pandas 性能与坑位
- `iterrows` 性能差，建议向量化
- `apply(axis=1)` 性能差，建议向量化
- `SettingWithCopy` 风险：链式索引赋值（如 `df[df.a > 0]["b"] = ...`）

---

## Notebook 支持
- 自动读取 `.ipynb` 的 code cell
- 每个 code cell 作为“虚拟文件”扫描并标注为：
  - `notebook.ipynb#cell-1`
  - `notebook.ipynb#cell-2`
- 行号为 **cell 内部行号**，便于定位

---

## 覆盖率评估
TestGen 阶段会自动执行：
- `coverage run -m pytest`
- `coverage report -m`

覆盖率摘要会写入 `report.md` / `report.tex` / `report.pdf`。

---

## 报告与产物
### Review
- `review.json`：全部问题与原始信息（包含工具原始输出片段）

### TestGen
- `testgen.json`：生成测试的摘要与索引
- `generated_tests/`：生成的测试文件

### Report
- `report.md`：最终 Markdown 报告，包含统计汇总/Top 20/覆盖率等
- `report.tex`：LaTeX 版本（更接近 `main.tex` 排版）
- `report.pdf`：GUI 可预览/下载的 PDF
  - 优先使用 `xelatex` 编译
  - `xelatex` 不可用时使用 `tectonic`
  - 若均不可用，则回退为 `reportlab` 简版 PDF

---

## 批处理（多仓库实验）
适用于批量实验或论文统计。

### 1) 准备仓库列表
创建 `repos.txt`：
```text
# one repo path per line
D:/code/repo1
D:/code/repo2
```

### 2) 执行批量
```bash
python -m src.cli batch --mode all --repos repos.txt --out reports_batch
```

输出结构示例：
```
reports_batch/
  repo1/
    review.json
    testgen.json
    report.md
  repo2/
    review.json
    testgen.json
    report.md
```

---

## 可视化界面说明
GUI 采用 ChatGPT 风格布局：左侧工作区 + 右侧执行与结果看板。

- **Actions**：一键运行 Review / TestGen / All
- **Review Board**：
  - 总问题数
  - DS 规则问题数
  - 工具来源数
  - 严重性/工具/DS 分布图
- **Test Generation**：展示测试统计与覆盖率摘要
- **Report**：预览与下载 `report.pdf`，保留 `report.md` 作为原始文本
- **Batch results**：多仓库运行时显示汇总，并可切换查看某个仓库的 Review/TestGen/Report

---

## 目录结构与函数索引（逐文件/逐函数）
> 说明：此处只覆盖 **本项目源码与入口文件**。`my_repo/`、`Git_repo/` 为外部样例仓库缓存；`reports/`、`generated_tests/` 为运行时产物，不在函数索引范围。

### 根目录（入口与配置）
#### `app.py`（Streamlit GUI 入口）
- `main() -> None`：应用入口；使用方式：`streamlit run app.py`
- `_inject_css() -> None`：注入 ChatGPT 风格 CSS；由 `main()` 自动调用
- `_ensure_dirs(out_dir: str) -> None`：确保输出目录存在；在运行 Review/TestGen/报告前调用
- `_parse_repo_inputs(text: str) -> List[str]`：解析多行仓库输入（忽略空行和注释行）
- `_is_repo_root(path: Path) -> bool`：判断目录是否像一个仓库（.git/pyproject/setup/requirements 等标记）
- `_expand_local_repos(path: Path) -> List[Path]`：当输入目录含多个仓库时自动展开
- `_unique_name(name: str, used: Dict[str, int]) -> str`：批量输出时生成不重复的仓库名
- `_is_github_url(value: str) -> bool`：判断输入是否为 GitHub URL；用于 UI 的仓库输入框
- `_github_slug(url: str) -> Optional[str]`：解析 GitHub URL 为 `owner__repo` 缓存名；由 `_resolve_repo_input()` 调用
- `_resolve_repo_input(repo_input: str, cache_dir: str) -> Optional[str]`：处理单个本地路径或 GitHub URL；必要时 clone 到缓存目录
- `_prepare_cfg(cfg: Dict[str, Any], test_out: Optional[Path]) -> Dict[str, Any]`：为批量模式调整 testgen 输出目录
- `_resolve_repo_inputs(repo_text: str, cache_dir: str) -> List[Dict[str, str]]`：解析多行输入并展开为仓库列表
- `_markdown_to_text(md_text: str) -> str`：将 Markdown 简化为纯文本；用于 PDF fallback
- `_build_pdf_from_markdown(md_text: str) -> Optional[bytes]`：用 reportlab 输出简版 PDF；当 LaTeX 无法编译时调用
- `_render_pdf_preview(pdf_bytes: bytes) -> None`：在 UI 中嵌入 PDF 预览 iframe
- `_compile_latex(tex_path: Path) -> Optional[Path]`：执行 `xelatex`/`tectonic` 编译 report.tex
- `_load_report_sources(out_dir: str, state)`：从 session 或 `review.json` / `testgen.json` 读取数据
- `_make_pdf_bytes(review, testgen, md_text: str) -> Optional[bytes]`：优先使用 `pdf_builder` 生成 PDF
- `_write_report(out_dir: str, review, testgen) -> (Path, Optional[Path])`：生成 `report.md` / `report.tex` / `report.pdf`
- `_existing_report_path(out_dir: str, state) -> Optional[Path]`：定位最近一次报告文件
- `_plot_counts(values: List[str], title: str) -> None`：用 Plotly 绘制分布；不可用时降级为文本条形图
- `_show_findings_table(rows) -> None`：使用 pandas DataFrame 渲染；不可用时用 JSON

#### `config.yaml`
- 统一配置入口；被 `load_config()` 读取后传入 `Orchestrator`

#### `requirements.txt`
- Python 依赖列表（含 `streamlit`、`ruff`、`bandit`、`pip-audit`、`radon`、`reportlab`、`tectonic` 等）

---

### `src/`（核心代码）
#### `src/cli.py`（命令行入口）
- `_load_repo_list(path: str)`：读取多仓库列表文件（忽略空行与注释）
- `_prepare_cfg(cfg, test_out: Optional[Path])`：为 CLI 批处理覆盖 test 输出目录
- `main()`：CLI 入口；通过 `python -m src.cli ...` 调用

#### `src/core/config.py`
- `load_config(path: str) -> Dict[str, Any]`：加载 YAML 配置；被 UI/CLI/Orchestrator 调用

#### `src/core/fs.py`
- `iter_files(repo_path, include_globs, exclude_globs, max_files) -> List[Path]`：按 glob 规则扫描文件；用于 Review/TestGen 的文件列表构建

#### `src/core/subproc.py`
- `run_cmd(cmd, cwd=None, timeout=1800) -> Dict[str, Any]`：统一子进程执行入口；返回 `ok/returncode/stdout/stderr`

#### `src/core/orchestrator.py`
- `class Orchestrator`：调度 Review/TestGen 两条流水线
  - `__init__(cfg)`：保存配置
  - `_file_list(repo_path)`：调用 `iter_files()` 按配置生成扫描文件清单
  - `run_review(repo_path)`：调用 `run_review_pipeline()`
  - `run_testgen(repo_path)`：调用 `run_testgen_pipeline()`

---

### `src/features/review/`（代码审查）
#### `types.py`
- `class ReviewFinding(BaseModel)`：统一发现对象（tool/rule/severity/message/file/line/col/extra）

#### `ast_rules.py`
- `scan_file_ast(path, repo_root)`：扫描单文件 AST，检测 `eval/exec` 与裸 `except`
- `scan_source_ast(source, rel_path)`：扫描字符串源码（用于 Notebook code cell）

#### `ds_rules.py`（数据科学专项规则）
- `scan_file_ds(path, repo_root)`：读取文件并调用 `scan_source_ds()`
- `scan_source_ds(source, rel_path)`：对源码执行 DS 规则扫描，返回 ReviewFinding 列表
- `_call_name(node)`：提取调用名（Name/Attribute）
- `_attr_chain(node)`：提取完整属性链（如 `np.random.rand`）
- `_assigned_names(target)`：解析赋值目标名集合
- `_has_kw(call, name)`：判断调用是否包含关键字参数
- `_is_chained_subscript(node)`：检测链式索引（SettingWithCopy 风险）
- `_is_apply_axis1(call)`：检测 `apply(axis=1)` 场景
- `class _DSVisitor(ast.NodeVisitor)`：核心 DS 规则遍历器
  - `__init__(rel_path)`：初始化别名、随机性、pipeline 统计等状态
  - `_add(rule, severity, message, node)`：统一生成 ReviewFinding
  - `visit_Import/visit_ImportFrom`：记录 numpy/random 的导入别名
  - `visit_Assign/visit_AnnAssign/visit_AugAssign`：检测 SettingWithCopy 与 scaler 变量
  - `visit_Call`：检测随机性、random_state、fit_transform、iterrows、apply(axis=1)、pipeline
  - `_is_seed_call(chain, name, node)`：判断是否显式设置了随机种子
  - `_is_random_usage(chain, name)`：判断是否出现随机调用
  - `finalize()`：生成汇总类规则（如随机未设种子、fit_transform 在分割前等）
- `class _DummyNode`：内部占位，用于补充行号

#### `notebook.py`
- `extract_code_cells(path)`：解析 `.ipynb`，返回 `(cell_index, code)` 列表

#### `parsers.py`
- `parse_ruff_json(stdout)`：解析 ruff JSON 输出为 ReviewFinding
- `parse_bandit_json(stdout)`：解析 bandit JSON 输出为 ReviewFinding
- `parse_pip_audit_json(stdout)`：解析 pip-audit JSON 输出为 ReviewFinding（兼容多种格式）

#### `review_runner.py`
- `run_review_pipeline(repo_path, files, cfg)`：审查主流程；整合 AST/DS/ruff/bandit/pip-audit/radon/mypy

---

### `src/features/testgen/`（测试生成）
#### `ast_extract.py`
- `extract_public_functions(path)`：提取模块中非下划线开头的函数签名与 docstring

#### `templates.py`
- `make_test_module(module_rel, funcs, use_hypothesis)`：生成 pytest 测试模板文本
- `PYTEST_HEADER` / `HYPOTHESIS_HEADER`：模板头部

#### `coverage_runner.py`
- `run_coverage(repo_path, pytest_args)`：执行 `coverage run -m pytest` 与 `coverage report -m`

#### `testgen_runner.py`
- `run_testgen_pipeline(repo_path, files, cfg)`：主流程；生成测试文件 + 可选覆盖率报告

---

### `src/reporting/`（报告生成）
#### `report_builder.py`（Markdown）
- `build_markdown_report(review, testgen)`：生成 `report.md`
- `_truncate(text, limit)`：摘要截断
- `_md_table(headers, rows)`：生成 Markdown 表格
- `_counter_rows(counter)`：统计输出转表格
- `_format_loc(finding)`：格式化定位信息

#### `latex_builder.py`（LaTeX）
- `build_latex_report(review, testgen)`：生成 `report.tex`
- `_latex_preamble()`：LaTeX 预置字体/表格/标题样式
- `latex_escape(text)`：LaTeX 特殊字符转义
- `latex_path(path)`：路径转 `\codepath{}` 格式
- `format_loc(finding)`：格式化定位
- `parse_radon_rows(stdout)`：解析 Radon CC 输出为表格行
- `parse_coverage_rows(stdout)`：解析 coverage 输出为表格行

#### `pdf_builder.py`（ReportLab fallback）
- `build_pdf_report(review, testgen)`：生成 PDF 二进制（若 LaTeX 不可用）
- `_make_table(rows, font_name, col_widths, header)`：统一表格样式
- `_counter_rows(label, counter)`：统计表格行
- `_format_loc(finding)`：定位格式化
- `_truncate(text, limit)`：摘要截断
- `_escape(text)`：HTML 转义用于 Paragraph
- `_para(text, style)`：生成 ReportLab Paragraph
- `_register_cjk_font(pdfmetrics, TTFont)`：尝试注册中文字体（Windows/macOS/Linux）

---

## 常见问题
### 1) pandas 报错（比如 `pandas` 缺 `__version__`）
- 可能被项目中的 `pandas.py` 影子模块覆盖
- 重新安装：`pip install -U pandas`
- UI 已支持降级渲染，避免崩溃

### 2) PDF 报告中文显示异常
- 依赖系统中文字体（Windows 通常为 `Microsoft YaHei` / `SimSun`）
- 若系统缺失中文字体，请安装后重启再生成报告
- PDF 优先使用 LaTeX（xelatex/tectonic）；不可用时回退为简化版 PDF

### 3) pip-audit 输出解析异常
- 某些版本输出格式不一致
- 已做解析兼容，如仍有问题可关闭：
  ```yaml
  review:
    enable_pip_audit: false
  ```

### 4) ruff 参数报错
- 你的 ruff 版本可能不支持 `--format json`
- 可在 `config.yaml` 中调整 `ruff_args`

---

## 扩展与二次开发
### 添加新的 DS 规则
- 入口：`src/features/review/ds_rules.py`
- 规则输出使用 `ReviewFinding`
- 将你的规则归类为新 `rule` 名称

### Notebook 扫描
- 入口：`src/features/review/notebook.py`
- 支持扩展更多 cell 元数据

### UI 调整
- 入口：`app.py`
- 通过 `_inject_css()` 自定义主题与布局




