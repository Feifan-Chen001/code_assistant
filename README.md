# CodeAssistant 智能代码助手（数据科学专项）

面向数据科学/机器学习项目的“代码审查 + 自动测例生成 + 报告输出”一体化工具。支持 Python 仓库与 Notebook 扫描，内置 DS 专项规则与插件规则，提供 Streamlit GUI、批处理 CLI、PDF/LaTeX 报告与可选 LLM 辅助。

## ✨ 功能概览
- 代码审查：AST 规则 + DS 规则 + Ruff/Bandit/pip-audit/Radon/Mypy
- 数据科学规则：复现性、数据泄漏、Pipeline 建议、pandas 性能坑、进阶 ML 评估
- Notebook 支持：扫描 `.ipynb` code cell，并定位到 `file.ipynb#cell-idx`
- 测试生成：从函数签名生成 pytest/Hypothesis 模板（`.py/.ipynb`）
- 覆盖率评估：自动运行 `coverage run -m pytest` + `coverage report -m`
- 报告输出：`report.md` / `report.tex` / `report.pdf`（UI 预览+下载）
- LLM 辅助：修复计划、应用修改、项目推荐
- 批处理：多仓库实验批量运行

---

## 🚀 快速开始

### 依赖要求
- Python 3.8+（建议 3.10/3.11）
- Git（用于克隆 GitHub 仓库）
- 可选：TeX Live / MiKTeX（提供 `xelatex`），或 `tectonic`

### 安装依赖

建议使用虚拟环境
```bash
python -m venv .venv  
.venv\Scripts\activate 
```   
安装依赖                                                     
```bash
pip install -r requirements.txt
```

### 启动 UI
```bash
streamlit run app.py
```
Windows 可用：`start_gui.bat`

### CLI 示例
```bash
python -m src.cli review --repo <path> --out reports
python -m src.cli testgen --repo <path> --out generated_tests
python -m src.cli all --repo <path> --out reports
python -m src.cli batch --repos repos.txt --mode all --out reports_batch
```

---

## 🧭 输入与输出

### 仓库输入（UI 与 CLI 通用）
- 本地路径：`D:/code/my_repo`
- GitHub URL：`https://github.com/user/repo`
- 多行输入：每行一个仓库
- 目录展开：若输入目录包含多个子仓库，会自动展开（依据 `.git`/`pyproject.toml`/`setup.py`/`requirements.txt`）

### GitHub 缓存
- URL 会被 clone 到缓存目录（默认 `Git_repo/owner__repo`）
- 缓存存在时直接复用

### 输出结构
- 单仓库：
  - `reports/`：`review.json` / `testgen.json` / `report.md` / `report.tex` / `report.pdf`
  - `generated_tests/`：生成的测试文件
- 批处理：
  - `reports_batch/<repo_name>/...`

---

## 🖥️ UI 使用说明

### 侧边栏
- 配置文件路径（默认 `config.yaml`）
- 仓库路径或 GitHub 链接（多行）
- GitHub 缓存目录（默认 `Git_repo`）
- 输出目录（默认 `reports`）
- DS 规则开关 / 插件规则开关 / 工具开关 / 日志配置

### Review 看板
- 指标卡：问题总数、DS 规则、插件规则、其它工具
- 规则详情：DS 规则类型与严重性、插件规则分类概览
- 总览图表：严重性分布、工具分布
- Top 20 问题表：按严重性与工具排序

### TestGen 看板
- 生成文件数、覆盖函数数、输出目录
- 覆盖率摘要（如启用 coverage）

### Report 看板
- `report.pdf` 预览与下载
- `report.md` 下载

### LLM Actions
- Generate fix plan：生成修复计划
- Apply changes：应用修改并打包下载
- Get recommendations：推荐相关优质项目

---

## ⚙️ 配置说明（config.yaml）

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
    - "**/.mypy_cache/**"
    - "**/.pytest_cache/**"
    - "**/.coverage"
review:
  enable_ruff: true
  enable_mypy: true
  enable_bandit: true
  enable_pip_audit: true
  enable_radon: true
  enable_ds_rules: true
  enable_ds_rules_advanced: true
  enable_notebook: true
  # optional: force_enable_advanced_ds: true

testgen:
  output_dir: "generated_tests"
  use_hypothesis: true
  max_functions: 200
coverage:
  enable: true
  pytest_args: ["-q"]
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
  base_url: "https://api.openai.com/v1"
  temperature: 0.2
  max_tokens: 1200
  timeout: 60
  allow_empty_key: false
  # optional: allow_new_files: true
```

说明：
- Streamlit 模式为性能默认禁用高级 DS 规则；如需启用，可在 `config.yaml` 添加 `review.force_enable_advanced_ds: true`。
- `severity_config.yaml` 可用于规则严重性映射。

---

## 📊 数据科学专项规则（DS Rules）

### 1) 复现性
- 随机数未设 seed（`random` / `numpy.random`）
- sklearn 组件缺 `random_state`
- PyTorch 随机数未设 `torch.manual_seed`
- TensorFlow 随机数未设 `tf.random.set_seed`

### 2) 数据泄漏启发式
- `train_test_split` 之前出现 `fit_transform`

### 3) Pipeline 建议
- 独立 `fit/transform` 但未使用 `Pipeline`

### 4) 高级 ML 规则（Advanced）
- 特征选择未嵌套 CV
- SMOTE 未入 Pipeline
- 类别不平衡未处理
- CV 未分层
- 无验证集或未设置 validation_split
- 在训练集上评估
- 评价指标不完整

### 5) pandas 性能与坑位
- `iterrows`
- `apply(axis=1)`
- `SettingWithCopy`

---

## 📓 Notebook 支持
- Review：提取 code cell 作为“虚拟文件”扫描（`notebook.ipynb#cell-idx`）
- TestGen：将 notebook 代码写入 `generated_tests/_notebooks/nb_<slug>.py`

---

## 📄 报告与产物
- `report.md`：Markdown 报告
- `report.tex`：LaTeX 报告
- `report.pdf`：优先 `xelatex`（双跑修复目录），其次 `tectonic`，最后回退 reportlab

---

## 🤖 LLM 智能辅助
- 生成修复计划
- 应用修改并打包下载
- 推荐相关优质项目

配置方式：在 `config.yaml` 中配置 `llm`，或设置环境变量（默认 `OPENAI_API_KEY`）。

---

## 🧩 目录结构与函数索引（逐文件/逐函数）
说明：仅覆盖本项目源码与入口文件。`my_repo/`、`Git_repo/` 为外部样例仓库缓存；`reports/`、`generated_tests/` 为运行时产物。

### 根目录
#### `app.py`（Streamlit GUI 入口）
- `main()`：应用入口
- `_inject_css()`：注入 UI CSS 与背景
- `_hide_theme_picker()`：隐藏 Streamlit 主题切换入口
- `_ensure_dirs(out_dir)`：创建输出目录
- `_parse_repo_inputs(text)`：解析多行仓库输入
- `_is_repo_root(path)`：判断是否为仓库根
- `_expand_local_repos(path)`：展开多仓库目录
- `_unique_name(name, used)`：批量唯一命名
- `_is_github_url(value)`：判断 GitHub URL
- `_github_slug(url)`：生成缓存目录名
- `_resolve_repo_input(repo_input, cache_dir)`：解析本地路径或克隆 URL
- `_prepare_cfg(cfg, test_out)`：批量模式覆盖 test 输出目录
- `_resolve_repo_inputs(repo_text, cache_dir)`：解析多仓库输入
- `_markdown_to_text(md_text)`：Markdown -> 纯文本
- `_build_pdf_from_markdown(md_text)`：reportlab PDF 回退
- `_render_pdf_preview(pdf_bytes)`：UI 内嵌 PDF 预览
- `_compile_latex(tex_path)`：调用 xelatex / tectonic
- `_load_report_sources(out_dir, state)`：从 session 或磁盘读取报告源
- `_make_pdf_bytes(review, testgen, md_text)`：生成 PDF 二进制
- `_write_report(out_dir, review, testgen)`：写出 md/tex/pdf
- `_existing_report_path(out_dir, state)`：定位最近报告路径
- `_plotly_go()`：安全导入 plotly（规避 pandas 影子模块）
- `_plot_counts(values, title)`：绘制分布图（支持原始值或 `(label, count)`）
- `_show_findings_table(rows)`：DataFrame/JSON 展示
- `_truncate_text(text, limit)`：截断长文本
- `_extract_json_block(text)`：从 LLM 输出中提取 JSON
- `_normalize_plan(raw)` / `_fallback_plan()`：修复计划规范化
- `_normalize_recommendations(raw)`：推荐结果规范化
- `_format_file_context(files)`：整理上下文片段
- `_collect_context_files(repo_root, review_src)`：提取与发现相关的文件片段
- `_llm_ready(cfg)`：校验 LLM 配置
- `_apply_llm_changes(repo_root, files, allow_new)`：应用 LLM 修改
- `_build_changes_zip(changed, repo_root)`：打包修改文件
- `_llm_generate_plan(...)` / `_llm_generate_changes(...)` / `_llm_generate_recommendations(...)`

#### `config.yaml`
- 默认运行配置

#### `severity_config.yaml`
- 规则严重性映射表

#### `requirements.txt`
- Python 依赖列表

#### `.streamlit/config.toml`
- UI 主题（固定 light）

#### `start_gui.bat`
- Windows 启动脚本（基于 `.venv`）

### `src/cli.py`
- `_load_repo_list(path)`：读取仓库清单
- `_prepare_cfg(cfg, test_out)`：覆盖 test 输出目录
- `main()`：CLI 入口

### `src/core/config.py`
- `load_config(path, validate=True)`
- `load_config_strict(path)`

### `src/core/config_validator.py`
- `AssistantConfig` / `ReviewConfig` / `TestGenConfig` / `CoverageConfig` / `CodeAssistantConfig`
- `validate_config(cfg)`

### `src/core/fs.py`
- `iter_files(repo_path, include_globs, exclude_globs, max_files)`

### `src/core/llm_client.py`
- `build_llm_config(cfg)`
- `_extract_text(payload)`
- `llm_chat(messages, cfg)`

### `src/core/logger.py`
- `setup_logger()` / `get_logger()`
- `PerformanceLogger` / `ColoredFormatter` / `StructuredFormatter`

### `src/core/orchestrator.py`
- `Orchestrator`：`_file_list()` / `run_review()` / `run_testgen()`

### `src/core/subproc.py`
- `run_cmd(cmd, cwd=None, timeout=1800)`

### `src/features/review/`
- `types.py`：`ReviewFinding`
- `ast_rules.py`：`scan_file_ast`, `scan_source_ast`
- `ds_rules.py`：`scan_file_ds`, `scan_source_ds` 及若干辅助函数
- `ds_rules_advanced.py`：`scan_file_advanced_ds`, `scan_source_advanced_ds`
- `builtin_rules.py`：内置规则类与 `register_builtin_rules()`
- `rule_plugin.py`：规则注册与分类
- `notebook.py`：`extract_code_cells()`
- `parsers.py`：`parse_ruff_json`, `parse_bandit_json`, `parse_pip_audit_json`
- `review_runner.py`：`run_review_pipeline` 等

### `src/features/testgen/`
- `ast_extract.py`：`extract_public_functions`, `extract_public_functions_from_source`
- `templates.py`：`make_test_module`, `PYTEST_HEADER`, `HYPOTHESIS_HEADER`
- `coverage_runner.py`：`run_coverage`
- `testgen_runner.py`：`run_testgen_pipeline` 与 notebook 处理

### `src/reporting/`
- `report_builder.py`：`build_markdown_report` 与表格辅助函数
- `latex_builder.py`：`build_latex_report` 与表格解析
- `pdf_builder.py`：reportlab PDF 回退与字体注册

---

## 🧰 常见问题

### 图表显示为 #
Plotly 失败会降级为文本条形图。请确认 Streamlit 运行环境中安装了 `plotly`，并避免仓库里有 `pandas.py` 等同名影子模块。

### 同一仓库有时 0 个问题
多因输入路径错误、扫描文件为空、或工具在 UI 中被关闭。

### PDF 中文显示异常
请安装中文字体并使用 `xelatex` 编译；无法编译时会回退到 reportlab 简版 PDF。

---

## 🛠️ 扩展与二次开发
- DS 规则：`src/features/review/ds_rules.py`
- 插件规则：`src/features/review/rule_plugin.py`
- UI 调整：`app.py` 中的 `_inject_css()` 与布局块
- 报告排版：`src/reporting/latex_builder.py`
