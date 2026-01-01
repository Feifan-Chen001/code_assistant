# CodeAssistant 智能代码助手（数据科学专项）

一个面向数据科学/机器学习项目的“代码审查 + 自动测例生成 + 报告输出”一体化工具。支持 Python 仓库与 Notebook 扫描，内置数据科学专项规则（复现性/泄漏/Pipeline/性能坑），提供可视化看板与批量实验 CLI。

## 数据科学专项规则（DS Rules）
规则基于 **AST + 启发式**，面向数据科学/ML 项目给出可执行提示。

### 1) 复现性
- **随机数未设 seed**：`random` / `numpy.random` 调用但未见 `seed`
- **sklearn 随机组件缺 `random_state`**：如 `train_test_split`、`KFold`、`RandomForest`、`KMeans`
- **PyTorch 随机数未设 seed**：检测 `torch.rand/torch.randn` 但未 `torch.manual_seed`
- **TensorFlow 随机数未设 seed**：检测 `tf.random.*` 但未 `tf.random.set_seed`

### 2) 数据泄漏启发式
- **`train_test_split` 之前出现 `fit_transform`**：提示可能有数据泄漏

### 3) Pipeline 建议
- **独立 `fit/transform` 但未使用 Pipeline**：`StandardScaler/MinMaxScaler/OneHotEncoder/Imputer` 等

### 4) 高级 ML 规则（Advanced）
- **特征选择未嵌套 CV**：`DS_FEATURE_SELECTION_NO_NESTED_CV`
- **SMOTE 未入 Pipeline**：`DS_IMBALANCE_NOT_IN_PIPELINE`
- **类别不平衡未处理**：`DS_IMBALANCE_UNHANDLED`
- **分类任务 CV 未分层**：`DS_CV_NOT_STRATIFIED`
- **无验证集或未设置 validation_split**：`DS_NO_VALIDATION_SPLIT`
- **在训练集上评估**：`DS_EVAL_ON_TRAIN`
- **评价指标不完整**：`DS_EVALUATION_INCOMPLETE`

### 5) pandas 性能与坑位
- `iterrows` 逐行遍历
- `apply(axis=1)` 行级 apply
- `SettingWithCopy`（如 `df[df.a > 0]["b"] = ...`）

---

## Notebook 支持 支持 ?? 支持
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
- **LLM Actions**：生成修复计划/应用修改/项目推荐（需配置 API Key）
- **Batch results**：多仓库运行时显示汇总，并可切换查看某个仓库的 Review/TestGen/Report

## LLM 智能辅助
用于在报告生成后进行智能修复与推荐。

使用步骤：
1. 侧边栏 LLM Settings 中配置 Base URL / Model / API Key（或设置环境变量）。
2. 运行 Review/TestGen/All 生成报告。
3. 在报告区域点击：Generate fix plan / Apply changes / Get recommendations。

说明：
- Apply Changes 仅在你确认后写回文件，并提供修改文件打包下载。
- 推荐结果基于当前报告与代码上下文，用于对比最佳实践。

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
- **XeLaTeX**（TeX Live / MiKTeX）以获得最佳排版
- Python 依赖列表（含 `streamlit`、`ruff`、`bandit`、`pip-audit`、`radon`、`reportlab` 等）
- Python 依赖列表（含 `streamlit`、`ruff`、`bandit`、`pip-audit`、`radon`、`reportlab` 等）
- `_load_report_sources(out_dir: str, state)`：从 session 或 `review.json` / `testgen.json` 读取数据
- `_make_pdf_bytes(review, testgen, md_text: str) -> Optional[bytes]`：优先使用 `pdf_builder` 生成 PDF
- `_write_report(out_dir: str, review, testgen) -> (Path, Optional[Path])`：生成 `report.md` / `report.tex` / `report.pdf`
- `_existing_report_path(out_dir: str, state) -> Optional[Path]`：定位最近一次报告文件
- `_plot_counts(values: List[str], title: str) -> None`：用 Plotly 绘制分布；不可用时降级为文本条形图
- `_show_findings_table(rows) -> None`：使用 pandas DataFrame 渲染；不可用时用 JSON

#### `config.yaml`
- 统一配置入口；被 `load_config()` 读取后传入 `Orchestrator`

#### `requirements.txt`
- Python 依赖列表（含 `streamlit`、`ruff`、`bandit`、`pip-audit`、`radon`、`reportlab` 等）

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
- **XeLaTeX**（TeX Live / MiKTeX）以获得最佳排版
- Python 依赖列表（含 `streamlit`、`ruff`、`bandit`、`pip-audit`、`radon`、`reportlab` 等）
- Python 依赖列表（含 `streamlit`、`ruff`、`bandit`、`pip-audit`、`radon`、`reportlab` 等）

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






