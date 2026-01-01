# 🚀 CodeAssistant - 智能代码助手

<div align="center">

**一体化数据科学代码审查、测试生成与报告工具**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*专为数据科学与机器学习项目设计的代码质量保障工具*

</div>

---

## 📋 目录

- [✨ 项目特色](#-项目特色)
- [🎯 核心功能](#-核心功能)
- [🛠️ 技术栈](#️-技术栈)
- [📦 快速开始](#-快速开始)
- [💻 使用指南](#-使用指南)
- [⚙️ 配置说明](#️-配置说明)
- [📊 数据科学规则](#-数据科学规则)
- [📝 报告输出](#-报告输出)
- [🔧 高级功能](#-高级功能)
- [❓ 常见问题](#-常见问题)
- [🤝 贡献指南](#-贡献指南)

---

## ✨ 项目特色

### 🎯 为数据科学项目量身定制

- **专项规则检测**：随机种子、数据泄漏、Pipeline 建议、性能陷阱
- **Notebook 支持**：完整支持 Jupyter Notebook (.ipynb) 文件分析
- **ML 工作流优化**：针对 sklearn、pandas、numpy 等常用库的最佳实践检查

### 🚀 一站式解决方案

```
代码审查 → 问题发现 → 测试生成 → 报告输出
```

- ✅ 静态代码分析（AST + Linting）
- ✅ 安全漏洞扫描（Bandit）
- ✅ 复杂度评估（Radon）
- ✅ 自动生成单元测试
- ✅ 生成 PDF/Markdown 报告

### 🎨 现代化 GUI 界面

- **Streamlit Web 界面**：无需安装，浏览器即用
- **实时进度显示**：透明的执行过程
- **批量处理**：支持多仓库并行分析
- **可视化报告**：图表展示问题分布

---

## 🎯 核心功能

### 1️⃣ 代码审查（Code Review）

#### 📌 基础检查
- **语法规范**：Ruff Linter 快速检查
- **安全漏洞**：Bandit 安全扫描
- **代码复杂度**：Radon 圈复杂度分析
- **类型检查**：MyPy 静态类型验证（可选）

#### 📌 数据科学专项规则

##### 🎯 基础规则（7项）
| 规则 ID | 严重性 | 检测内容 |
|---------|--------|----------|
| `DS_RANDOM_SEED` | MEDIUM | 缺少随机种子设置（影响可复现性） |
| `DS_SKLEARN_RANDOM_STATE` | MEDIUM | sklearn 模型缺少 random_state |
| `DS_LEAKAGE_FIT_BEFORE_SPLIT` | HIGH | 数据泄露（在划分前拟合预处理器） |
| `DS_PIPELINE_SUGGEST` | LOW | 缩放器未在 Pipeline 中使用 |
| `DS_MODEL_PICKLE_UNSAFE` | MEDIUM | 使用 pickle 序列化模型（安全与兼容性） |
| `DS_HYPERPARAMS_HARDCODED` | LOW | 模型超参数硬编码 |
| `DS_PANDAS_PERFORMANCE` | LOW | pandas 低效操作（iterrows/apply） |

##### 🚀 高级规则（5项）
| 规则 ID | 严重性 | 检测内容 |
|---------|--------|----------|
| `DS_FEATURE_SELECTION_NO_NESTED_CV` | MEDIUM | 特征选择后未使用嵌套 CV |
| `DS_IMBALANCE_NOT_IN_PIPELINE` | HIGH | SMOTE 等采样方法未在 Pipeline 中 |
| `DS_IMBALANCE_UNHANDLED` | LOW | 未处理数据不平衡 |
| `DS_EVALUATION_INCOMPLETE` | LOW | 评估指标不足 |
| `DS_TEST_SET_REUSE` | HIGH | 测试集被重复使用 |

##### 🔌 插件规则（4项）
- `PY_MUTABLE_DEFAULT_ARG`：可变默认参数陷阱
- `PY_GLOBAL_VARIABLE`：过度使用全局变量
- `PY_RESOURCE_LEAK`：文件资源未使用 with 语句
- `PY_LOOP_INVARIANT`：循环内的不变表达式

### 2️⃣ 测试生成（Test Generation）

- **AST 解析**：自动提取公开函数和类
- **Pytest 模板**：生成标准的单元测试框架
- **覆盖率分析**：集成 coverage.py 评估测试覆盖率
- **智能命名**：基于函数名生成有意义的测试名称

### 3️⃣ 报告生成（Report Building）

#### 📄 支持格式
- **Markdown**：轻量级文本格式，便于版本控制
- **PDF**：专业打印格式（LaTeX/ReportLab）
- **JSON**：结构化数据，便于二次处理

#### 📊 报告内容
- 执行摘要（发现问题总数、严重性分布）
- 详细问题列表（文件、行号、规则、消息）
- 数据可视化（问题分布图表）
- 测试生成统计
- 建议与最佳实践

---

## 🛠️ 技术栈

### 核心框架
- **Python 3.11+**：主要开发语言
- **Streamlit 1.34+**：Web GUI 框架
- **Pydantic v1.10**：配置验证与数据模型

### 代码分析工具
- **AST Parser**：Python 内置抽象语法树
- **Ruff**：极速 Python Linter
- **Bandit**：安全漏洞扫描器
- **Radon**：代码复杂度分析
- **MyPy**：静态类型检查

### 报告生成
- **ReportLab**：PDF 生成库
- **XeLaTeX / Tectonic**：高质量 LaTeX 编译器
- **Plotly**：交互式图表库

### 配置管理
- **PyYAML**：YAML 配置文件解析
- **TOML**：pyproject.toml 支持

---

## 📦 快速开始

### 1. 环境准备

#### 前置要求
```bash
# Python 3.11 或更高版本
python --version

# Git（用于克隆 GitHub 仓库）
git --version
```

#### 克隆项目
```bash
git clone <your-repo-url>
cd CodeAssistant
```

### 2. 安装依赖

#### 方式 1：使用虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 方式 2：使用 Conda
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate codeassistant
```

### 3. 启动应用

#### GUI 模式（推荐）
```bash
streamlit run app.py
```
浏览器会自动打开 `http://localhost:8501`

#### CLI 模式
```bash
# 单仓库分析
python -m src.cli --repo my_repo --out reports

# 批量分析
python -m src.cli --repos repo1 repo2 repo3 --out batch_reports

# 完整流程
python -m src.cli --repo my_repo --review --testgen --report
```

---

## 💻 使用指南

### 🖥️ GUI 界面使用

#### 主工作区

1. **配置工作区**（侧边栏）
   ```
   📂 工作区
   ├─ 配置文件：config.yaml
   ├─ 仓库路径：本地路径或 GitHub URL
   ├─ GitHub 缓存：Git_repo
   └─ 输出文件夹：reports
   ```

2. **启用检查规则**（侧边栏）
   ```
   🔧 Advanced Settings
   ├─ 🎯 DS 规则（数据科学）
   │   ├─ ✓ 启用基础 DS 规则
   │   └─ ☐ 启用高级 DS 规则（较慢）
   ├─ 🔌 规则插件
   ├─ 🛠️ 工具
   │   ├─ ✓ Ruff
   │   ├─ ✓ Bandit
   │   ├─ ✓ Radon
   │   └─ ☐ MyPy
   └─ 📋 日志设置
   ```

3. **执行操作**（主面板）
   ```
   🚀 执行操作
   ├─ 🔍 代码审查 ──→ 发现问题
   ├─ 🧪 测试生成 ──→ 生成单元测试
   └─ ⚡ 全部运行 ──→ 一键完成所有流程
   ```

4. **查看结果**
   - **📋 审查报告**：问题统计、严重性分布、详细列表
   - **📊 数据可视化**：问题分布图表
   - **🧪 测试生成结果**：生成文件数、输出目录
   - **📄 综合报告**：下载 PDF/Markdown 格式报告

#### 规则文档页面

查看所有可用规则的详细说明、示例代码和最佳实践。

#### 配置管理页面

实时验证 `config.yaml` 配置文件，显示 Pydantic 数据模型。

### 📝 命令行使用

#### 基础命令
```bash
# 仅代码审查
python -m src.cli --repo my_repo --review

# 仅测试生成
python -m src.cli --repo my_repo --testgen

# 完整流程
python -m src.cli --repo my_repo --all

# 指定配置文件
python -m src.cli --repo my_repo --config custom_config.yaml
```

#### 批量处理
```bash
# 分析多个仓库
python -m src.cli \
  --repos repo1 repo2 repo3 \
  --out batch_reports \
  --all

# 从文件读取仓库列表
python -m src.cli \
  --repos-file repos.txt \
  --out batch_reports
```

#### GitHub 仓库
```bash
# 克隆并分析 GitHub 仓库
python -m src.cli \
  --repo https://github.com/user/repo.git \
  --cache-dir Git_repo \
  --all
```

---

## ⚙️ 配置说明

### config.yaml 结构

```yaml
assistant:
  max_files: 2000              # 最大扫描文件数
  include_globs:               # 包含的文件模式
    - "**/*.py"
    - "**/*.ipynb"
  exclude_globs:               # 排除的文件模式
    - "**/node_modules/**"
    - "**/.venv/**"
    - "**/test_*.py"

review:
  enable_ds_rules: true        # 启用 DS 基础规则
  enable_ds_rules_advanced: false  # 启用 DS 高级规则（较慢）
  enable_ruff: true            # 启用 Ruff
  enable_bandit: true          # 启用 Bandit
  enable_radon: true           # 启用 Radon
  enable_mypy: false           # 启用 MyPy（较慢）
  
  ruff_args: ["--select=E,F,W"]  # Ruff 参数
  bandit_args: ["-ll"]         # Bandit 参数

testgen:
  output_dir: "generated_tests"  # 测试输出目录
  template: "pytest"           # 测试框架模板
  include_private: false       # 包含私有函数

coverage:
  enable: true                 # 启用覆盖率分析
  min_coverage: 80             # 最低覆盖率要求
  report_format: "html"        # 报告格式
```

### severity_config.yaml（严重级别配置）

```yaml
rules:
  # HIGH - 导致错误结果或严重安全问题
  DS_LEAKAGE_FIT_BEFORE_SPLIT: "high"       # 导致无效结论（数据泄漏）
  DS_IMBALANCE_NOT_IN_PIPELINE: "high"      # 数据泄漏风险
  DS_TEST_SET_REUSE: "high"                 # 测试集重复使用导致过拟合
  
  # MEDIUM - 影响可复现性、兼容性或有安全隐患
  DS_RANDOM_SEED: "medium"                  # 影响可复现性（不影响正确性）
  DS_SKLEARN_RANDOM_STATE: "medium"         # 影响可复现性
  DS_MODEL_PICKLE_UNSAFE: "medium"          # 安全风险和兼容性问题
  DS_FEATURE_SELECTION_NO_NESTED_CV: "medium"
  
  # LOW - 最佳实践建议和性能优化
  DS_PIPELINE_SUGGEST: "low"                # 最佳实践建议
  DS_HYPERPARAMS_HARDCODED: "low"           # 可维护性
  DS_PANDAS_PERFORMANCE: "low"              # 性能优化
  DS_IMBALANCE_UNHANDLED: "low"
  DS_EVALUATION_INCOMPLETE: "low"
```

---

## 📊 数据科学规则

### 规则详解

#### 🔴 HIGH - 导致错误结果或严重安全问题

##### DS_LEAKAGE_FIT_BEFORE_SPLIT
**问题**：在数据拆分前进行 fit_transform，导致测试集信息泄漏，评估结果无效

**错误示例**：
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ❌ 错误：在拆分前 fit（数据泄漏）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)
```

**正确示例**：
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ✅ 正确：先拆分再 fit
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 只 transform，不 fit
```

#### 🟡 MEDIUM - 影响可复现性、兼容性或有安全隐患

##### DS_RANDOM_SEED
**问题**：使用随机性但未设置种子，导致结果不可复现（但不影响正确性）

**错误示例**：
```python
import random
import numpy as np

# ❌ 未设置种子（结果每次不同）
data = np.random.randn(100)
sample = random.sample(range(100), 10)
```

**正确示例**：
```python
import random
import numpy as np

# ✅ 设置种子
random.seed(42)
np.random.seed(42)

data = np.random.randn(100)
sample = random.sample(range(100), 10)
```

##### DS_MODEL_PICKLE_UNSAFE
**问题**：使用 pickle 序列化模型，存在安全风险和跨版本兼容性问题

**推荐方案**：
```python
# ✅ 使用模型专用的保存方法
import joblib
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 使用 joblib（比 pickle 更高效安全）
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')
```

#### 🟢 LOW - 最佳实践建议和性能优化

##### DS_PIPELINE_SUGGEST
**问题**：未使用 Pipeline 封装预处理和模型，容易出错

**推荐示例**：
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ✅ 使用 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

##### DS_PANDAS_PERFORMANCE
**问题**：使用低效的 pandas 操作（性能优化建议）

**错误示例**：
```python
# ❌ 低效：iterrows
for idx, row in df.iterrows():
    df.loc[idx, 'new_col'] = row['col1'] + row['col2']
```

**正确示例**：
```python
# ✅ 高效：向量化
df['new_col'] = df['col1'] + df['col2']
```

---

## 📝 报告输出

### 报告格式

#### 1. Markdown 报告 (report.md)
```markdown
# Code Review Report

## Summary
- Total Findings: 42
- High Severity: 8
- Medium Severity: 20
- Low Severity: 14

## Findings by Tool
- ds-rule: 15 findings
- ruff: 18 findings
- bandit: 9 findings

## Detailed Findings
...
```

#### 2. PDF 报告 (report.pdf)
- 专业排版，适合打印和分享
- 包含图表和统计信息
- 支持中文字体

#### 3. JSON 数据 (review.json / testgen.json)
```json
{
  "findings": [
    {
      "tool": "ds-rule",
      "rule": "DS_RANDOM_SEED",
      "severity": "high",
      "file": "src/model.py",
      "line": 42,
      "message": "Random seed not set before using np.random"
    }
  ],
  "summary": {
    "total": 42,
    "by_severity": {"high": 8, "medium": 20, "low": 14}
  }
}
```

---

## 🔧 高级功能

### 批量处理

#### 配置批量仓库
```python
# repos.txt
my_repo
Git_repo/scikit-learn__scikit-learn
https://github.com/pandas-dev/pandas.git
```

```bash
python -m src.cli --repos-file repos.txt --all
```

#### 查看批量结果
```
reports/
├── repo1/
│   ├── review.json
│   ├── testgen.json
│   └── report.pdf
├── repo2/
│   ├── review.json
│   └── report.pdf
└── summary.json
```

### Notebook 支持

工具会自动：
1. 解析 `.ipynb` 文件
2. 提取 code cell 内容
3. 生成虚拟 Python 文件
4. 定位问题到具体 cell

**示例输出**：
```
File: notebook.ipynb (Cell 5)
Line: 3
Rule: DS_RANDOM_SEED
Message: np.random used without setting seed
```

### 插件系统

#### 创建自定义规则

```python
# src/features/review/custom_rule.py
from src.features.review.rule_plugin import register_rule, RulePlugin

@register_rule(
    rule_id="CUSTOM_NO_PRINT",
    category="Best Practice",
    severity="low",
    description="Avoid using print() in production code"
)
class NoPrintRule(RulePlugin):
    def check(self, node, findings, file_path):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                findings.append({
                    "rule": "CUSTOM_NO_PRINT",
                    "severity": "low",
                    "message": "Avoid print() in production",
                    "line": node.lineno
                })
```

---

## ❓ 常见问题

### Q1: GUI 运行 "Run All" 时卡顿？

**A**: 高级 DS 规则会显著增加扫描时间。建议：
- 首次使用时禁用高级规则
- 对于大型仓库（>1000 文件），只启用基础规则
- 使用进度条观察执行状态

### Q2: PDF 预览显示旧版本？

**A**: 已修复。现在会从最新的 Markdown 报告重新生成 PDF。

### Q3: 严重级别分类标准？

**A**: 我们使用三级分类：
- **HIGH**：导致错误结果（数据泄漏）或严重安全问题
- **MEDIUM**：影响可复现性、兼容性或有安全隐患（但不致命）
- **LOW**：最佳实践建议和性能优化

例如：random seed 是 MEDIUM（影响可复现性但不影响正确性），数据泄漏是 HIGH（导致无效结论）

### Q4: 如何自定义严重级别？

**A**: 编辑 `severity_config.yaml`：
```yaml
rules:
  DS_RANDOM_SEED: "medium"  # 改为你需要的级别
  DS_LEAKAGE_FIT_BEFORE_SPLIT: "high"
```

### Q5: 如何添加自定义排除规则？

**A**: 编辑 `config.yaml`：
```yaml
assistant:
  exclude_globs:
    - "**/migrations/**"
    - "**/test_*.py"
    - "**/your_custom_path/**"
```

### Q5: Pydantic 版本冲突？

**A**: 本项目使用 Pydantic v1.10。如果环境中有 v2，请：
```bash
pip install "pydantic>=1.10,<2.0"
```

---

## 🤝 贡献指南

### 贡献方式

1. **报告 Bug**：在 Issues 中详细描述问题
2. **提出功能建议**：描述您的需求场景
3. **提交 Pull Request**：
   - Fork 项目
   - 创建特性分支 (`git checkout -b feature/AmazingFeature`)
   - 提交更改 (`git commit -m 'Add some AmazingFeature'`)
   - 推送到分支 (`git push origin feature/AmazingFeature`)
   - 打开 Pull Request

### 开发指南

#### 项目结构
```
CodeAssistant/
├── app.py                      # Streamlit GUI 入口
├── config.yaml                 # 主配置文件
├── severity_config.yaml        # 严重级别配置
├── requirements.txt            # Python 依赖
├── src/
│   ├── cli.py                 # 命令行入口
│   ├── core/                  # 核心模块
│   │   ├── config.py          # 配置加载
│   │   ├── config_validator.py  # Pydantic 验证
│   │   ├── orchestrator.py    # 流程编排
│   │   ├── fs.py              # 文件系统工具
│   │   └── logger.py          # 日志系统
│   ├── features/              # 功能模块
│   │   ├── review/            # 代码审查
│   │   │   ├── review_runner.py
│   │   │   ├── ds_rules.py    # DS 基础规则
│   │   │   ├── ds_rules_advanced.py  # DS 高级规则
│   │   │   ├── rule_plugin.py  # 插件系统
│   │   │   └── builtin_rules.py  # 内置规则
│   │   └── testgen/           # 测试生成
│   │       └── testgen_runner.py
│   └── reporting/             # 报告生成
│       ├── report_builder.py  # Markdown 构建
│       ├── pdf_builder.py     # PDF 构建
│       └── latex_builder.py   # LaTeX 构建
├── tests/                     # 单元测试
└── generated_tests/           # 生成的测试文件
```

#### 代码规范
- 遵循 PEP 8 风格指南
- 使用类型提示（Type Hints）
- 编写文档字符串（Docstrings）
- 提交前运行 `ruff check`

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🌟 致谢

感谢以下开源项目：

- [Streamlit](https://streamlit.io/) - 强大的 Web 应用框架
- [Ruff](https://github.com/astral-sh/ruff) - 极速 Python Linter
- [Bandit](https://github.com/PyCQA/bandit) - 安全漏洞扫描
- [Radon](https://github.com/rubik/radon) - 代码复杂度分析
- [Pydantic](https://pydantic-docs.helpmanual.io/) - 数据验证框架

---

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：

- **Issues**：[GitHub Issues](https://https://github.com/Feifan-Chen001/code_assistant/issues)
- **Email**：cff-yyds@sjtu.edu.cn

---

<div align="center">

**⭐ 如果觉得项目有帮助，请给个 Star！⭐**

Made with ❤️ for Data Science Community

</div>
