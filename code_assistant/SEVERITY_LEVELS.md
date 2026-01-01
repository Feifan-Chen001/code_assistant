# 代码审查严重级别指南
# Code Review Severity Levels Guide

## 目录 | Table of Contents
- [概述](#概述)
- [严重级别定义](#严重级别定义)
- [规则分类](#规则分类)
- [修复优先级](#修复优先级)
- [配置说明](#配置说明)
- [最佳实践](#最佳实践)

---

## 概述

本文档定义了代码审查系统中各类问题的严重级别（HIGH、MEDIUM、LOW），帮助开发人员：
- 快速识别最关键的问题
- 合理分配修复资源
- 建立清晰的优化优先级

严重级别基于问题对**模型有效性、系统安全性和代码质量**的影响。

---

## 严重级别定义

### 🔴 HIGH（高）- 必须立即修复

**定义**: 严重问题，直接导致错误结果、数据泄漏或系统安全问题

**特征**:
- ✗ 导致模型结果错误或严重失真（数据泄漏）
- ✗ 安全漏洞，可能被恶意利用
- ✗ 数据完整性问题，可能丢失或破坏数据

**修复优先级**: 🔴🔴🔴 **最高**

**修复时间**: 立即（同天）

### 📊 示例规则

| 规则 | 影响 | 原因 |
|------|------|------|
| **DS_LEAKAGE_FIT_BEFORE_SPLIT** | 模型评估无效 | 在测试集上拟合模型，导致数据泄漏 |
| **DS_IMBALANCE_NOT_IN_PIPELINE** | 数据泄漏风险 | SMOTE 等采样方法未在 Pipeline 中 |
| **DS_TEST_SET_REUSE** | 模型过拟合 | 测试集被重复使用调参 |
| **DS_PANDAS_SETTINGWITHCOPY** | 数据被意外修改 | 破坏原始数据集的完整性 |
| **AST_HARDCODED_PASSWORD** | 安全漏洞 | 凭证硬编码，版本控制中暴露 |
| **B201 (Flask debug=True)** | 远程代码执行 | 生产环境调试模式导致的严重漏洞 |
| **B301 (pickle.loads)** | 远程代码执行 | 反序列化不可信数据，任意代码执行 |
| **pip_audit_vulnerability** | 依赖漏洞 | 依赖包存在已知安全漏洞 |

---

### 🟡 MEDIUM（中）- 应该尽快修复

**定义**: 中等问题，影响可复现性、兼容性或代码质量

**特征**:
- ⚠ 结果不可复现，但不影响正确性
- ⚠ 兼容性问题或安全隐患（非致命）
- ⚠ 可能导致后续问题的潜在风险
- ⚠ 代码质量问题，难以维护或测试
- ⚠ 风格不一致，降低可读性
- ⚠ 可能的运行时错误

**修复优先级**: 🟡🟡 **中等**

**修复时间**: 本周内

### 📊 示例规则

| 规则 | 影响 | 原因 |
|------|------|------|
| **DS_RANDOM_SEED** | 实验不可复现 | 没有固定随机数种子，结果无法重现 |
| **DS_SKLEARN_RANDOM_STATE** | 模型不可复现 | sklearn 模型缺少 random_state 参数 |
| **DS_MODEL_PICKLE_UNSAFE** | 安全与兼容性 | pickle 有安全风险和跨版本兼容问题 |
| **DS_FEATURE_SELECTION_NO_NESTED_CV** | 评估偏差 | 特征选择未用嵌套CV，可能过拟合 |
| **AST_BROAD_EXCEPTION** | 隐藏错误 | 捕获所有异常，难以调试 |
| **B603 (subprocess shell=True)** | 命令注入风险 | 用户输入可能破坏命令 |
| **complexity_high** | 维护困难 | 函数复杂度高，测试覆盖不足 |
| **AST_BARE_EXCEPT** | 捕获系统异常 | 会捕获 KeyboardInterrupt、SystemExit |

---

### 🟢 LOW（低）- 可后续优化

**定义**: 低级问题，主要是代码风格或非关键建议

**特征**:
- ℹ 代码风格或格式问题
- ℹ 未使用的代码或导入
- ℹ 次要的最佳实践建议
- ℹ 可选的性能优化

**修复优先级**: 🟢 **最低**

**修复时间**: 有时间再处理（每月/每季度）

### 📊 示例规则

| 规则 | 影响 | 原因 |
|------|------|------|
| **DS_PIPELINE_SUGGEST** | 代码可维护性 | 未使用 Pipeline 导致代码结构混乱 |
| **DS_HYPERPARAMS_HARDCODED** | 难以优化 | 超参数硬编码，无法进行网格搜索 |
| **DS_PANDAS_ITERROWS** | 性能严重下降 | 逐行迭代，1000行数据可能慢10倍+ |
| **DS_PANDAS_APPLY_AXIS1** | 性能问题 | apply(axis=1) 慢于向量化操作 |
| **DS_IMBALANCE_UNHANDLED** | 模型性能 | 未处理类不平衡，可能影响少数类 |
| **DS_EVALUATION_INCOMPLETE** | 评估不全面 | 只用单一指标，无法全面评估 |
| **AST_UNUSED_IMPORT** | 代码冗余 | 导入但未使用，增加复杂度 |
| **W (Ruff warnings)** | 代码风格 | 空行多余、缩进问题等 |
| **B404 (subprocess import)** | 代码质量 | 仅提示，需谨慎使用 |

---

## 规则分类

### 按风险领域分类

#### 1️⃣ 数据泄露 (Data Leakage) - 🔴 **HIGH**

**问题**: 在模型评估前，使用了应该来自测试集的信息

**常见情况**:
```python
# ❌ 错误示例 - 数据泄露
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✓ 正确做法
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

**影响**: 模型性能严重高估，论文结果无效

**修复**: 使用 Pipeline，确保所有预处理在训练集上拟合

---

#### 2️⃣ 可复现性 (Reproducibility) - 🔴 **HIGH**

**问题**: 没有固定随机数种子，实验无法复现

**常见情况**:
```python
# ❌ 错误示例 - 结果每次都不同
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ✓ 正确做法
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 也需要在拆分时固定种子
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**影响**: 无法复现论文结果，合作者无法验证

**修复**: 设置 `random_state=42` 或其他常数

---

#### 3️⃣ 安全问题 (Security) - 🔴 **HIGH**

**问题**: 代码可能被恶意利用

**常见情况**:
```python
# ❌ 错误示例 - 多重安全问题
import pickle
password = "admin123"  # 硬编码密码
with open("model.pkl", "rb") as f:
    model = pickle.load(f)  # pickle 不安全

import subprocess
user_input = input("Enter command: ")
subprocess.call(user_input, shell=True)  # 命令注入

# ✓ 正确做法
import joblib  # 或用 dill、cloudpickle
password = os.environ.get("DB_PASSWORD")  # 环境变量
model = joblib.load("model.pkl")
args = user_input.split()  # 避免 shell=True
subprocess.run(args, check=True)
```

**影响**: 系统被攻击、数据泄露、任意代码执行

**修复**: 
- 避免硬编码密码，使用环境变量
- 用 joblib 替代 pickle
- subprocess 避免 `shell=True`

---

#### 4️⃣ 数据完整性 (Data Integrity) - 🔴 **HIGH**

**问题**: 原始数据被意外修改

**常见情况**:
```python
# ❌ 错误示例 - SettingWithCopyWarning
df_filtered = df[df['age'] > 18]
df_filtered['salary'] = df_filtered['salary'] * 1.1  # 警告！

# ✓ 正确做法
df_filtered = df[df['age'] > 18].copy()
df_filtered['salary'] = df_filtered['salary'] * 1.1

# 或使用 .loc[]
df.loc[df['age'] > 18, 'salary'] *= 1.1
```

**影响**: 数据被意外修改，分析结果不一致

**修复**: 使用 `.copy()` 或 `.loc[]`

---

#### 5️⃣ 性能问题 (Performance) - 🟡 **MEDIUM**

**问题**: 低效的数据处理导致性能下降

**常见情况**:
```python
# ❌ 错误示例 - 逐行迭代，极度低效
for idx, row in df.iterrows():
    df.loc[idx, 'new_col'] = process_row(row)

# ✓ 正确做法 - 向量化操作
df['new_col'] = df.apply(lambda row: process_row(row), axis=0)

# 最好的做法 - 完全向量化
df['new_col'] = process_vectorized(df)

# 性能对比
# iterrows: 10000 行需要 30 秒
# apply:    10000 行需要 3 秒
# 向量化:   10000 行需要 0.1 秒
```

**影响**: 运行时间增加 10-100 倍

**修复**: 使用 pandas 向量化操作、numpy、或 apply

---

#### 6️⃣ 代码质量 (Code Quality) - 🟡 **MEDIUM** ~ 🟢 **LOW**

**问题**: 代码难以维护、测试或理解

**常见情况**:
```python
# ❌ 错误示例 - 过于宽泛的异常
try:
    result = risky_operation()
except:
    print("Something went wrong")

# ✓ 正确做法
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except TimeoutError as e:
    logger.warning(f"Operation timeout: {e}")
    return None
```

**影响**: 难以调试，隐藏真实错误

---

## 修复优先级

### 优先级顺序

```
🔴 HIGH (严重)
├─ 数据泄露 (Data Leakage)
├─ 可复现性 (Reproducibility) 
├─ 数据完整性 (Data Integrity)
├─ 安全问题 (Security)
│
🟡 MEDIUM (中等)
├─ 性能问题 (Performance)
├─ 复杂度问题 (Complexity)
├─ 代码质量 (Code Quality)
│
🟢 LOW (低等)
├─ 代码风格 (Style)
├─ 未使用代码 (Unused Code)
```

### 修复策略

**冲刺 1 - 审查阶段** (2 小时)
- 修复所有 🔴 HIGH 问题
- 更新 severity_config.yaml 配置

**冲刺 2 - 改进阶段** (1 周)
- 修复所有 🟡 MEDIUM 问题
- 添加单元测试

**冲刺 3 - 优化阶段** (有时间)
- 修复 🟢 LOW 问题
- 代码风格统一

---

## 配置说明

### severity_config.yaml 结构

```yaml
default_severity: "medium"  # 未配置规则的默认级别

rules:
  DS_RANDOM_SEED:
    severity: "high"        # 严重级别
    description: "..."      # 问题描述
    category: "reproducibility"  # 分类
    impact: "..."          # 影响说明

categories:
  reproducibility:
    display_name: "可复现性 / Reproducibility"
    order: 1               # 显示优先级
```

### 使用配置文件

```python
from src.features.review.review_runner import _load_severity_config, _adjust_severity_by_config

# 加载配置
config = _load_severity_config("severity_config.yaml")

# 应用到 findings
adjusted_findings = _adjust_severity_by_config(findings, config)
```

### 在 config.yaml 中配置

```yaml
review:
  enable_ds_rules: true
  severity_config_path: "./severity_config.yaml"  # 配置文件路径
```

---

## 最佳实践

### ✅ 最佳实践清单

- [ ] **在模型前设置随机数种子**
  ```python
  import numpy as np
  import random
  random.seed(42)
  np.random.seed(42)
  ```

- [ ] **使用 Pipeline 处理数据**
  ```python
  from sklearn.pipeline import Pipeline
  pipeline = Pipeline([
      ('preprocessor', StandardScaler()),
      ('classifier', LogisticRegression())
  ])
  ```

- [ ] **数据分割时固定 random_state**
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  ```

- [ ] **记录所有超参数**
  ```python
  hyperparams = {
      'learning_rate': 0.01,
      'n_estimators': 100,
      'random_state': 42
  }
  model = RandomForestClassifier(**hyperparams)
  ```

- [ ] **使用具体的异常类型**
  ```python
  try:
      value = int(user_input)
  except ValueError:
      logger.error("Input must be integer")
  ```

- [ ] **避免硬编码敏感信息**
  ```python
  import os
  api_key = os.environ.get('API_KEY')
  ```

---

## 常见问题解答

### Q: 为什么 DS_RANDOM_SEED 是 HIGH？
**A**: 没有固定随机数种子意味着每次运行结果都不同，无法复现论文的实验结果，这对科研工作是致命的。

### Q: MEDIUM 问题可以忽略吗？
**A**: 不建议。虽然不如 HIGH 问题紧急，但会逐渐积累：
- 性能问题导致 10 倍速度下降
- 复杂度问题导致难以维护
- 应在本周内修复

### Q: 如何自定义严重级别？
**A**: 编辑 `severity_config.yaml`：
```yaml
rules:
  MY_CUSTOM_RULE:
    severity: "high"  # 或 "medium", "low"
    description: "..."
    category: "custom"
```

### Q: 工具如何识别规则？
**A**: 通过规则 ID 和前缀匹配：
```python
# 完全匹配
if "DS_RANDOM_SEED" in config['rules']:
    severity = config['rules']["DS_RANDOM_SEED"]['severity']

# 前缀匹配 (如 E101, E902)
if rule[0] in config['rules']:
    severity = config['rules'][rule[0]]['severity']
```

---

## 参考资源

### 数据科学最佳实践
- [scikit-learn 用户指南](https://scikit-learn.org/stable/user_guide.html)
- [Pandas 文档 - 避免 SettingWithCopyWarning](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html)

### 安全编程
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Bandit 文档](https://bandit.readthedocs.io/)

### Python 代码质量
- [PEP 8 - 风格指南](https://pep8.org/)
- [Ruff 规则](https://docs.astral.sh/ruff/rules/)

---

**最后更新**: 2026年1月
**维护者**: Code Assistant
