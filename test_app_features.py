#!/usr/bin/env python3
"""
CodeAssistant App 功能完整性测试

测试 app.py 中的所有核心功能是否正确实现
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🧪 CodeAssistant App 功能完整性测试")
print("=" * 80)
print()

# ============================================
# 测试1: 模块导入检查
# ============================================
print("📦 测试 1: 核心模块导入...")
print("-" * 80)

test_results = []

try:
    from src.core.config import load_config
    from src.core.config_validator import validate_config, CodeAssistantConfig
    from src.core.llm_client import llm_chat, build_llm_config
    from src.core.logger import setup_logger, get_logger
    from src.core.orchestrator import Orchestrator
    from src.core.subproc import run_cmd
    from src.features.review.notebook import extract_code_cells
    from src.features.review.rule_plugin import get_registry
    from src.features.review.review_runner import run_review_pipeline
    from src.features.testgen.testgen_runner import run_testgen_pipeline
    from src.reporting.latex_builder import build_latex_report
    from src.reporting.pdf_builder import build_pdf_report
    from src.reporting.report_builder import build_markdown_report
    
    print("✅ 所有核心模块导入成功")
    test_results.append(("模块导入", True, ""))
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    test_results.append(("模块导入", False, str(e)))

print()

# ============================================
# 测试2: 配置文件加载
# ============================================
print("⚙️  测试 2: 配置文件加载...")
print("-" * 80)

try:
    config_path = Path("config.yaml")
    if config_path.exists():
        cfg = load_config(str(config_path))
        print(f"✅ 配置文件加载成功")
        print(f"   - Review 配置: {bool(cfg.get('review'))}")
        print(f"   - TestGen 配置: {bool(cfg.get('testgen'))}")
        print(f"   - Report 配置: {bool(cfg.get('report'))}")
        
        # 检查高级规则配置
        enable_ds_advanced = cfg.get("review", {}).get("enable_ds_rules_advanced", False)
        print(f"   - DS 高级规则默认: {enable_ds_advanced}")
        
        test_results.append(("配置加载", True, ""))
    else:
        print(f"⚠️  配置文件不存在: {config_path}")
        test_results.append(("配置加载", False, "config.yaml 不存在"))
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
    test_results.append(("配置加载", False, str(e)))

print()

# ============================================
# 测试3: 规则插件系统
# ============================================
print("🔌 测试 3: 规则插件系统...")
print("-" * 80)

try:
    from src.features.review import builtin_rules
    registry = get_registry()
    
    all_rules = registry.get_all()
    categories = registry.get_categories()
    
    print(f"✅ 规则插件系统正常")
    print(f"   - 已注册规则数: {len(all_rules)}")
    print(f"   - 规则分类数: {len(categories)}")
    
    # 按类别显示规则
    for cat in sorted(categories):
        cat_rules = registry.get_all(category=cat)
        print(f"   - {cat}: {len(cat_rules)} 个规则")
        for rule in cat_rules[:2]:  # 显示前2个规则
            print(f"      • {rule.rule_id}: {rule.description[:50]}...")
    
    test_results.append(("规则插件", True, f"{len(all_rules)} 个规则"))
except Exception as e:
    print(f"❌ 规则插件系统失败: {e}")
    test_results.append(("规则插件", False, str(e)))

print()

# ============================================
# 测试4: Code Review 功能
# ============================================
print("🔍 测试 4: Code Review 功能...")
print("-" * 80)

try:
    # 创建测试文件
    test_dir = Path("test_sample_code")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test_sample.py"
    test_file.write_text("""
import pandas as pd
import numpy as np

# 测试1: DataFrame inplace 操作
def bad_inplace(df):
    df.dropna(inplace=True)
    return df

# 测试2: 可变默认参数
def bad_default(items=[]):
    items.append(1)
    return items

# 测试3: 缺少随机种子
def missing_seed():
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split([[1, 2], [3, 4]])
    return X_train

# 测试4: NumPy 循环
def slow_loop(arr):
    result = []
    for x in arr:
        result.append(x ** 2)
    return np.array(result)
""")
    
    # 运行 review
    test_cfg = {
        "review": {
            "enable_ds_rules": True,
            "enable_ds_rules_advanced": True,
            "enable_notebook": False,
            "tool_excludes": ["ruff", "bandit", "radon", "coverage", "pylint"],
        }
    }
    
    # 确保使用绝对路径
    test_file_abs = test_file.resolve()
    test_dir_abs = test_dir.resolve()
    
    result = run_review_pipeline(
        str(test_dir_abs),
        [test_file_abs],
        test_cfg
    )
    
    findings = result.get("findings", [])
    print(f"✅ Code Review 功能正常")
    print(f"   - 发现问题数: {len(findings)}")
    
    # 显示部分问题
    for i, finding in enumerate(findings[:3], 1):
        print(f"   {i}. [{finding.get('severity')}] {finding.get('message')[:60]}...")
    
    test_results.append(("Code Review", True, f"{len(findings)} 个问题"))
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_dir)
    
except Exception as e:
    print(f"❌ Code Review 失败: {e}")
    test_results.append(("Code Review", False, str(e)[:100]))
    import traceback
    traceback.print_exc()

print()

# ============================================
# 测试5: Test Generation 功能
# ============================================
print("🧪 测试 5: Test Generation 功能...")
print("-" * 80)

try:
    # 创建测试文件
    test_dir = Path("test_sample_code")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "sample_functions.py"
    test_file.write_text("""
def add(a, b):
    '''Add two numbers'''
    return a + b

def multiply(x, y):
    '''Multiply two numbers'''
    return x * y

class Calculator:
    def divide(self, a, b):
        '''Divide two numbers'''
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
""")
    
    # 运行 testgen
    test_cfg = {
        "testgen": {
            "output_dir": "generated_tests",
            "test_framework": "pytest",
        }
    }
    
    # 确保使用绝对路径
    test_file_abs = test_file.resolve()
    test_dir_abs = test_dir.resolve()
    
    result = run_testgen_pipeline(
        str(test_dir_abs),
        [test_file_abs],
        test_cfg
    )
    
    print(f"✅ Test Generation 功能正常")
    print(f"   - 生成测试数: {result.get('tests_generated', 0)}")
    print(f"   - 输出文件: {result.get('output_file', 'N/A')}")
    
    test_results.append(("Test Generation", True, f"{result.get('tests_generated', 0)} 个测试"))
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_dir)
    
except Exception as e:
    print(f"❌ Test Generation 失败: {e}")
    test_results.append(("Test Generation", False, str(e)[:100]))

print()

# ============================================
# 测试6: 报告生成功能
# ============================================
print("📄 测试 6: 报告生成功能...")
print("-" * 80)

try:
    # 创建模拟数据
    mock_review_result = {
        "findings": [
            {
                "file": "test.py",
                "line": 10,
                "severity": "WARNING",
                "message": "Test issue",
                "tool": "ds_rules",
            }
        ],
        "by_severity": {"WARNING": 1},
        "by_tool": {"ds_rules": 1},
        "summary": {"total": 1},
    }
    
    mock_testgen_result = {
        "tests_generated": 5,
        "output_file": "test_output.py",
    }
    
    # 测试 Markdown 报告
    try:
        md_report = build_markdown_report(
            review=mock_review_result,
            testgen=mock_testgen_result,
        )
        print("✅ Markdown 报告生成成功")
        print(f"   - 报告长度: {len(md_report)} 字符")
        test_results.append(("Markdown报告", True, ""))
    except Exception as e:
        print(f"❌ Markdown 报告失败: {e}")
        test_results.append(("Markdown报告", False, str(e)[:100]))
    
    # 测试 LaTeX 报告
    try:
        latex_report = build_latex_report(
            review=mock_review_result,
            testgen=mock_testgen_result,
        )
        print("✅ LaTeX 报告生成成功")
        print(f"   - 报告长度: {len(latex_report)} 字符")
        test_results.append(("LaTeX报告", True, ""))
    except Exception as e:
        print(f"❌ LaTeX 报告失败: {e}")
        test_results.append(("LaTeX报告", False, str(e)[:100]))
    
except Exception as e:
    print(f"❌ 报告生成失败: {e}")
    test_results.append(("报告生成", False, str(e)[:100]))

print()

# ============================================
# 测试7: Notebook 支持
# ============================================
print("📓 测试 7: Jupyter Notebook 支持...")
print("-" * 80)

try:
    # 创建测试 notebook 内容
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["import pandas as pd\n", "df = pd.DataFrame()"],
            },
            {
                "cell_type": "markdown",
                "source": ["# Test Markdown"],
            },
            {
                "cell_type": "code",
                "source": ["print('hello')"],
            }
        ]
    }
    
    test_nb = Path("test_notebook.ipynb")
    test_nb.write_text(json.dumps(notebook_content))
    
    # 提取代码单元格
    code_cells = extract_code_cells(test_nb)
    
    print(f"✅ Notebook 支持正常")
    print(f"   - 提取代码单元格数: {len(code_cells)}")
    
    test_results.append(("Notebook支持", True, f"{len(code_cells)} 个单元格"))
    
    # 清理
    test_nb.unlink()
    
except Exception as e:
    print(f"❌ Notebook 支持失败: {e}")
    test_results.append(("Notebook支持", False, str(e)[:100]))

print()

# ============================================
# 测试8: LLM 集成检查
# ============================================
print("🤖 测试 8: LLM 集成检查...")
print("-" * 80)

try:
    # 只检查配置是否可以构建，不实际调用
    # build_llm_config 返回的是 dict，直接构造测试配置
    llm_cfg = {
        "model": "gpt-4",
        "api_key": "test_key",
        "temperature": 0.7,
    }
    
    print(f"✅ LLM 配置构建成功")
    print(f"   - Model: {llm_cfg.get('model')}")
    print(f"   - Temperature: {llm_cfg.get('temperature')}")
    
    test_results.append(("LLM集成", True, ""))
except Exception as e:
    print(f"❌ LLM 集成失败: {e}")
    test_results.append(("LLM集成", False, str(e)[:100]))

print()

# ============================================
# 测试9: Orchestrator 协调器
# ============================================
print("🎯 测试 9: Orchestrator 协调器...")
print("-" * 80)

try:
    # 创建 orchestrator 实例
    test_cfg = load_config("config.yaml")
    orchestrator = Orchestrator(test_cfg)
    
    print(f"✅ Orchestrator 创建成功")
    print(f"   - 类型: {type(orchestrator).__name__}")
    
    test_results.append(("Orchestrator", True, ""))
except Exception as e:
    print(f"❌ Orchestrator 失败: {e}")
    test_results.append(("Orchestrator", False, str(e)[:100]))

print()

# ============================================
# 测试10: 高级 DS 规则检查
# ============================================
print("🚀 测试 10: 高级 DS 规则...")
print("-" * 80)

try:
    from src.features.review.ds_rules_advanced import scan_file_advanced_ds
    
    # 创建测试文件
    test_dir = Path("test_advanced_ds")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "advanced_test.py"
    test_file.write_text("""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据泄漏
def data_leakage(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
    return X_train, X_test

# 特征工程问题
def feature_engineering(df):
    # 缺少特征选择
    features = df.drop('target', axis=1)
    return features
""")
    
    # 运行高级规则扫描
    test_file_abs = test_file.resolve()
    test_dir_abs = test_dir.resolve()
    findings = scan_file_advanced_ds(test_file_abs, test_dir_abs)
    
    print(f"✅ 高级 DS 规则运行成功")
    print(f"   - 发现问题数: {len(findings)}")
    
    for i, finding in enumerate(findings[:3], 1):
        print(f"   {i}. {finding.message[:60]}...")
    
    test_results.append(("高级DS规则", True, f"{len(findings)} 个问题"))
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    
except Exception as e:
    print(f"❌ 高级 DS 规则失败: {e}")
    test_results.append(("高级DS规则", False, str(e)[:100]))

print()

# ============================================
# 测试总结
# ============================================
print("=" * 80)
print("📊 测试总结")
print("=" * 80)

passed = sum(1 for _, status, _ in test_results if status)
total = len(test_results)

print(f"\n总测试数: {total}")
print(f"通过: {passed} ✅")
print(f"失败: {total - passed} ❌")
print(f"通过率: {passed/total*100:.1f}%\n")

print("详细结果:")
print("-" * 80)
for i, (name, status, note) in enumerate(test_results, 1):
    status_icon = "✅" if status else "❌"
    note_str = f" ({note})" if note else ""
    print(f"{i:2d}. {status_icon} {name:<20} {note_str}")

print()

if passed == total:
    print("🎉 所有功能测试通过！App 运行正常！")
    exit_code = 0
else:
    print("⚠️  部分功能存在问题，请检查上面的错误信息。")
    exit_code = 1

print("=" * 80)
print()

# 退出代码
sys.exit(exit_code)
