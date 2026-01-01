"""高级数据科学规则检查模块

此模块提供更高级的数据科学特定规则，补充基础的 ds_rules.py
"""
from __future__ import annotations
import ast
from pathlib import Path
from typing import List, Set, Dict, Optional

from .types import ReviewFinding

TOOL = "ds-rule-advanced"

# 特征选择相关的函数/类
FEATURE_SELECTION_METHODS = {
    "SelectKBest", "SelectPercentile", "RFE", "SequentialFeatureSelector",
    "mutual_info_classif", "mutual_info_regression", "f_classif", "f_regression",
    "chi2"
}

# 采样方法（不平衡处理）
IMBALANCE_METHODS = {
    "SMOTE", "RandomOverSampler", "RandomUnderSampler", "ADASYN",
    "BorderlineSMOTE", "SVMSMOTE"
}

# 交叉验证方法
CV_METHODS = {
    "KFold", "StratifiedKFold", "LeaveOneOut", "ShuffleSplit",
    "cross_val_score", "cross_validate", "cross_val_predict"
}


def scan_file_advanced_ds(path: Path, repo_root: Path) -> List[ReviewFinding]:
    """扫描单个文件的高级 DS 规则"""
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    rel = str(path.relative_to(repo_root)).replace("\\", "/")
    return scan_source_advanced_ds(src, rel)


def scan_source_advanced_ds(source: str, rel_path: str) -> List[ReviewFinding]:
    """扫描源代码的高级 DS 规则（快速返回如果源代码不含 ML 库）"""
    # 快速检查：如果源代码不包含任何 ML 库，直接返回空
    ml_keywords = {
        "sklearn", "pandas", "numpy", "cv_", "Pipeline", "SMOTE", "cross_val",
        "StratifiedKFold", "SelectKBest", "feature_selection", "imbalance_learn"
    }
    has_ml_imports = any(kw in source for kw in ml_keywords)
    if not has_ml_imports:
        return []
    
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    visitor = _AdvancedDSVisitor(rel_path)
    visitor.visit(tree)
    return visitor.finalize()


class _AdvancedDSVisitor(ast.NodeVisitor):
    """高级数据科学规则访问者"""

    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.findings: List[ReviewFinding] = []
        
        # 规则1：特征选择后未使用嵌套 CV
        self.feature_selection_used = False
        self.cv_used = False
        self.feature_selection_line: Optional[int] = None
        
        # 规则2：采样方法未放在 Pipeline 中
        self.imbalance_methods_used: List[tuple] = []  # (方法名, 行号)
        self.pipeline_used = False
        
        # 规则3：不平衡数据处理缺失
        self.class_weight_used = False
        self.stratified_cv_used = False
        self.imbalance_handler_used = False
        self.has_fit_method = False
        
        # 规则4：特征缩放不一致
        self.scaled_features: Set[str] = set()
        self.feature_usage: List[int] = []
        self.scaling_call_lines: List[int] = []
        
        # 规则5：模型评估方法不足
        self.evaluation_metrics: Set[str] = set()

    def _add(self, rule: str, severity: str, message: str, node: Optional[ast.AST]):
        """添加发现"""
        self.findings.append(
            ReviewFinding(
                tool=TOOL,
                rule=rule,
                severity=severity,
                message=message,
                file=self.rel_path,
                line=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
            )
        )

    def _call_name(self, node: ast.AST) -> str:
        """获取调用的函数/类名"""
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return ""

    def _attr_chain(self, node: ast.AST) -> str:
        """获取完整的属性链"""
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    def visit_Call(self, node: ast.Call):
        """访问函数调用节点"""
        name = self._call_name(node.func)
        chain = self._attr_chain(node.func)

        # 检测特征选择
        if name in FEATURE_SELECTION_METHODS:
            self.feature_selection_used = True
            self.feature_selection_line = node.lineno

        # 检测 Pipeline
        if name in {"Pipeline", "make_pipeline"} or chain.endswith(".Pipeline"):
            self.pipeline_used = True

        # 检测 CV 方法
        if name in CV_METHODS:
            self.cv_used = True
            # 检查是否使用了分层 CV
            if name.startswith("Stratified"):
                self.stratified_cv_used = True

        # 检测不平衡处理
        if name in IMBALANCE_METHODS:
            self.imbalance_methods_used.append((name, node.lineno))

        # 检测评估指标
        if name in {"accuracy_score", "precision_score", "recall_score", "f1_score",
                    "roc_auc_score", "confusion_matrix", "classification_report",
                    "mean_squared_error", "mean_absolute_error", "r2_score"}:
            self.evaluation_metrics.add(name)

        # 检测 class_weight
        if self._has_kw(node, "class_weight"):
            self.class_weight_used = True

        # 检测 fit 方法调用（标记模型训练）
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            self.has_fit_method = True

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """访问赋值节点"""
        if isinstance(node.value, ast.Call):
            name = self._call_name(node.value.func)
            if name in IMBALANCE_METHODS:
                self.imbalance_handler_used = True

        self.generic_visit(node)

    def _has_kw(self, node: ast.Call, name: str) -> bool:
        """检查调用是否有指定的关键字参数"""
        return any(kw.arg == name for kw in node.keywords if kw.arg is not None)

    def finalize(self) -> List[ReviewFinding]:
        """完成检查并返回所有发现"""
        
        # 规则1：特征选择后应使用嵌套 CV 以避免过拟合
        if self.feature_selection_used and not self.cv_used:
            self._add(
                "DS_FEATURE_SELECTION_NO_NESTED_CV",
                "medium",
                "Feature selection detected without nested cross-validation; this may lead to optimistic bias.",
                _DummyNode(self.feature_selection_line),
            )

        # 规则2：采样方法应在 Pipeline 中使用
        if self.imbalance_methods_used and not self.pipeline_used:
            for method, line in self.imbalance_methods_used:
                self._add(
                    "DS_IMBALANCE_NOT_IN_PIPELINE",
                    "high",
                    f"Imbalance handling '{method}' used outside Pipeline; this may cause data leakage.",
                    _DummyNode(line),
                )

        # 规则3：检测不平衡数据处理缺失
        if self.has_fit_method and not (self.class_weight_used or self.imbalance_handler_used or self.stratified_cv_used):
            self._add(
                "DS_IMBALANCE_UNHANDLED",
                "low",
                "Model training detected without explicit imbalance handling; consider class_weight, SMOTE, or stratified CV.",
                _DummyNode(1),  # 默认行号
            )

        # 规则4：评估指标不足
        if self.has_fit_method and len(self.evaluation_metrics) < 2:
            msg = (
                "Limited evaluation metrics detected; "
                "use multiple metrics (accuracy, precision, recall, F1, ROC-AUC) for comprehensive assessment."
            )
            self._add("DS_EVALUATION_INCOMPLETE", "low", msg, _DummyNode(1))

        return self.findings


class _DummyNode:
    """虚拟 AST 节点用于报告"""
    def __init__(self, lineno: Optional[int]):
        self.lineno = lineno
        self.col_offset = None
