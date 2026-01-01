"""规则插件系统示例和内置规则"""
import ast
from typing import List

from .types import ReviewFinding
from .rule_plugin import (
    DataScienceRule, SecurityRule, PerformanceRule,
    register_rule, get_registry, get_logger_instance
)


class MutableDefaultArgumentRule(DataScienceRule):
    """检测可变默认参数"""

    @property
    def rule_id(self) -> str:
        return "PY_MUTABLE_DEFAULT_ARG"

    @property
    def description(self) -> str:
        return "可变默认参数会被所有函数调用共享，可能导致意外行为"

    @property
    def severity(self) -> str:
        return "medium"

    def check(self, tree: ast.AST, source: str, rel_path: str) -> List[ReviewFinding]:
        """检查可变默认参数"""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults:
                    # 检查是否为列表、字典或集合
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append(
                            ReviewFinding(
                                tool="rule-plugin",
                                rule=self.rule_id,
                                severity=self.severity,
                                message=f"函数 '{node.name}' 使用了可变默认参数; 使用 None 代替",
                                file=rel_path,
                                line=default.lineno,
                                col=default.col_offset,
                            )
                        )

        return findings


class GlobalVariableRule(DataScienceRule):
    """检测全局变量污染"""

    @property
    def rule_id(self) -> str:
        return "PY_GLOBAL_VARIABLE"

    @property
    def description(self) -> str:
        return "过度使用全局变量会使代码难以测试和维护"

    @property
    def severity(self) -> str:
        return "low"

    def check(self, tree: ast.AST, source: str, rel_path: str) -> List[ReviewFinding]:
        """检查全局变量使用"""
        findings = []
        
        # 跟踪全局变量
        global_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        global_vars.add(target.id)

        # 检查 global 语句的使用
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    if name in global_vars:
                        findings.append(
                            ReviewFinding(
                                tool="rule-plugin",
                                rule=self.rule_id,
                                severity=self.severity,
                                message=f"检测到全局变量 '{name}' 的使用; 考虑使用参数传递代替",
                                file=rel_path,
                                line=node.lineno,
                                col=node.col_offset,
                            )
                        )

        return findings


class ResourceLeakRule(SecurityRule):
    """检测资源泄漏"""

    @property
    def rule_id(self) -> str:
        return "PY_RESOURCE_LEAK"

    @property
    def description(self) -> str:
        return "文件和数据库连接应使用 with 语句确保正确关闭"

    @property
    def severity(self) -> str:
        return "high"

    def check(self, tree: ast.AST, source: str, rel_path: str) -> List[ReviewFinding]:
        """检查资源泄漏"""
        findings = []

        for node in ast.walk(tree):
            # 检查 open() 调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    # 检查是否在 with 语句中
                    parent_is_with = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.With):
                            for item in parent.items:
                                if item.context_expr == node:
                                    parent_is_with = True
                                    break
                    
                    if not parent_is_with:
                        findings.append(
                            ReviewFinding(
                                tool="rule-plugin",
                                rule=self.rule_id,
                                severity=self.severity,
                                message="open() 调用应在 with 语句中使用以确保正确关闭",
                                file=rel_path,
                                line=node.lineno,
                                col=node.col_offset,
                            )
                        )

        return findings


class LoopInvariantRule(PerformanceRule):
    """检测循环内重复计算"""

    @property
    def rule_id(self) -> str:
        return "PY_LOOP_INVARIANT"

    @property
    def description(self) -> str:
        return "循环内的不变表达式应提取到循环外"

    @property
    def severity(self) -> str:
        return "low"

    def check(self, tree: ast.AST, source: str, rel_path: str) -> List[ReviewFinding]:
        """检查循环内的重复计算"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # 检查循环体中的 len() 调用
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == "len":
                            findings.append(
                                ReviewFinding(
                                    tool="rule-plugin",
                                    rule=self.rule_id,
                                    severity=self.severity,
                                    message="考虑在循环外计算长度，避免每次迭代都重新计算",
                                    file=rel_path,
                                    line=child.lineno,
                                    col=child.col_offset,
                                )
                            )

        return findings


# 自动注册内置规则
def register_builtin_rules():
    """注册所有内置规则"""
    registry = get_registry()
    
    builtin_rules = [
        MutableDefaultArgumentRule(),
        GlobalVariableRule(),
        ResourceLeakRule(),
        LoopInvariantRule(),
    ]
    
    for rule in builtin_rules:
        try:
            registry.register(rule)
        except ValueError:
            # 规则已注册，忽略
            pass


# 模块初始化时自动注册规则
register_builtin_rules()
