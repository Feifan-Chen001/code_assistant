"""规则插件系统 - 使规则可扩展和模块化

这个模块提供了一个灵活的规则系统，允许用户自定义和扩展规则检查
"""
from __future__ import annotations
import ast
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional, Set
from pathlib import Path

from .types import ReviewFinding

# 延迟导入 logger，避免循环依赖
logger = None

def get_logger_instance():
    global logger
    if logger is None:
        try:
            from ...core.logger import get_logger
            logger = get_logger(__name__)
        except (ImportError, ValueError):
            import logging
            logger = logging.getLogger(__name__)
    return logger


class Rule(ABC):
    """所有规则的抽象基类"""

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """规则唯一标识符（例如: 'DS_RANDOM_SEED'）"""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """规则类别（例如: 'reproducibility', 'security', 'performance'）"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """规则描述"""
        pass

    @property
    @abstractmethod
    def severity(self) -> str:
        """默认严重性等级: 'low', 'medium', 'high'"""
        pass

    @abstractmethod
    def check(self, tree: ast.AST, source: str, rel_path: str) -> List[ReviewFinding]:
        """执行规则检查
        
        Args:
            tree: AST 树
            source: 源代码字符串
            rel_path: 相对文件路径
            
        Returns:
            发现列表
        """
        pass


class RuleRegistry:
    """规则注册表 - 集中管理所有规则"""

    def __init__(self):
        """初始化规则注册表"""
        self._rules: Dict[str, Rule] = {}
        self._rules_by_category: Dict[str, List[Rule]] = {}

    def register(self, rule: Rule) -> None:
        """注册一个规则
        
        Args:
            rule: Rule 实例
            
        Raises:
            ValueError: 如果规则 ID 已被注册
        """
        if rule.rule_id in self._rules:
            raise ValueError(f"规则 '{rule.rule_id}' 已被注册")

        self._rules[rule.rule_id] = rule
        
        # 按类别索引
        if rule.category not in self._rules_by_category:
            self._rules_by_category[rule.category] = []
        self._rules_by_category[rule.category].append(rule)
        
        get_logger_instance().debug(f"已注册规则: {rule.rule_id} ({rule.category})")

    def unregister(self, rule_id: str) -> None:
        """注销规则
        
        Args:
            rule_id: 规则 ID
        """
        if rule_id not in self._rules:
            get_logger_instance().warning(f"规则 '{rule_id}' 未找到")
            return

        rule = self._rules.pop(rule_id)
        
        # 从类别索引中移除
        if rule.category in self._rules_by_category:
            self._rules_by_category[rule.category] = [
                r for r in self._rules_by_category[rule.category]
                if r.rule_id != rule_id
            ]

    def get(self, rule_id: str) -> Optional[Rule]:
        """获取规则
        
        Args:
            rule_id: 规则 ID
            
        Returns:
            Rule 实例或 None
        """
        return self._rules.get(rule_id)

    def get_all(self, category: Optional[str] = None) -> List[Rule]:
        """获取所有规则
        
        Args:
            category: 可选的类别过滤
            
        Returns:
            规则列表
        """
        if category:
            return self._rules_by_category.get(category, [])
        return list(self._rules.values())

    def get_categories(self) -> Set[str]:
        """获取所有规则类别
        
        Returns:
            类别集合
        """
        return set(self._rules_by_category.keys())

    def check(
        self,
        tree: ast.AST,
        source: str,
        rel_path: str,
        rule_ids: Optional[List[str]] = None,
        exclude_rule_ids: Optional[List[str]] = None,
    ) -> List[ReviewFinding]:
        """使用指定规则进行检查
        
        Args:
            tree: AST 树
            source: 源代码
            rel_path: 相对文件路径
            rule_ids: 要使用的规则 ID 列表（None 表示全部）
            exclude_rule_ids: 要排除的规则 ID 列表
            
        Returns:
            所有发现
        """
        rules = self.get_all()
        
        # 过滤规则
        if rule_ids:
            rules = [r for r in rules if r.rule_id in rule_ids]
        
        if exclude_rule_ids:
            rules = [r for r in rules if r.rule_id not in exclude_rule_ids]

        findings = []
        for rule in rules:
            try:
                findings.extend(rule.check(tree, source, rel_path))
            except Exception as e:
                get_logger_instance().error(f"规则 '{rule.rule_id}' 执行失败: {e}", exc_info=True)
                continue

        return findings


# 全局规则注册表
_global_registry: Optional[RuleRegistry] = None


def get_registry() -> RuleRegistry:
    """获取全局规则注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = RuleRegistry()
    return _global_registry


def register_rule(rule: Rule) -> None:
    """向全局注册表注册规则"""
    get_registry().register(rule)


class DataScienceRule(Rule):
    """数据科学规则的基类"""

    @property
    def category(self) -> str:
        return "data_science"


class SecurityRule(Rule):
    """安全规则的基类"""

    @property
    def category(self) -> str:
        return "security"


class PerformanceRule(Rule):
    """性能规则的基类"""

    @property
    def category(self) -> str:
        return "performance"


class StyleRule(Rule):
    """代码风格规则的基类"""

    @property
    def category(self) -> str:
        return "style"
