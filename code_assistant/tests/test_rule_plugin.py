"""规则插件系统单元测试"""
import ast
import pytest
from src.features.review.rule_plugin import (
    Rule, RuleRegistry, DataScienceRule, get_registry, register_rule
)
from src.features.review.types import ReviewFinding


class SimpleTestRule(DataScienceRule):
    """简单的测试规则"""

    @property
    def rule_id(self) -> str:
        return "TEST_SIMPLE"

    @property
    def description(self) -> str:
        return "A simple test rule"

    @property
    def severity(self) -> str:
        return "low"

    def check(self, tree: ast.AST, source: str, rel_path: str) -> list:
        """检查是否包含 'TODO' 注释"""
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and "TODO" in node.value.value:
                    findings.append(
                        ReviewFinding(
                            tool="test",
                            rule=self.rule_id,
                            severity=self.severity,
                            message="Found TODO comment",
                            file=rel_path,
                            line=node.lineno,
                            col=node.col_offset,
                        )
                    )
        return findings


class TestRuleRegistry:
    """测试规则注册表"""

    def test_register_rule(self):
        """测试规则注册"""
        registry = RuleRegistry()
        rule = SimpleTestRule()
        
        registry.register(rule)
        assert registry.get("TEST_SIMPLE") is not None

    def test_duplicate_registration_raises(self):
        """重复注册应抛出错误"""
        registry = RuleRegistry()
        rule = SimpleTestRule()
        
        registry.register(rule)
        with pytest.raises(ValueError):
            registry.register(rule)

    def test_unregister_rule(self):
        """测试规则注销"""
        registry = RuleRegistry()
        rule = SimpleTestRule()
        
        registry.register(rule)
        registry.unregister("TEST_SIMPLE")
        
        assert registry.get("TEST_SIMPLE") is None

    def test_get_all_rules(self):
        """测试获取所有规则"""
        registry = RuleRegistry()
        rule1 = SimpleTestRule()
        rule2 = SimpleTestRule()
        rule2._rule_id = "TEST_RULE_2"
        
        registry.register(rule1)
        registry.register(rule2)
        
        assert len(registry.get_all()) == 2

    def test_get_rules_by_category(self):
        """测试按类别获取规则"""
        registry = RuleRegistry()
        rule = SimpleTestRule()
        
        registry.register(rule)
        rules = registry.get_all(category="data_science")
        
        assert len(rules) == 1
        assert rules[0].rule_id == "TEST_SIMPLE"

    def test_get_categories(self):
        """测试获取规则类别"""
        registry = RuleRegistry()
        rule = SimpleTestRule()
        
        registry.register(rule)
        categories = registry.get_categories()
        
        assert "data_science" in categories


class TestRuleExecution:
    """测试规则执行"""

    def test_rule_check_basic(self):
        """测试规则基本检查"""
        code = '''
"""TODO: Fix this"""
x = 1
'''
        tree = ast.parse(code)
        rule = SimpleTestRule()
        findings = rule.check(tree, code, "test.py")
        
        assert len(findings) > 0
        assert findings[0].rule == "TEST_SIMPLE"

    def test_registry_check(self):
        """测试注册表检查"""
        registry = RuleRegistry()
        rule = SimpleTestRule()
        registry.register(rule)
        
        code = '''
"""TODO: Implement this"""
pass
'''
        tree = ast.parse(code)
        findings = registry.check(tree, code, "test.py")
        
        assert len(findings) > 0

    def test_registry_check_with_filters(self):
        """测试带过滤的注册表检查"""
        registry = RuleRegistry()
        rule1 = SimpleTestRule()
        
        class AnotherTestRule(DataScienceRule):
            @property
            def rule_id(self) -> str:
                return "TEST_ANOTHER"
            
            @property
            def description(self) -> str:
                return "Another test"
            
            @property
            def severity(self) -> str:
                return "high"
            
            def check(self, tree, source, rel_path):
                return []
        
        rule2 = AnotherTestRule()
        registry.register(rule1)
        registry.register(rule2)
        
        code = "pass"
        tree = ast.parse(code)
        
        # 只执行 TEST_SIMPLE
        findings = registry.check(tree, code, "test.py", rule_ids=["TEST_SIMPLE"])
        assert findings is not None

    def test_exclude_rules(self):
        """测试规则排除"""
        registry = RuleRegistry()
        rule = SimpleTestRule()
        registry.register(rule)
        
        code = '''
"""TODO: Fix"""
pass
'''
        tree = ast.parse(code)
        findings = registry.check(tree, code, "test.py", exclude_rule_ids=["TEST_SIMPLE"])
        
        assert len(findings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
