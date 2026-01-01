"""配置验证单元测试"""
import pytest
from pydantic import ValidationError
from src.core.config_validator import (
    AssistantConfig,
    ReviewConfig,
    TestGenConfig,
    CoverageConfig,
    CodeAssistantConfig,
    validate_config,
)


class TestAssistantConfig:
    """测试助手配置"""

    def test_valid_assistant_config(self):
        """测试有效的助手配置"""
        cfg = AssistantConfig(
            max_files=2000,
            include_globs=["**/*.py"],
            exclude_globs=["**/.venv/**"]
        )
        assert cfg.max_files == 2000
        assert "**/*.py" in cfg.include_globs

    def test_invalid_max_files(self):
        """max_files 必须为正数"""
        with pytest.raises(ValidationError):
            AssistantConfig(max_files=0)

    def test_empty_include_globs_invalid(self):
        """include_globs 不能为空"""
        with pytest.raises(ValidationError):
            AssistantConfig(include_globs=[])


class TestReviewConfig:
    """测试审查配置"""

    def test_valid_review_config(self):
        """测试有效的审查配置"""
        cfg = ReviewConfig(
            enable_ruff=True,
            enable_ds_rules=True,
            enable_notebook=True
        )
        assert cfg.enable_ruff is True
        assert cfg.enable_ds_rules is True

    def test_boolean_flags(self):
        """所有 enable_* 字段应为布尔值"""
        cfg = ReviewConfig()
        assert isinstance(cfg.enable_ruff, bool)
        assert isinstance(cfg.enable_mypy, bool)
        assert isinstance(cfg.enable_bandit, bool)

    def test_command_args_are_lists(self):
        """命令参数应为列表"""
        cfg = ReviewConfig()
        assert isinstance(cfg.ruff_args, list)
        assert isinstance(cfg.mypy_args, list)


class TestTestGenConfig:
    """测试 TestGen 配置"""

    def test_valid_testgen_config(self):
        """测试有效的测试生成配置"""
        cfg = TestGenConfig(
            output_dir="generated_tests",
            use_hypothesis=True,
            max_functions=200
        )
        assert cfg.output_dir == "generated_tests"
        assert cfg.use_hypothesis is True

    def test_max_functions_positive(self):
        """max_functions 必须为正数"""
        with pytest.raises(ValidationError):
            TestGenConfig(max_functions=0)


class TestCodeAssistantConfig:
    """测试完整配置"""

    def test_complete_config(self):
        """测试完整配置对象"""
        cfg = CodeAssistantConfig()
        assert cfg.assistant is not None
        assert cfg.review is not None
        assert cfg.testgen is not None
        assert cfg.coverage is not None

    def test_from_dict(self):
        """从字典创建配置"""
        data = {
            "assistant": {
                "max_files": 1000,
                "include_globs": ["**/*.py"]
            },
            "review": {
                "enable_ruff": True,
                "enable_ds_rules": True
            }
        }
        cfg = CodeAssistantConfig.from_dict(data)
        assert cfg.assistant.max_files == 1000
        assert cfg.review.enable_ruff is True

    def test_to_dict(self):
        """转换为字典"""
        cfg = CodeAssistantConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "assistant" in d
        assert "review" in d


class TestValidateConfig:
    """测试配置验证函数"""

    def test_validate_valid_config(self):
        """验证有效配置"""
        data = {
            "assistant": {"max_files": 2000},
            "review": {"enable_ruff": True},
        }
        cfg = validate_config(data)
        assert isinstance(cfg, CodeAssistantConfig)

    def test_validate_invalid_config(self):
        """验证无效配置应抛出错误"""
        data = {
            "assistant": {"max_files": -1},  # 无效
        }
        with pytest.raises(ValueError):
            validate_config(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
