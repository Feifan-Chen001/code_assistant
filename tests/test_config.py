"""配置模块单元测试"""
import pytest
from pathlib import Path
from src.core.config import load_config


class TestConfigLoading:
    """测试配置文件加载"""

    def test_load_default_config(self):
        """测试加载默认配置"""
        cfg = load_config("config.yaml")
        assert cfg is not None
        assert "assistant" in cfg
        assert "review" in cfg

    def test_review_config_structure(self):
        """验证审查配置结构"""
        cfg = load_config("config.yaml")
        review_cfg = cfg.get("review", {})
        
        # 验证基本配置字段
        assert "enable_ruff" in review_cfg
        assert "enable_ds_rules" in review_cfg
        assert "enable_notebook" in review_cfg

    def test_assistant_config_structure(self):
        """验证助手配置结构"""
        cfg = load_config("config.yaml")
        asst_cfg = cfg.get("assistant", {})
        
        assert "max_files" in asst_cfg
        assert "include_globs" in asst_cfg
        assert "exclude_globs" in asst_cfg


class TestConfigValidation:
    """测试配置值验证"""

    def test_max_files_positive(self):
        """max_files 应为正数"""
        cfg = load_config("config.yaml")
        max_files = cfg.get("assistant", {}).get("max_files", 0)
        assert max_files > 0

    def test_globs_are_lists(self):
        """include/exclude_globs 应为列表"""
        cfg = load_config("config.yaml")
        asst_cfg = cfg.get("assistant", {})
        assert isinstance(asst_cfg.get("include_globs"), list)
        assert isinstance(asst_cfg.get("exclude_globs"), list)

    def test_boolean_flags(self):
        """所有 enable_* 标志应为布尔值"""
        cfg = load_config("config.yaml")
        review_cfg = cfg.get("review", {})
        
        for key in review_cfg:
            if key.startswith("enable_"):
                assert isinstance(review_cfg[key], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
