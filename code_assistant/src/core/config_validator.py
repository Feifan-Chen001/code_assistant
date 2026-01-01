"""配置验证模块 - 使用 Pydantic 进行类型检查和验证"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class AssistantConfig(BaseModel):
    """助手主配置"""
    max_files: int = Field(2000, ge=1, le=100000, description="最大扫描文件数")
    include_globs: List[str] = Field(default=["**/*.py", "**/*.ipynb"], description="包含的文件 glob 模式")
    exclude_globs: List[str] = Field(
        default=["**/.venv/**", "**/venv/**", "**/__pycache__/**", "**/build/**"],
        description="排除的文件 glob 模式"
    )

    @validator('include_globs')
    def validate_include_globs(cls, v: List[str]) -> List[str]:
        """验证包含 glob 列表"""
        if not v:
            raise ValueError("include_globs 不能为空")
        return v


class ReviewConfig(BaseModel):
    """代码审查配置"""
    enable_ruff: bool = Field(True, description="启用 ruff linter")
    enable_mypy: bool = Field(False, description="启用 mypy 类型检查")
    enable_bandit: bool = Field(True, description="启用 bandit 安全检查")
    enable_pip_audit: bool = Field(True, description="启用 pip-audit 依赖检查")
    enable_radon: bool = Field(True, description="启用 radon 复杂度分析")
    enable_ds_rules: bool = Field(True, description="启用基础数据科学规则")
    enable_ds_rules_advanced: bool = Field(True, description="启用高级数据科学规则")
    enable_notebook: bool = Field(True, description="启用 Jupyter Notebook 支持")

    ruff_args: List[str] = Field(["check", "--format", "json"], description="ruff 命令行参数")
    mypy_args: List[str] = Field(["--show-error-codes", "--no-error-summary"], description="mypy 命令行参数")
    bandit_args: List[str] = Field(["-r", "-f", "json"], description="bandit 命令行参数")
    pip_audit_args: List[str] = Field(["-f", "json"], description="pip-audit 命令行参数")


class TestGenConfig(BaseModel):
    """测试生成配置"""
    output_dir: str = Field("generated_tests", description="测试输出目录")
    use_hypothesis: bool = Field(True, description="使用 Hypothesis 属性测试")
    max_functions: int = Field(200, ge=1, description="最多处理函数数")


class CoverageConfig(BaseModel):
    """覆盖率配置"""
    enable: bool = Field(True, description="启用覆盖率分析")
    pytest_args: List[str] = Field(["-q"], description="pytest 参数")


class CodeAssistantConfig(BaseModel):
    """完整的 CodeAssistant 配置"""
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    testgen: TestGenConfig = Field(default_factory=TestGenConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CodeAssistantConfig:
        """从字典创建配置对象"""
        return cls(
            assistant=AssistantConfig(**data.get("assistant", {})),
            review=ReviewConfig(**data.get("review", {})),
            testgen=TestGenConfig(**data.get("testgen", {})),
            coverage=CoverageConfig(**data.get("coverage", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict(by_alias=False)


def validate_config(cfg: Dict[str, Any]) -> CodeAssistantConfig:
    """验证配置字典
    
    Args:
        cfg: 配置字典
        
    Returns:
        验证后的 CodeAssistantConfig 对象
        
    Raises:
        ValueError: 配置验证失败
    """
    try:
        return CodeAssistantConfig.from_dict(cfg)
    except Exception as e:
        raise ValueError(f"配置验证失败: {str(e)}")
