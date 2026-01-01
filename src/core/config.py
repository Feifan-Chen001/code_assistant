from __future__ import annotations
from typing import Any, Dict
import yaml
import logging
from .config_validator import validate_config, CodeAssistantConfig

logger = logging.getLogger(__name__)


def load_config(path: str, validate: bool = True) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        path: 配置文件路径
        validate: 是否验证配置（推荐开启）
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 配置验证失败
        yaml.YAMLError: YAML 解析错误
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML 解析错误: {e}")
        raise

    if validate:
        try:
            validated_cfg = validate_config(cfg)
            logger.debug(f"配置验证成功: {path}")
            return validated_cfg.to_dict()
        except ValueError as e:
            logger.error(f"配置验证失败: {e}")
            raise

    return cfg


def load_config_strict(path: str) -> CodeAssistantConfig:
    """严格模式加载配置 - 返回 Pydantic 验证对象
    
    Args:
        path: 配置文件路径
        
    Returns:
        CodeAssistantConfig 对象
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 配置验证失败
    """
    cfg_dict = load_config(path, validate=False)
    return validate_config(cfg_dict)

