"""统一日志系统模块

提供结构化日志、性能跟踪、错误报告等功能
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 基础信息
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加额外的上下文信息
        if record.exc_info:
            import traceback
            log_data["exception"] = traceback.format_exception(*record.exc_info)

        # 如果是异常，添加异常信息
        if record.exc_info:
            log_data["exception_type"] = record.exc_info[0].__name__

        # 添加额外的属性
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # ANSI 颜色代码
    COLORS = {
        "DEBUG": "\033[36m",      # 青色
        "INFO": "\033[32m",       # 绿色
        "WARNING": "\033[33m",    # 黄色
        "ERROR": "\033[31m",      # 红色
        "CRITICAL": "\033[35m",   # 品红
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """格式化带颜色的日志"""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    use_color: bool = True,
    use_json: bool = False,
) -> logging.Logger:
    """设置日志记录器
    
    Args:
        name: 记录器名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        use_color: 是否使用彩色输出（仅用于控制台）
        use_json: 是否使用 JSON 格式（仅用于文件）
        
    Returns:
        配置好的 Logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 移除已有处理器以避免重复
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_color:
        console_formatter = ColoredFormatter(
            fmt="%(levelname)s - %(name)s - %(message)s"
        )
    else:
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器（可选）
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)

        if use_json:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取已配置的记录器
    
    Args:
        name: 记录器名称（通常为模块名 __name__）
        
    Returns:
        Logger 对象
    """
    return logging.getLogger(name)


class PerformanceLogger:
    """性能追踪日志记录器"""

    def __init__(self, logger: logging.Logger):
        """初始化性能追踪器
        
        Args:
            logger: 日志记录器实例
        """
        self.logger = logger
        self.timings: dict = {}

    def start_timer(self, task_name: str):
        """启动计时器
        
        Args:
            task_name: 任务名称
        """
        import time
        self.timings[task_name] = time.time()
        self.logger.debug(f"开始任务: {task_name}")

    def end_timer(self, task_name: str, log_level: int = logging.INFO):
        """结束计时器并记录耗时
        
        Args:
            task_name: 任务名称
            log_level: 日志级别
        """
        import time
        if task_name not in self.timings:
            self.logger.warning(f"任务 '{task_name}' 未启动计时器")
            return

        elapsed = time.time() - self.timings[task_name]
        del self.timings[task_name]
        
        self.logger.log(
            log_level,
            f"任务完成: {task_name} (耗时 {elapsed:.2f}s)"
        )

    def log_stats(self, stats: dict, log_level: int = logging.INFO):
        """记录统计信息
        
        Args:
            stats: 统计字典
            log_level: 日志级别
        """
        stats_str = json.dumps(stats, ensure_ascii=False, indent=2)
        self.logger.log(log_level, f"统计信息:\n{stats_str}")


# 全局根记录器
root_logger = setup_logger("code-assistant", level=logging.INFO)
