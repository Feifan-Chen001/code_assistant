from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import sys
import yaml

from ...core.subproc import run_cmd
from ...core.logger import PerformanceLogger, get_logger
from .parsers import parse_ruff_json, parse_bandit_json, parse_pip_audit_json
from .ast_rules import scan_file_ast, scan_source_ast
from .ds_rules import scan_file_ds, scan_source_ds
from .ds_rules_advanced import scan_file_advanced_ds, scan_source_advanced_ds
from .notebook import extract_code_cells

logger = get_logger(__name__)

# 检测是否在 Streamlit 中运行
def _is_streamlit():
    return "streamlit" in sys.modules

def _load_severity_config(severity_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载严重级别配置文件。
    
    Args:
        severity_config_path: 配置文件的路径，如果为 None 则使用默认路径
    
    Returns:
        配置字典，包含规则及其严重级别
    """
    if severity_config_path is None:
        # 尝试从项目根目录加载
        default_path = Path(__file__).parent.parent.parent.parent / "severity_config.yaml"
        if default_path.exists():
            severity_config_path = str(default_path)
        else:
            logger.debug("未找到 severity_config.yaml，使用默认严重级别")
            return {}
    
    try:
        config_file = Path(severity_config_path)
        if not config_file.exists():
            logger.warning(f"严重级别配置文件不存在: {severity_config_path}")
            return {}
        
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        logger.debug(f"成功加载严重级别配置: {severity_config_path}")
        return config if config else {}
    
    except Exception as e:
        logger.warning(f"加载严重级别配置失败: {e}")
        return {}

def _adjust_severity_by_config(findings: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    根据配置调整 findings 的严重级别。
    
    Args:
        findings: 问题列表
        config: 严重级别配置
    
    Returns:
        调整后的 findings 列表
    """
    if not config or "rules" not in config:
        return findings
    
    rules_config = config.get("rules", {})
    default_severity = config.get("default_severity", "medium")
    
    adjusted_findings = []
    for finding in findings:
        finding_copy = finding.copy()
        rule = finding.get("rule", "")
        
        # 根据规则 ID 查找配置
        if rule in rules_config:
            finding_copy["severity"] = rules_config[rule].get("severity", default_severity)
        # 尝试从规则前缀查找（如 E, W, F 等）
        elif rule and len(rule) > 0:
            prefix = rule[0]
            if prefix in rules_config:
                finding_copy["severity"] = rules_config[prefix].get("severity", default_severity)
        
        adjusted_findings.append(finding_copy)
    
    return adjusted_findings

def run_review_pipeline(repo_path: str, files: List[Path], cfg: Dict[str, Any]) -> Dict[str, Any]:
    review_cfg = cfg.get("review", {})
    repo_root = Path(repo_path).resolve()
    
    perf = PerformanceLogger(logger)
    perf.start_timer("review_pipeline")
    
    logger.info(f"开始审查: {repo_root}")

    findings = []
    enable_ds = review_cfg.get("enable_ds_rules", True)
    enable_advanced_ds = review_cfg.get("enable_ds_rules_advanced", True)
    enable_notebook = review_cfg.get("enable_notebook", True)

    # 在 Streamlit 中禁用高级 DS 规则以提高性能（用户可通过侧边栏启用）
    if _is_streamlit() and enable_advanced_ds:
        logger.info("在 Streamlit 中检测到，默认禁用高级 DS 规则以提高速度（可在侧边栏启用）")
        enable_advanced_ds = review_cfg.get("force_enable_advanced_ds", False)

    file_count = 0
    files_list = list(files)
    total_files = len(files_list)
    
    if _is_streamlit():
        import streamlit as st
        progress_bar = st.progress(0)
    
    for idx, p in enumerate(files_list):
        file_count += 1
        
        # 更新进度条
        if _is_streamlit():
            progress_bar.progress(min((idx + 1) / total_files, 0.95))
        
        try:
            if p.suffix == ".ipynb":
                if not enable_notebook:
                    continue
                logger.debug(f"处理 Notebook: {p.name}")
                rel = str(p.relative_to(repo_root)).replace("\\", "/")
                for cidx, code in enumerate(extract_code_cells(p)):
                    cell_rel = f"{rel}#cell-{cidx}"
                    findings.extend([f.model_dump() for f in scan_source_ast(code, cell_rel)])
                    if enable_ds:
                        findings.extend([f.model_dump() for f in scan_source_ds(code, cell_rel)])
                    if enable_advanced_ds:
                        findings.extend([f.model_dump() for f in scan_source_advanced_ds(code, cell_rel)])
                continue
                
            findings.extend([f.model_dump() for f in scan_file_ast(p, repo_root)])
            if enable_ds and p.suffix == ".py":
                findings.extend([f.model_dump() for f in scan_file_ds(p, repo_root)])
            if enable_advanced_ds and p.suffix == ".py":
                findings.extend([f.model_dump() for f in scan_file_advanced_ds(p, repo_root)])
        except Exception as e:
            logger.warning(f"处理文件失败 {p}: {e}", exc_info=True)
            continue
    
    if _is_streamlit():
        progress_bar.progress(1.0)

    logger.info(f"已处理 {file_count} 个文件，共发现 {len(findings)} 个基础问题")

    if review_cfg.get("enable_ruff", True):
        perf.start_timer("ruff")
        try:
            args = ["ruff"] + review_cfg.get("ruff_args", ["check", "--format", "json"]) + [str(repo_root)]
            res = run_cmd(args, cwd=str(repo_root))
            ruff_findings = [f.model_dump() for f in parse_ruff_json(res["stdout"])]
            findings.extend(ruff_findings)
            logger.info(f"ruff: 发现 {len(ruff_findings)} 个问题")
            perf.end_timer("ruff")
        except Exception as e:
            logger.error(f"ruff 执行失败: {e}", exc_info=True)

    if review_cfg.get("enable_bandit", True):
        perf.start_timer("bandit")
        try:
            args = ["bandit"] + review_cfg.get("bandit_args", ["-r", "-f", "json"]) + [str(repo_root)]
            res = run_cmd(args, cwd=str(repo_root))
            bandit_findings = [f.model_dump() for f in parse_bandit_json(res["stdout"])]
            findings.extend(bandit_findings)
            logger.info(f"bandit: 发现 {len(bandit_findings)} 个问题")
            perf.end_timer("bandit")
        except Exception as e:
            logger.error(f"bandit 执行失败: {e}", exc_info=True)

    if review_cfg.get("enable_pip_audit", True):
        perf.start_timer("pip-audit")
        try:
            args = ["pip-audit"] + review_cfg.get("pip_audit_args", ["-f", "json"])
            res = run_cmd(args, cwd=str(repo_root))
            pip_findings = [f.model_dump() for f in parse_pip_audit_json(res["stdout"])]
            findings.extend(pip_findings)
            logger.info(f"pip-audit: 发现 {len(pip_findings)} 个问题")
            perf.end_timer("pip-audit")
        except Exception as e:
            logger.warning(f"pip-audit 执行失败（非致命): {e}")

    complexity = None
    if review_cfg.get("enable_radon", True):
        perf.start_timer("radon")
        try:
            res = run_cmd(["radon", "cc", "-s", "-a", str(repo_root)], cwd=str(repo_root))
            complexity = {"ok": res["ok"], "stdout": res["stdout"][:200000], "stderr": res["stderr"][:20000]}
            logger.info("radon: 复杂度分析完成")
            perf.end_timer("radon")
        except Exception as e:
            logger.warning(f"radon 执行失败: {e}")

    mypy_out = None
    if review_cfg.get("enable_mypy", False):
        perf.start_timer("mypy")
        try:
            args = ["mypy"] + review_cfg.get("mypy_args", []) + [str(repo_root)]
            res = run_cmd(args, cwd=str(repo_root))
            mypy_out = {"ok": res["ok"], "stdout": res["stdout"][:200000], "stderr": res["stderr"][:20000]}
            logger.info("mypy: 类型检查完成")
            perf.end_timer("mypy")
        except Exception as e:
            logger.warning(f"mypy 执行失败: {e}")

    perf.end_timer("review_pipeline")
    
    # 加载并应用严重级别配置
    severity_config_path = review_cfg.get("severity_config_path")
    severity_config = _load_severity_config(severity_config_path)
    if severity_config:
        findings = _adjust_severity_by_config(findings, severity_config)
        logger.info("已根据配置调整问题严重级别")
    
    result = {
        "repo": str(repo_root),
        "findings": findings,
        "tool_raw": {"complexity_radon": complexity, "mypy": mypy_out},
        "stats": {"total_findings": len(findings)},
    }
    
    logger.info(f"审查完成: {result['stats']}")
    return result
