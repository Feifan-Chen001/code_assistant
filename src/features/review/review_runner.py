from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import ast
import logging
import sys
import yaml
import re

from ...core.subproc import run_cmd
from ...core.logger import PerformanceLogger, get_logger
from .parsers import parse_ruff_json, parse_bandit_json, parse_pip_audit_json
from .ast_rules import scan_file_ast, scan_source_ast
from .ds_rules import scan_file_ds, scan_source_ds
from .ds_rules_advanced import scan_file_advanced_ds, scan_source_advanced_ds
from .notebook import extract_code_cells
from .rule_plugin import get_registry
from . import builtin_rules  # 确保内置规则被注册

logger = get_logger(__name__)

def _strip_ipython_magics(source: str) -> str:
    """Remove IPython magics/shell escapes so AST-based rules won't crash on notebooks."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%%") or stripped.startswith("%") or stripped.startswith("!"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _is_streamlit():
    return "streamlit" in sys.modules

def _tool_excludes(review_cfg: Dict[str, Any]) -> List[str]:
    raw = review_cfg.get("tool_excludes") or []
    out: List[str] = []
    for item in raw:
        val = str(item).strip()
        if val:
            out.append(val)
    return out

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

    tool_excludes = _tool_excludes(review_cfg)
    testgen_out = (cfg.get("testgen", {}) or {}).get("output_dir")
    if testgen_out:
        out_path = Path(testgen_out)
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()
        else:
            out_path = out_path.resolve()
        try:
            rel = out_path.relative_to(repo_root)
        except ValueError:
            rel = None
        if rel:
            rel_posix = rel.as_posix().rstrip("/")
            if rel_posix and rel_posix not in tool_excludes:
                tool_excludes.append(rel_posix)

    # Optional Streamlit override for advanced DS rules.
    if _is_streamlit():
        forced_advanced = review_cfg.get("force_enable_advanced_ds")
        if forced_advanced is not None:
            enable_advanced_ds = bool(forced_advanced)
            logger.info("Streamlit override: force_enable_advanced_ds=%s", enable_advanced_ds)

    file_count = 0
    files_list = list(files)
    total_files = len(files_list)
    
    tool_candidates = [
        ("ruff", review_cfg.get("enable_ruff", True)),
        ("bandit", review_cfg.get("enable_bandit", True)),
        ("pip-audit", review_cfg.get("enable_pip_audit", True)),
        ("radon", review_cfg.get("enable_radon", True)),
        ("mypy", review_cfg.get("enable_mypy", False)),
    ]
    tool_names = [name for name, enabled in tool_candidates if enabled]
    total_tools = len(tool_names)
    
    file_weight = 1.0
    tool_weight = 0.0
    if total_files and total_tools:
        file_weight = 0.7
        tool_weight = 0.3
    elif total_tools and not total_files:
        file_weight = 0.0
        tool_weight = 1.0
    
    tool_index = 0
    progress_bar = None  # 先初始化为None
    
    if _is_streamlit():
        import streamlit as st
        progress_bar = st.progress(0)
    
    def _update_progress(value: float):
        if _is_streamlit() and progress_bar is not None:
            progress_bar.progress(min(max(value, 0.0), 1.0))
    
    def _finish_tool():
        nonlocal tool_index
        if total_tools <= 0:
            return
        tool_index += 1
        _update_progress(file_weight + tool_weight * (tool_index / total_tools))
    
    _update_progress(0.0)
    
    for idx, p in enumerate(files_list):
        file_count += 1
        
        # 更新进度条
        if _is_streamlit():
            if total_files:
                _update_progress(file_weight * ((idx + 1) / total_files))
        
        try:
            if p.suffix == ".ipynb":
                if not enable_notebook:
                    continue
                logger.debug(f"处理 Notebook: {p.name}")
                rel = str(p.relative_to(repo_root)).replace("\\", "/")
                for cell_idx, cell_code in extract_code_cells(p):
                    cell_rel = f"{rel}#cell-{cell_idx}"
                    cleaned = _strip_ipython_magics(cell_code)
                    if not cleaned.strip():
                        continue
                    findings.extend([f.model_dump() for f in scan_source_ast(cleaned, cell_rel)])
                    if enable_ds:
                        findings.extend([f.model_dump() for f in scan_source_ds(cleaned, cell_rel)])
                    if enable_advanced_ds:
                        findings.extend([f.model_dump() for f in scan_source_advanced_ds(cleaned, cell_rel)])
                    
                    # 调用规则插件系统
                    try:
                        tree = ast.parse(cleaned)
                        plugin_findings = get_registry().check(tree, cleaned, cell_rel)
                        for f in plugin_findings:
                            try:
                                findings.append(f.model_dump())
                            except Exception as dump_e:
                                logger.error(f"无法序列化规则检查结果 {cell_rel}: {dump_e}", exc_info=True)
                    except SyntaxError as se:
                        logger.debug(f"Notebook单元格语法错误 {cell_rel}, 跳过规则检查: {se}")
                    except Exception as e:
                        logger.warning(f"Notebook规则插件检查失败 {cell_rel}: {e}", exc_info=True)
                continue
                
            findings.extend([f.model_dump() for f in scan_file_ast(p, repo_root)])
            if enable_ds and p.suffix == ".py":
                findings.extend([f.model_dump() for f in scan_file_ds(p, repo_root)])
            if enable_advanced_ds and p.suffix == ".py":
                findings.extend([f.model_dump() for f in scan_file_advanced_ds(p, repo_root)])
            
            # 调用规则插件系统
            if p.suffix == ".py":
                try:
                    src = p.read_text(encoding="utf-8", errors="ignore")
                    tree = ast.parse(src)
                    rel = str(p.relative_to(repo_root)).replace("\\", "/")
                    plugin_findings = get_registry().check(tree, src, rel)
                    if plugin_findings:
                        logger.info(f"规则插件在 {rel} 中发现 {len(plugin_findings)} 个问题")
                    for f in plugin_findings:
                        try:
                            findings.append(f.model_dump())
                        except Exception as dump_e:
                            logger.error(f"无法序列化规则检查结果 {rel}: {dump_e}", exc_info=True)
                except SyntaxError as se:
                    logger.debug(f"文件语法错误 {p}, 跳过规则检查: {se}")
                except Exception as e:
                    logger.warning(f"规则插件检查失败 {p}: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"处理文件失败 {p}: {e}", exc_info=True)
            continue
    
    if _is_streamlit():
        _update_progress(file_weight)

    logger.info(f"已处理 {file_count} 个文件，共发现 {len(findings)} 个基础问题")

    if review_cfg.get("enable_ruff", True):
        perf.start_timer("ruff")
        try:
            args = ["ruff"] + review_cfg.get("ruff_args", ["check", "--format", "json"])
            if tool_excludes and not any(a in ("--exclude", "--extend-exclude") for a in args):
                args += ["--exclude", ",".join(tool_excludes)]
            args += [str(repo_root)]
            res = run_cmd(args, cwd=str(repo_root))
            ruff_findings = [f.model_dump() for f in parse_ruff_json(res["stdout"])]
            findings.extend(ruff_findings)
            logger.info(f"ruff: 发现 {len(ruff_findings)} 个问题")
            perf.end_timer("ruff")
        except Exception as e:
            logger.error(f"ruff 执行失败: {e}", exc_info=True)

        _finish_tool()
    if review_cfg.get("enable_bandit", True):
        perf.start_timer("bandit")
        try:
            args = ["bandit"] + review_cfg.get("bandit_args", ["-r", "-f", "json"])
            if tool_excludes and not any(a in ("-x", "--exclude") for a in args):
                args += ["-x", ",".join(tool_excludes)]
            args += [str(repo_root)]
            res = run_cmd(args, cwd=str(repo_root))
            bandit_findings = [f.model_dump() for f in parse_bandit_json(res["stdout"])]
            findings.extend(bandit_findings)
            logger.info(f"bandit: 发现 {len(bandit_findings)} 个问题")
            perf.end_timer("bandit")
        except Exception as e:
            logger.error(f"bandit 执行失败: {e}", exc_info=True)

        _finish_tool()
    if review_cfg.get("enable_pip_audit", True):
        perf.start_timer("pip-audit")
        try:
            args = ["pip-audit"] + review_cfg.get("pip_audit_args", ["-f", "json"])
            has_req_arg = any(a in ("-r", "--requirement") for a in args)
            req_files: List[Path] = []
            cfg_reqs = review_cfg.get("pip_audit_requirements")
            if isinstance(cfg_reqs, list):
                for item in cfg_reqs:
                    p = Path(item)
                    if not p.is_absolute():
                        p = repo_root / p
                    if p.exists():
                        req_files.append(p)
            if not has_req_arg and not req_files:
                for name in (
                    "requirements.txt",
                    "requirements-dev.txt",
                    "requirements.in",
                    "requirements-dev.in",
                    "constraints.txt",
                ):
                    p = repo_root / name
                    if p.exists():
                        req_files.append(p)
                        break
    
            if not has_req_arg and not req_files:
                logger.info("pip-audit skipped: no requirements file found")
                perf.end_timer("pip-audit")
            else:
                for req in req_files:
                    args += ["-r", str(req)]
                res = run_cmd(args, cwd=str(repo_root))
                pip_findings = [f.model_dump() for f in parse_pip_audit_json(res["stdout"])]
                findings.extend(pip_findings)
                logger.info(f"pip-audit: found {len(pip_findings)} issues")
                perf.end_timer("pip-audit")
        except Exception as e:
            logger.warning(f"pip-audit failed (non-fatal): {e}")

        _finish_tool()
    complexity = None
    if review_cfg.get("enable_radon", True):
        perf.start_timer("radon")
        try:
            args = ["radon", "cc", "-s", "-a"]
            if tool_excludes and not any(a in ("-e", "--exclude") for a in args):
                args += ["-e", ",".join(tool_excludes)]
            args += [str(repo_root)]
            res = run_cmd(args, cwd=str(repo_root))
            complexity = {"ok": res["ok"], "stdout": res["stdout"][:200000], "stderr": res["stderr"][:20000]}
            logger.info("radon: 复杂度分析完成")
            perf.end_timer("radon")
        except Exception as e:
            logger.warning(f"radon 执行失败: {e}")

        _finish_tool()
    mypy_out = None
    if review_cfg.get("enable_mypy", False):
        perf.start_timer("mypy")
        try:
            args = ["mypy"] + review_cfg.get("mypy_args", [])
            if tool_excludes and "--exclude" not in args:
                pattern = "(" + "|".join(re.escape(x) for x in tool_excludes) + ")"
                args += ["--exclude", pattern]
            args += [str(repo_root)]
            res = run_cmd(args, cwd=str(repo_root))
            mypy_out = {"ok": res["ok"], "stdout": res["stdout"][:200000], "stderr": res["stderr"][:20000]}
            logger.info("mypy: 类型检查完成")
            perf.end_timer("mypy")
        except Exception as e:
            logger.warning(f"mypy 执行失败: {e}")

        _finish_tool()
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
