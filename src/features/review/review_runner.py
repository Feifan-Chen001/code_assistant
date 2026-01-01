from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from ...core.subproc import run_cmd
from .parsers import parse_ruff_json, parse_bandit_json, parse_pip_audit_json
from .ast_rules import scan_file_ast, scan_source_ast
from .ds_rules import scan_file_ds, scan_source_ds
from .notebook import extract_code_cells

def run_review_pipeline(repo_path: str, files: List[Path], cfg: Dict[str, Any]) -> Dict[str, Any]:
    review_cfg = cfg.get("review", {})
    repo_root = Path(repo_path).resolve()

    findings = []
    enable_ds = review_cfg.get("enable_ds_rules", True)
    enable_notebook = review_cfg.get("enable_notebook", True)

    for p in files:
        if p.suffix == ".ipynb":
            if not enable_notebook:
                continue
            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            for idx, code in extract_code_cells(p):
                cell_rel = f"{rel}#cell-{idx}"
                findings.extend([f.model_dump() for f in scan_source_ast(code, cell_rel)])
                if enable_ds:
                    findings.extend([f.model_dump() for f in scan_source_ds(code, cell_rel)])
            continue
        findings.extend([f.model_dump() for f in scan_file_ast(p, repo_root)])
        if enable_ds and p.suffix == ".py":
            findings.extend([f.model_dump() for f in scan_file_ds(p, repo_root)])

    if review_cfg.get("enable_ruff", True):
        args = ["ruff"] + review_cfg.get("ruff_args", ["check", "--format", "json"]) + [str(repo_root)]
        res = run_cmd(args, cwd=str(repo_root))
        findings.extend([f.model_dump() for f in parse_ruff_json(res["stdout"])])

    if review_cfg.get("enable_bandit", True):
        args = ["bandit"] + review_cfg.get("bandit_args", ["-r", "-f", "json"]) + [str(repo_root)]
        res = run_cmd(args, cwd=str(repo_root))
        findings.extend([f.model_dump() for f in parse_bandit_json(res["stdout"])])

    if review_cfg.get("enable_pip_audit", True):
        args = ["pip-audit"] + review_cfg.get("pip_audit_args", ["-f", "json"])
        res = run_cmd(args, cwd=str(repo_root))
        findings.extend([f.model_dump() for f in parse_pip_audit_json(res["stdout"])])

    complexity = None
    if review_cfg.get("enable_radon", True):
        res = run_cmd(["radon", "cc", "-s", "-a", str(repo_root)], cwd=str(repo_root))
        complexity = {"ok": res["ok"], "stdout": res["stdout"][:200000], "stderr": res["stderr"][:20000]}

    mypy_out = None
    if review_cfg.get("enable_mypy", False):
        args = ["mypy"] + review_cfg.get("mypy_args", []) + [str(repo_root)]
        res = run_cmd(args, cwd=str(repo_root))
        mypy_out = {"ok": res["ok"], "stdout": res["stdout"][:200000], "stderr": res["stderr"][:20000]}

    return {
        "repo": str(repo_root),
        "findings": findings,
        "tool_raw": {"complexity_radon": complexity, "mypy": mypy_out},
        "stats": {"total_findings": len(findings)},
    }
