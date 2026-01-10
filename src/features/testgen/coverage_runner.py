from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
from ...core.subproc import run_cmd

def run_coverage(repo_path: str, pytest_args: List[str], test_paths: Optional[List[str]] = None, source_paths: Optional[List[str]] = None, omit_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    repo = Path(repo_path).resolve()
    # 修复：--source 和 --omit 参数必须放在 "-m pytest" 之前，因为它们是 coverage 的参数，不是 pytest 的参数
    args = ["coverage", "run"]
    
    # 添加 coverage 参数
    if source_paths:
        for src_path in source_paths:
            args.extend(["--source", src_path])
    
    if omit_patterns:
        for omit_pat in omit_patterns:
            args.extend(["--omit", omit_pat])
    
    # 然后添加 pytest 相关参数
    args.extend(["-m", "pytest"])
    args.extend(pytest_args or [])
    
    if test_paths:
        args += list(test_paths)
    
    res = run_cmd(args, cwd=str(repo))
    rep = run_cmd(["coverage", "report", "-m"], cwd=str(repo))
    return {
        "ok": res["ok"],
        "pytest": {"stdout": res["stdout"][:200000], "stderr": res["stderr"][:20000], "cmd": res["cmd"]},
        "report": {"stdout": rep["stdout"][:200000], "stderr": rep["stderr"][:20000], "cmd": rep["cmd"]},
    }
