from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from ...core.subproc import run_cmd

def run_coverage(repo_path: str, pytest_args: List[str]) -> Dict[str, Any]:
    repo = Path(repo_path).resolve()
    res = run_cmd(["coverage", "run", "-m", "pytest"] + (pytest_args or []), cwd=str(repo))
    rep = run_cmd(["coverage", "report", "-m"], cwd=str(repo))
    return {
        "ok": res["ok"],
        "pytest": {"stdout": res["stdout"][:200000], "stderr": res["stderr"][:20000], "cmd": res["cmd"]},
        "report": {"stdout": rep["stdout"][:200000], "stderr": rep["stderr"][:20000], "cmd": rep["cmd"]},
    }
