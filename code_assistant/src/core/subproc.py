from __future__ import annotations
import subprocess
from typing import List, Optional, Dict, Any

def run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 1800) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "cmd": cmd}
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "returncode": -1, "stdout": e.stdout or "", "stderr": f"TIMEOUT: {e}", "cmd": cmd}
    except FileNotFoundError as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"NOT_FOUND: {e}", "cmd": cmd}
