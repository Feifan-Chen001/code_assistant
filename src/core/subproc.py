from __future__ import annotations
import subprocess
from typing import List, Optional, Dict, Any

def run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 1800) -> Dict[str, Any]:
    try:
        # 在Windows上，creationflags=subprocess.CREATE_NEW_PROCESS_GROUP 可以帮助正确处理Ctrl+C
        import sys
        kwargs = {
            "cwd": cwd,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": timeout,
            "check": False,
        }
        # Windows特定处理
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        
        p = subprocess.run(cmd, **kwargs)
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "cmd": cmd}
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "returncode": -1, "stdout": e.stdout or "", "stderr": f"TIMEOUT: {e}", "cmd": cmd}
    except FileNotFoundError as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"NOT_FOUND: {e}", "cmd": cmd}
    except KeyboardInterrupt:
        return {"ok": False, "returncode": -2, "stdout": "", "stderr": "INTERRUPTED: User cancelled", "cmd": cmd}
