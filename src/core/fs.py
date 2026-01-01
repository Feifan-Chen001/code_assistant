from __future__ import annotations
from pathlib import Path
from typing import List
import fnmatch

def iter_files(repo_path: str, include_globs: List[str], exclude_globs: List[str], max_files: int) -> List[Path]:
    base = Path(repo_path).resolve()
    out: List[Path] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(base)).replace("\\", "/")
        if exclude_globs and any(fnmatch.fnmatch(rel, g) for g in exclude_globs):
            continue
        if include_globs and not any(fnmatch.fnmatch(rel, g) for g in include_globs):
            continue
        out.append(p)
        if len(out) >= max_files:
            break
    return out
