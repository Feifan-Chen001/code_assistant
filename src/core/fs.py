from __future__ import annotations
from pathlib import Path
from typing import List
import fnmatch

def _normalize_globs(globs: List[str]) -> List[str]:
    """Normalize glob patterns to work well with fnmatch on POSIX-style relative paths.

    - Convert backslashes to slashes.
    - Drop leading './'.
    - Expand patterns that start with '**/' so they also match root-level paths.
      (e.g. '**/generated_tests/**' won't match 'generated_tests/x.py' in fnmatch)
    """
    if not globs:
        return []
    out: List[str] = []
    seen = set()
    for g in globs:
        if g is None:
            continue
        pat = str(g).strip().replace('\\', '/')
        if not pat:
            continue
        if pat.startswith("./"):
            pat = pat[2:]
        candidates = [pat]
        if pat.startswith("**/"):
            candidates.append(pat[3:])
        for c in candidates:
            if c not in seen:
                out.append(c)
                seen.add(c)
    return out

def iter_files(repo_path: str, include_globs: List[str], exclude_globs: List[str], max_files: int) -> List[Path]:
    base = Path(repo_path).resolve()
    include = _normalize_globs(include_globs)
    exclude = _normalize_globs(exclude_globs)

    out: List[Path] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()

        # Exclude first (cheap and avoids self-pollution)
        if exclude and any(fnmatch.fnmatchcase(rel, g) for g in exclude):
            continue

        # Include filter
        if include and not any(fnmatch.fnmatchcase(rel, g) for g in include):
            continue

        out.append(p)
        if len(out) >= max_files:
            break
    return out
