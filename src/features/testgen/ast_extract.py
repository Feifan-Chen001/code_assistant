from __future__ import annotations
import ast
from pathlib import Path
from typing import List, Dict, Any


def extract_public_functions_from_source(source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except Exception:
        return out

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            args = [a.arg for a in node.args.args]
            doc = ast.get_docstring(node) or ""
            out.append({"name": node.name, "args": args, "lineno": getattr(node, "lineno", None), "doc": doc})
    return out


def extract_public_functions(path: Path) -> List[Dict[str, Any]]:
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return extract_public_functions_from_source(src)
