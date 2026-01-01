from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple


def extract_code_cells(path: Path) -> List[Tuple[int, str]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw) if raw.strip() else {}
    except Exception:
        return []
    cells = obj.get("cells", []) if isinstance(obj, dict) else []
    out: List[Tuple[int, str]] = []
    for idx, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            text = "".join(src)
        else:
            text = str(src)
        if text.strip():
            out.append((idx, text))
    return out
