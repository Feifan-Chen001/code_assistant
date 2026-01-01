from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel

class ReviewFinding(BaseModel):
    tool: str
    rule: str
    severity: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    extra: Dict[str, Any] = {}
