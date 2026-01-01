from __future__ import annotations
import json
from typing import List
from .types import ReviewFinding

def parse_ruff_json(stdout: str) -> List[ReviewFinding]:
    out = []
    try:
        items = json.loads(stdout) if stdout.strip() else []
    except Exception:
        return out
    for it in items:
        out.append(ReviewFinding(
            tool="ruff",
            rule=str(it.get("code") or "RUFF"),
            severity="medium",
            message=str(it.get("message") or ""),
            file=it.get("filename"),
            line=it.get("location", {}).get("row"),
            col=it.get("location", {}).get("column"),
            extra={"fix": it.get("fix")},
        ))
    return out

def parse_bandit_json(stdout: str) -> List[ReviewFinding]:
    out = []
    try:
        obj = json.loads(stdout) if stdout.strip() else {}
    except Exception:
        return out
    for it in obj.get("results", []):
        sev = (it.get("issue_severity") or "LOW").lower()
        out.append(ReviewFinding(
            tool="bandit",
            rule=str(it.get("test_id") or "BANDIT"),
            severity={"low":"low","medium":"medium","high":"high"}.get(sev, "medium"),
            message=str(it.get("issue_text") or ""),
            file=it.get("filename"),
            line=it.get("line_number"),
            col=None,
            extra={"more_info": it.get("more_info"), "confidence": it.get("issue_confidence")},
        ))
    return out

def parse_pip_audit_json(stdout: str) -> List[ReviewFinding]:
    out = []
    try:
        obj = json.loads(stdout) if stdout.strip() else []
    except Exception:
        return out
    items = []
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        for key in ("dependencies", "results"):
            val = obj.get(key)
            if isinstance(val, list):
                items = val
                break
        if not items and ("dependency" in obj or "vulns" in obj):
            items = [obj]
    else:
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        dep = it.get("dependency", {}) or {}
        if not isinstance(dep, dict):
            dep = {}
        vulns = it.get("vulns", None)
        if vulns is None:
            vulns = it.get("vulnerabilities", [])
        if not isinstance(vulns, list):
            continue
        for v in vulns:
            if not isinstance(v, dict):
                continue
            out.append(ReviewFinding(
                tool="pip-audit",
                rule="VULN",
                severity="high",
                message=f"{dep.get('name')} {dep.get('version')} -> {v.get('id')} {v.get('description','')}".strip(),
                extra={"fix_versions": v.get("fix_versions"), "aliases": v.get("aliases")},
            ))
    return out
