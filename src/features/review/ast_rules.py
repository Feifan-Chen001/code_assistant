from __future__ import annotations
import ast
from pathlib import Path
from typing import List
from .types import ReviewFinding

BANNED_CALLS = {"eval", "exec"}

def scan_file_ast(path: Path, repo_root: Path) -> List[ReviewFinding]:
    findings: List[ReviewFinding] = []
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return findings

    rel = str(path.relative_to(repo_root)).replace("\\", "/")

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                findings.append(ReviewFinding(
                    tool="ast-rule",
                    rule="BANNED_CALL",
                    severity="high",
                    message=f"禁止使用 {node.func.id}（项目/领域规则），建议安全替代或白名单。",
                    file=rel,
                    line=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                ))
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler):
            if node.type is None:
                findings.append(ReviewFinding(
                    tool="ast-rule",
                    rule="BARE_EXCEPT",
                    severity="medium",
                    message="避免裸 except:，应捕获明确异常类型。",
                    file=rel,
                    line=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                ))
            self.generic_visit(node)

    V().visit(tree)
    return findings


def scan_source_ast(source: str, rel_path: str) -> List[ReviewFinding]:
    findings: List[ReviewFinding] = []
    try:
        tree = ast.parse(source)
    except Exception:
        return findings

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                findings.append(ReviewFinding(
                    tool="ast-rule",
                    rule="BANNED_CALL",
                    severity="high",
                    message=f"Use of {node.func.id} is banned by project policy.",
                    file=rel_path,
                    line=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                ))
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler):
            if node.type is None:
                findings.append(ReviewFinding(
                    tool="ast-rule",
                    rule="BARE_EXCEPT",
                    severity="medium",
                    message="Avoid bare except; catch explicit exception types.",
                    file=rel_path,
                    line=getattr(node, "lineno", None),
                    col=getattr(node, "col_offset", None),
                ))
            self.generic_visit(node)

    V().visit(tree)
    return findings
