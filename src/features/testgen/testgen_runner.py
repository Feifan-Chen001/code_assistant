from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import hashlib
import os

from .ast_extract import extract_public_functions, extract_public_functions_from_source
from .templates import make_test_module
from .coverage_runner import run_coverage
from ..review.notebook import extract_code_cells


def _strip_ipython_magics(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%%") or stripped.startswith("%") or stripped.startswith("!"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _notebook_source(path: Path) -> str:
    parts: List[str] = []
    for idx, code in extract_code_cells(path):
        cleaned = _strip_ipython_magics(code)
        if not cleaned.strip():
            continue
        parts.append(f"# --- notebook cell {idx} ---\n{cleaned}")
    return "\n\n".join(parts)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return slug or "notebook"

def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

def _rel_module_path(target: Path, base: Path) -> str:
    try:
        rel = os.path.relpath(target, base)
    except ValueError:
        return str(target.resolve())
    return Path(rel).as_posix()


def _is_test_like_path(p: Path, repo_root: Path) -> bool:
    """Avoid generating tests for existing tests (prevents test_test_... explosion)."""
    try:
        rel = p.relative_to(repo_root).as_posix().lower()
    except Exception:
        rel = p.name.lower()
    name = p.name.lower()
    if rel.startswith("tests/") or rel.startswith("test/"):
        return True
    if "/tests/" in rel or "/test/" in rel:
        return True
    if name == "conftest.py":
        return True
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return False


def _infer_source_paths(repo_root: Path, out_dir: Path) -> List[str]:
    """Pick a tighter --source set for coverage so TOTAL is interpretable."""
    candidates: List[Path] = []
    src_dir = repo_root / "src"
    if src_dir.exists() and src_dir.is_dir():
        candidates.append(src_dir)

    # Common layout: a single top-level package dir (e.g., pandas/, sklearn/)
    pkgs: List[Path] = []
    for child in repo_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name in {"tests", "test", "generated_tests", "reports", "__pycache__"}:
            continue
        # if out_dir is inside repo_root, don't include it as source
        try:
            child.resolve().relative_to(out_dir.resolve())
            continue
        except Exception:
            pass
        if (child / "__init__.py").exists():
            pkgs.append(child)
    if pkgs and not candidates:
        candidates.extend(pkgs)

    if not candidates:
        candidates = [repo_root]

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for c in candidates:
        # 修复：在Windows上使用原生路径格式，避免as_posix()导致的路径混合问题
        p = str(c.resolve())
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def run_testgen_pipeline(repo_path: str, files: List[Path], cfg: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = Path(repo_path).resolve()
    tc = cfg.get("testgen", {})
    out_dir = Path(tc.get("output_dir", "generated_tests"))
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_rel = _rel_module_path(repo_root, out_dir)

    use_hypo = bool(tc.get("use_hypothesis", True))
    max_funcs = int(tc.get("max_functions", 200))

    generated = []
    written_files = 0
    func_count = 0
    nb_dir: Optional[Path] = None

    for p in files:
        try:
            p.relative_to(out_dir)
        except ValueError:
            pass
        else:
            continue
        if p.suffix == '.py' and _is_test_like_path(p, repo_root):
            continue
        module_rel = None
        module_path = None
        module_name = None
        source_label = str(p)

        if p.suffix == ".py":
            funcs = extract_public_functions(p)
            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            rel_no_ext = rel.rsplit(".", 1)[0]
            slug = _safe_slug(rel_no_ext)
            suffix = _short_hash(rel)
            module_rel = rel
            module_name = f"mod_{slug}_{suffix}"
            module_path = _rel_module_path(p, out_dir)
            test_name = f"test_{slug}_{suffix}.py"
        elif p.suffix == ".ipynb":
            nb_source = _notebook_source(p)
            if not nb_source.strip():
                continue
            funcs = extract_public_functions_from_source(nb_source)
            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            rel_no_ext = rel.rsplit(".", 1)[0]
            slug = _safe_slug(rel_no_ext)
            suffix = _short_hash(rel_no_ext)
            if nb_dir is None:
                nb_dir = out_dir / "_notebooks"
                nb_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "__init__.py").write_text("", encoding="utf-8")
                (nb_dir / "__init__.py").write_text("", encoding="utf-8")
            module_name = f"nb_{slug}_{suffix}"
            module_file = nb_dir / f"{module_name}.py"
            module_file.write_text(nb_source, encoding="utf-8")
            module_rel = module_file.relative_to(out_dir).as_posix()
            module_path = module_rel
            test_name = f"test_{slug}_{suffix}.py"
            source_label = str(p)
        else:
            continue

        if not funcs:
            continue

        if func_count >= max_funcs:
            break
        if func_count + len(funcs) > max_funcs:
            funcs = funcs[: max(0, max_funcs - func_count)]
        func_count += len(funcs)

        content = make_test_module(
            module_rel,
            funcs,
            use_hypothesis=use_hypo,
            module_path=module_path,
            module_name=module_name,
            repo_root=repo_rel,
        )
        (out_dir / test_name).write_text(content, encoding="utf-8")
        written_files += 1
        record = {"source": source_label, "test_file": str(out_dir / test_name), "functions": funcs}
        if module_path:
            record["notebook_module"] = str((out_dir / module_path).resolve())
        generated.append(record)

    cov_cfg = cfg.get("coverage", {})
    cov_result = None
    if bool(cov_cfg.get("enable", True)):
        # Always run generated tests to measure their impact. If out_dir is outside the repo,
        # pass it explicitly to pytest; otherwise running it explicitly is still fine.
        test_paths = [str(out_dir)]

        # Coverage should measure the *target repo code*, not the generated tests themselves.
        # 修复：在Windows上使用原生路径格式（反斜杠），避免as_posix()导致的混合路径问题
        omit_patterns = [
            str(out_dir.resolve()) + "/**",
            str(out_dir.resolve()) + "/*",  # keep for older coverage glob behavior
        ]
        # Also omit repo-internal test directories when --source is wide (repo_root fallback).
        tests_dir = (repo_root / "tests")
        if tests_dir.exists():
            omit_patterns.append(str(tests_dir.resolve()) + "/**")
        test_dir = (repo_root / "test")
        if test_dir.exists():
            omit_patterns.append(str(test_dir.resolve()) + "/**")

        source_paths = _infer_source_paths(repo_root, out_dir)

        cov_result = run_coverage(
            repo_path,
            pytest_args=cov_cfg.get("pytest_args", ["-q"]),
            test_paths=test_paths,
            source_paths=source_paths,
            omit_patterns=omit_patterns,
        )

    return {
        "repo": str(repo_root),
        "output_dir": str(out_dir),
        "written_files": written_files,
        "function_count": func_count,
        "generated": generated[:50],
        "coverage": cov_result,
    }

