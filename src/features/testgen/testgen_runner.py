from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

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


def run_testgen_pipeline(repo_path: str, files: List[Path], cfg: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = Path(repo_path).resolve()
    tc = cfg.get("testgen", {})
    out_dir = Path(tc.get("output_dir", "generated_tests")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_hypo = bool(tc.get("use_hypothesis", True))
    max_funcs = int(tc.get("max_functions", 200))

    generated = []
    written_files = 0
    func_count = 0
    nb_dir: Optional[Path] = None

    for p in files:
        module_rel = None
        module_path = None
        module_name = None
        source_label = str(p)

        if p.suffix == ".py":
            funcs = extract_public_functions(p)
            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            module_rel = rel
            test_name = "test_" + p.stem + ".py"
        elif p.suffix == ".ipynb":
            nb_source = _notebook_source(p)
            if not nb_source.strip():
                continue
            funcs = extract_public_functions_from_source(nb_source)
            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            rel_no_ext = rel.rsplit(".", 1)[0]
            slug = _safe_slug(rel_no_ext)
            if nb_dir is None:
                nb_dir = out_dir / "_notebooks"
                nb_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "__init__.py").write_text("", encoding="utf-8")
                (nb_dir / "__init__.py").write_text("", encoding="utf-8")
            module_file = nb_dir / f"nb_{slug}.py"
            module_file.write_text(nb_source, encoding="utf-8")
            module_rel = module_file.relative_to(out_dir).as_posix()
            module_path = module_rel
            module_name = f"nb_{slug}"
            test_name = f"test_{slug}.py"
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
        cov_result = run_coverage(repo_path, pytest_args=cov_cfg.get("pytest_args", ["-q"]))

    return {
        "repo": str(repo_root),
        "output_dir": str(out_dir),
        "written_files": written_files,
        "function_count": func_count,
        "generated": generated[:50],
        "coverage": cov_result,
    }

