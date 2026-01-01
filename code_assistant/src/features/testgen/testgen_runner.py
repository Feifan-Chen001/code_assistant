from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from .ast_extract import extract_public_functions
from .templates import make_test_module
from .coverage_runner import run_coverage

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

    for p in files:
        if p.suffix != ".py":
            continue
        funcs = extract_public_functions(p)
        if not funcs:
            continue

        if func_count >= max_funcs:
            break
        if func_count + len(funcs) > max_funcs:
            funcs = funcs[: max(0, max_funcs - func_count)]
        func_count += len(funcs)

        rel = str(p.relative_to(repo_root)).replace("\\\\", "/")
        test_name = "test_" + p.stem + ".py"
        content = make_test_module(rel, funcs, use_hypothesis=use_hypo)
        (out_dir / test_name).write_text(content, encoding="utf-8")
        written_files += 1
        generated.append({"source": str(p), "test_file": str(out_dir / test_name), "functions": funcs})

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
