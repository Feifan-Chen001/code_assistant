from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
from .fs import iter_files
from ..features.review.review_runner import run_review_pipeline
from ..features.testgen.testgen_runner import run_testgen_pipeline

class Orchestrator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _file_list(self, repo_path: str):
        a = self.cfg.get("assistant", {})
        include_globs = list(a.get("include_globs", ["**/*.py", "**/*.ipynb"]))
        exclude_globs = list(a.get("exclude_globs", []))
        out_dir = (self.cfg.get("testgen", {}) or {}).get("output_dir")
        if out_dir:
            out_path = Path(out_dir)
            if not out_path.is_absolute():
                out_path = (Path(repo_path).resolve() / out_path).resolve()
            else:
                out_path = out_path.resolve()
            try:
                rel = out_path.relative_to(Path(repo_path).resolve())
            except ValueError:
                rel = None
            if rel:
                rel_posix = rel.as_posix().rstrip("/")
                if rel_posix:
                    exclude_globs.append(f"{rel_posix}/**")
        return iter_files(
            repo_path=repo_path,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_files=int(a.get("max_files", 2000)),
        )

    def run_review(self, repo_path: str):
        files = self._file_list(repo_path)
        return run_review_pipeline(repo_path, files, self.cfg)

    def run_testgen(self, repo_path: str):
        files = self._file_list(repo_path)
        return run_testgen_pipeline(repo_path, files, self.cfg)

