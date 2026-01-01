from __future__ import annotations
from typing import Any, Dict
from .fs import iter_files
from ..features.review.review_runner import run_review_pipeline
from ..features.testgen.testgen_runner import run_testgen_pipeline

class Orchestrator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _file_list(self, repo_path: str):
        a = self.cfg.get("assistant", {})
        return iter_files(
            repo_path=repo_path,
            include_globs=a.get("include_globs", ["**/*.py", "**/*.ipynb"]),
            exclude_globs=a.get("exclude_globs", []),
            max_files=int(a.get("max_files", 2000)),
        )

    def run_review(self, repo_path: str):
        files = self._file_list(repo_path)
        return run_review_pipeline(repo_path, files, self.cfg)

    def run_testgen(self, repo_path: str):
        files = self._file_list(repo_path)
        return run_testgen_pipeline(repo_path, files, self.cfg)

