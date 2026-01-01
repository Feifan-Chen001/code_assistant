from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

from src.core.config import load_config
from src.core.orchestrator import Orchestrator
from src.reporting.report_builder import build_markdown_report


def _load_repo_list(path: str):
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    repos = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    return repos


def _prepare_cfg(cfg, test_out: Optional[Path]):
    cfg = dict(cfg)
    if test_out is not None:
        cfg.setdefault("testgen", {})
        cfg["testgen"] = dict(cfg["testgen"])
        cfg["testgen"]["output_dir"] = str(test_out)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("review")
    p1.add_argument("--repo", type=str, required=True)
    p1.add_argument("--out", type=str, default="reports")

    p2 = sub.add_parser("testgen")
    p2.add_argument("--repo", type=str, required=True)
    p2.add_argument("--out", type=str, default="generated_tests")

    p3 = sub.add_parser("all")
    p3.add_argument("--repo", type=str, required=True)
    p3.add_argument("--out", type=str, default="reports")

    p4 = sub.add_parser("batch")
    p4.add_argument("--repos", type=str, required=True, help="Text file with repo paths, one per line.")
    p4.add_argument("--mode", type=str, choices=["review", "testgen", "all"], default="all")
    p4.add_argument("--out", type=str, default="reports_batch")

    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "testgen":
        cfg = _prepare_cfg(cfg, Path(args.out).resolve())

    orch = Orchestrator(cfg)

    if args.cmd == "review":
        r = orch.run_review(args.repo)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "review.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote {Path(args.out)/'review.json'}")

    elif args.cmd == "testgen":
        t = orch.run_testgen(args.repo)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "testgen.json").write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote {Path(args.out)/'testgen.json'}")

    elif args.cmd == "all":
        r = orch.run_review(args.repo)
        t = orch.run_testgen(args.repo)
        md = build_markdown_report(r, t)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "review.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        (Path(args.out) / "testgen.json").write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
        (Path(args.out) / "report.md").write_text(md, encoding="utf-8")
        print(f"[ok] wrote {Path(args.out)/'report.md'}")

    elif args.cmd == "batch":
        out_root = Path(args.out).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        repos = _load_repo_list(args.repos)
        for idx, repo in enumerate(repos, start=1):
            repo_path = Path(repo).resolve()
            name = repo_path.name or f"repo_{idx}"
            repo_out = out_root / name
            repo_out.mkdir(parents=True, exist_ok=True)

            test_out = None
            if args.mode in ("testgen", "all"):
                test_out = repo_out / "generated_tests"
            cfg_run = _prepare_cfg(cfg, test_out)
            orch = Orchestrator(cfg_run)

            if args.mode == "review":
                r = orch.run_review(str(repo_path))
                (repo_out / "review.json").write_text(
                    json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[ok] wrote {repo_out/'review.json'}")

            elif args.mode == "testgen":
                t = orch.run_testgen(str(repo_path))
                (repo_out / "testgen.json").write_text(
                    json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[ok] wrote {repo_out/'testgen.json'}")

            elif args.mode == "all":
                r = orch.run_review(str(repo_path))
                t = orch.run_testgen(str(repo_path))
                md = build_markdown_report(r, t)
                (repo_out / "review.json").write_text(
                    json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (repo_out / "testgen.json").write_text(
                    json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (repo_out / "report.md").write_text(md, encoding="utf-8")
                print(f"[ok] wrote {repo_out/'report.md'}")


if __name__ == "__main__":
    main()
