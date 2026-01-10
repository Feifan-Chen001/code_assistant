# -*- coding: utf-8 -*-
"""
Pretty viz (NO matplotlib): 4 stage-ready figures for CodeAssistantPro.

Fixes:
- Ignore reports root run (avoid 'reports' as a repo)
- Collapse notebook cell paths: xxx.ipynb#cell-123 -> xxx.ipynb
- Treemap aggregates by directory/module (depth=2), top-N + Other, no label clutter
- Replace Sankey with a clean "closed-loop funnel/progress" diagram
- Replace Radar with "risk small-multiples bar" (more readable & product-like)

Outputs:
  01_repo_rule_heatmap.png
  02_repo_risk_bars.png
  03_closed_loop_funnel__<repo>.png
  04_hotspot_treemap__<repo>.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# IO helpers
# ----------------------------
def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_name(s: str) -> str:
    s = (s or "").strip().replace("\\", "/")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    return s[:120] if len(s) > 120 else s

def _norm_path(p: str) -> str:
    return (p or "").replace("\\", "/")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ----------------------------
# Fonts (best effort)
# ----------------------------
def _load_font(size: int) -> ImageFont.ImageFont:
    cand = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for fp in cand:
        try:
            if os.path.exists(fp):
                return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


# ----------------------------
# Path normalization (KEY FIX)
# ----------------------------
_CELL_SUFFIX_RE = re.compile(r"(#cell-\d+)$", re.IGNORECASE)

def normalize_code_ref(path: str) -> str:
    """
    1) normalize slashes
    2) collapse notebook cell refs: xxx.ipynb#cell-123 -> xxx.ipynb
    """
    p = _norm_path(path).strip()
    p = _CELL_SUFFIX_RE.sub("", p)
    return p

def shorten_repo_id(repo_id: str) -> str:
    # show tail part for readability if needed
    return repo_id.replace("__", "/")

def dir_prefix(rel_path: str, depth: int = 2) -> str:
    rel_path = normalize_code_ref(rel_path).lstrip("/")
    parts = [p for p in rel_path.split("/") if p]
    if not parts:
        return "."
    return "/".join(parts[: max(1, depth)])


# ----------------------------
# Discover report runs
# ----------------------------
def discover_report_runs(reports_dir: str, include_root_run: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Return {repo_id: {"review": path, "testgen": path}} for folders containing both json files.
    By default: ignore reports_dir root run even if review.json exists there.
    Also ignore any folders like 'my_repo' (local sandbox repo).
    """
    runs: Dict[str, Dict[str, str]] = {}

    IGNORE_REPO_IDS = {"my_repo"}  # add more if needed

    for root, _, files in os.walk(reports_dir):
        fs = set(files)
        if "review.json" in fs and "testgen.json" in fs:
            # ignore root run unless include_root_run=True
            if os.path.abspath(root) == os.path.abspath(reports_dir) and not include_root_run:
                continue

            repo_id = os.path.basename(root)

            # ignore local sandbox repos
            if repo_id in IGNORE_REPO_IDS:
                continue

            runs[repo_id] = {
                "review": os.path.join(root, "review.json"),
                "testgen": os.path.join(root, "testgen.json"),
            }

    # optional root fallback
    if include_root_run:
        root_review = os.path.join(reports_dir, "review.json")
        root_testgen = os.path.join(reports_dir, "testgen.json")
        if os.path.exists(root_review) and os.path.exists(root_testgen):
            runs["reports_root"] = {"review": root_review, "testgen": root_testgen}

    return runs


# ----------------------------
# Domain helpers
# ----------------------------
SEV_WEIGHT = {"high": 3, "medium": 2, "low": 1}

def is_ds_finding(f: dict) -> bool:
    tool = (f.get("tool") or "").lower()
    rule = (f.get("rule") or "").upper()
    return tool == "ds-rule" or rule.startswith("DS_")

def is_generated_artifact_path(p: str) -> bool:
    p = _norm_path(p).lower()
    return (
        "/generated_tests/" in p or p.startswith("generated_tests/")
        or "/reports/" in p or p.startswith("reports/")
    )

def rule_short(rule: str) -> str:
    rule = (rule or "").upper()
    rule = rule.replace("DS_", "")
    rule = rule.replace("PANDAS_", "PD_")
    rule = rule.replace("SKLEARN_", "SK_")
    rule = rule.replace("HYPERPARAMS_", "HP_")
    return rule[:28]


# ----------------------------
# Repo stats (for normalization)
# ----------------------------
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "generated_tests", "reports", "dist", "build", ".idea", ".vscode",
}

def iter_code_files(repo_root: str, ex_dirs: Optional[set] = None) -> Iterable[str]:
    ex_dirs = ex_dirs or DEFAULT_EXCLUDE_DIRS
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ex_dirs]
        for fn in files:
            if fn.endswith(".py") or fn.endswith(".ipynb"):
                yield os.path.join(root, fn)

def count_files_and_kloc(repo_root: str) -> Tuple[int, float]:
    file_count = 0
    loc = 0
    for p in iter_code_files(repo_root):
        file_count += 1
        if p.endswith(".py"):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    loc += sum(1 for _ in f)
            except Exception:
                pass
    return max(file_count, 1), max(loc / 1000.0, 1e-6)


# ----------------------------
# Squarify treemap
# ----------------------------
def _normalize_sizes(sizes: List[float], dx: float, dy: float) -> List[float]:
    total = sum(sizes)
    if total <= 0:
        return [0 for _ in sizes]
    factor = dx * dy / total
    return [s * factor for s in sizes]

def _worst_ratio(row: List[float], w: float) -> float:
    if not row or w <= 0:
        return float("inf")
    s = sum(row)
    rmax = max(row)
    rmin = min(row)
    return max((w * w * rmax) / (s * s), (s * s) / (w * w * rmin))

def _layout_row(row: List[float], x: float, y: float, dx: float, dy: float, horizontal: bool):
    rects = []
    s = sum(row)
    if s <= 0:
        return rects
    if horizontal:
        h = s / dx if dx > 0 else 0
        cx = x
        for r in row:
            w = r / h if h > 0 else 0
            rects.append((cx, y, w, h))
            cx += w
    else:
        w = s / dy if dy > 0 else 0
        cy = y
        for r in row:
            h = r / w if w > 0 else 0
            rects.append((x, cy, w, h))
            cy += h
    return rects

def squarify(sizes: List[float], x: float, y: float, dx: float, dy: float):
    sizes = [float(s) for s in sizes if s > 0]
    sizes = sorted(sizes, reverse=True)
    sizes = _normalize_sizes(sizes, dx, dy)

    rects = []
    row = []
    w = min(dx, dy)

    while sizes:
        c = sizes[0]
        if not row or _worst_ratio(row + [c], w) <= _worst_ratio(row, w):
            row.append(c)
            sizes.pop(0)
        else:
            horizontal = dx >= dy
            rects.extend(_layout_row(row, x, y, dx, dy, horizontal))
            s = sum(row)
            if horizontal:
                h = s / dx if dx > 0 else 0
                y += h
                dy -= h
            else:
                ww = s / dy if dy > 0 else 0
                x += ww
                dx -= ww
            row = []
            w = min(dx, dy)

    if row:
        horizontal = dx >= dy
        rects.extend(_layout_row(row, x, y, dx, dy, horizontal))
    return rects


# ----------------------------
# Viridis-like palette (hand)
# ----------------------------
_VIRIDIS = [
    (68, 1, 84),
    (71, 44, 122),
    (59, 81, 139),
    (44, 113, 142),
    (33, 144, 141),
    (39, 173, 129),
    (92, 200, 99),
    (170, 220, 50),
    (253, 231, 37),
]

def viridis(t: float) -> Tuple[int, int, int]:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    n = len(_VIRIDIS)
    x = t * (n - 1)
    i = int(math.floor(x))
    j = min(i + 1, n - 1)
    a = x - i
    r0, g0, b0 = _VIRIDIS[i]
    r1, g1, b1 = _VIRIDIS[j]
    r = int(round(r0 * (1 - a) + r1 * a))
    g = int(round(g0 * (1 - a) + g1 * a))
    b = int(round(b0 * (1 - a) + b1 * a))
    return (r, g, b)


# ----------------------------
# Data model
# ----------------------------
@dataclass
class RepoRun:
    repo_id: str
    repo_root: str
    review: dict
    testgen: dict
    findings: List[dict]
    file_count: int
    kloc: float


def resolve_repo_root(repo_id: str, review_obj: dict, testgen_obj: dict, git_repo_dir: str) -> str:
    cand1 = os.path.join(git_repo_dir, repo_id)
    if os.path.isdir(cand1):
        return cand1
    v = review_obj.get("repo") or testgen_obj.get("repo")
    if isinstance(v, str) and os.path.isdir(v):
        return v
    return git_repo_dir


def load_runs(reports_dir: str, git_repo_dir: str, include_root_run: bool = False) -> List[RepoRun]:
    runs_map = discover_report_runs(reports_dir, include_root_run=include_root_run)
    if not runs_map:
        raise SystemExit(f"No report runs found under: {reports_dir}")

    out: List[RepoRun] = []
    for repo_id, paths in runs_map.items():
        review = _read_json(paths["review"])
        testgen = _read_json(paths["testgen"])
        findings = review.get("findings") or []

        repo_root = resolve_repo_root(repo_id, review, testgen, git_repo_dir)
        fc, kloc = (count_files_and_kloc(repo_root) if os.path.isdir(repo_root) else (1, 1e-6))

        out.append(RepoRun(
            repo_id=repo_id,
            repo_root=repo_root,
            review=review,
            testgen=testgen,
            findings=findings,
            file_count=fc,
            kloc=kloc,
        ))
    return out


# ----------------------------
# Drawing helpers
# ----------------------------
def save_img(img: Image.Image, out_path: str):
    _ensure_dir(os.path.dirname(out_path) or ".")
    img.save(out_path, format="PNG")

def draw_title(d: ImageDraw.ImageDraw, W: int, title: str, y: int, font: ImageFont.ImageFont):
    tw = d.textlength(title, font=font)
    d.text(((W - tw) / 2, y), title, fill=(20, 20, 20), font=font)

def ellipsize(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"

def draw_wrapped(d: ImageDraw.ImageDraw, xy, text: str, font, fill, max_w: int, line_h: int, max_lines: int = 2):
    x, y = xy

    # PIL ImageDraw.textlength() can't measure multiline text (contains '\n')
    # Normalize: replace newlines/tabs with spaces before wrapping.
    text = (text or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()

    words = [w for w in text.split(" ") if w]
    lines = []
    cur = ""

    for w in words:
        cand = (cur + " " + w).strip()
        # cand must be single-line for textlength
        if d.textlength(cand, font=font) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break

    if len(lines) < max_lines and cur:
        lines.append(cur)

    # If still too many, trim
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    # Add ellipsis if last line likely truncated (best-effort)
    if lines:
        last = lines[-1]
        if d.textlength(last, font=font) > max_w:
            while last and d.textlength(last + "…", font=font) > max_w:
                last = last[:-1]
            lines[-1] = (last + "…") if last else "…"

    for i, line in enumerate(lines):
        d.text((x, y + i * line_h), line, font=font, fill=fill)


# ----------------------------
# Plot 1: Heatmap (clean labels)
# ----------------------------
def make_heatmap_repo_rule(runs: List[RepoRun], out_path: str, top_rules: int = 10, normalize: str = "files"):
    rule_total: Dict[str, int] = {}
    repo_rule: Dict[Tuple[str, str], int] = {}

    for r in runs:
        for f in r.findings:
            file_ = f.get("file", "")
            if not file_ or is_generated_artifact_path(file_):
                continue
            if not is_ds_finding(f):
                continue
            rule = rule_short(f.get("rule") or "DS_UNKNOWN")
            repo_rule[(r.repo_id, rule)] = repo_rule.get((r.repo_id, rule), 0) + 1
            rule_total[rule] = rule_total.get(rule, 0) + 1

    if not rule_total:
        img = Image.new("RGB", (1200, 360), (255, 255, 255))
        d = ImageDraw.Draw(img)
        draw_title(d, 1200, "Repo × DS Rule Heatmap", 20, _load_font(28))
        d.text((40, 160), "No DS findings found in reports.", fill=(40, 40, 40), font=_load_font(18))
        save_img(img, out_path)
        return

    # choose top rules
    rules = [k for k, _ in sorted(rule_total.items(), key=lambda x: x[1], reverse=True)[:top_rules]]
    repo_ids = [r.repo_id for r in runs]

    # build matrix
    mat = np.zeros((len(repo_ids), len(rules)), dtype=float)
    for i, repo in enumerate(repo_ids):
        rr = next(x for x in runs if x.repo_id == repo)
        denom = 1.0
        if normalize == "files":
            denom = float(rr.file_count)
        elif normalize == "kloc":
            denom = float(rr.kloc)
        else:
            denom = 1.0
        for j, rule in enumerate(rules):
            mat[i, j] = repo_rule.get((repo, rule), 0) / denom

    vmax = float(np.max(mat)) if mat.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    # canvas
    cell = 52
    left = 320
    top = 120
    right = 360
    bottom = 60
    W = left + cell * len(rules) + right
    H = top + cell * len(repo_ids) + bottom

    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ft = _load_font(28)
    fs = _load_font(18)
    fsm = _load_font(16)

    draw_title(d, W, f"Repo × DS Rule Heatmap (per {normalize})", 20, ft)

    # x labels (short rule names, no rotation; use two-line wrap)
    for j, rule in enumerate(rules):
        x0 = left + j * cell
        # small header box
        d.rectangle([x0, top - 70, x0 + cell - 4, top - 6], fill=(248, 248, 248), outline=(230, 230, 230))
        draw_wrapped(d, (x0 + 6, top - 66), rule.replace("_", " "), fsm, (40, 40, 40), max_w=cell - 12, line_h=18, max_lines=3)

    # y labels
    for i, repo in enumerate(repo_ids):
        y0 = top + i * cell
        name = shorten_repo_id(repo)
        d.text((20, y0 + 16), ellipsize(name, 26), fill=(25, 25, 25), font=fs)

    # cells + numeric annotation
    for i in range(len(repo_ids)):
        for j in range(len(rules)):
            v = mat[i, j] / vmax
            color = viridis(v)
            x0 = left + j * cell
            y0 = top + i * cell
            d.rounded_rectangle([x0, y0, x0 + cell - 6, y0 + cell - 6], radius=10, fill=color, outline=(245, 245, 245))
            # annotate only if non-trivial
            if mat[i, j] > 0:
                txt = f"{mat[i, j]:.2f}"
                tw = d.textlength(txt, font=fsm)
                d.text((x0 + (cell - 6 - tw) / 2, y0 + 14), txt, fill=(10, 10, 10), font=fsm)

    # legend
    lx = left + cell * len(rules) + 24
    d.text((lx, top - 70), "Legend", fill=(20, 20, 20), font=fs)
    d.text((lx, top - 40), "Cell = DS hits / denom", fill=(70, 70, 70), font=fsm)

    # colorbar
    barw, barh = 220, 14
    bx, by = lx, top
    d.text((bx, by - 26), "Low → High", fill=(60, 60, 60), font=fsm)
    for x in range(barw):
        t = x / max(barw - 1, 1)
        d.line([(bx + x, by), (bx + x, by + barh)], fill=viridis(t))
    d.rectangle([bx, by, bx + barw, by + barh], outline=(0, 0, 0))
    d.text((bx, by + 24), f"max={vmax:.3f}", fill=(70, 70, 70), font=fsm)

    save_img(img, out_path)


# ----------------------------
# Plot 2: Repo risk (small multiples bars)
# ----------------------------
def bucket_scores(r: RepoRun) -> Dict[str, float]:
    buckets = {
        "Repro": 0.0,
        "Leak/Eval": 0.0,
        "Pandas": 0.0,
        "Security": 0.0,
        "Dependency": 0.0,
    }
    denom = float(r.file_count)

    for f in r.findings:
        file_ = f.get("file", "")
        if not file_ or is_generated_artifact_path(file_):
            continue

        tool = (f.get("tool") or "").lower()
        rule = (f.get("rule") or "").upper()
        msg = (f.get("message") or "").lower()

        if tool == "bandit":
            buckets["Security"] += 1.0
        if tool == "pip-audit":
            buckets["Dependency"] += 1.0

        if is_ds_finding(f):
            if "SEED" in rule or "RANDOM" in rule or "random_state" in msg:
                buckets["Repro"] += 1.0
            if "LEAK" in rule or "SPLIT" in rule or "CV" in rule or "STRATIFY" in rule or "leak" in msg:
                buckets["Leak/Eval"] += 1.0
            if "SETTING" in rule or "ITERROWS" in rule or "APPLY" in rule or "CHAIN" in rule:
                buckets["Pandas"] += 1.0

    for k in buckets:
        buckets[k] = buckets[k] / denom
    return buckets

def make_risk_bars(runs: List[RepoRun], out_path: str, max_repos: int = 6):
    # choose repos with most DS findings
    def ds_cnt(rr: RepoRun) -> int:
        return sum(1 for f in rr.findings if is_ds_finding(f) and not is_generated_artifact_path(f.get("file", "")))

    ranked = sorted(runs, key=ds_cnt, reverse=True)[:max_repos]
    if not ranked:
        img = Image.new("RGB", (1200, 360), (255, 255, 255))
        d = ImageDraw.Draw(img)
        draw_title(d, 1200, "Repo Risk (bars)", 20, _load_font(28))
        d.text((40, 160), "No repos/runs found.", fill=(40, 40, 40), font=_load_font(18))
        save_img(img, out_path)
        return

    axes = ["Repro", "Leak/Eval", "Pandas", "Security", "Dependency"]
    vals = []
    for rr in ranked:
        b = bucket_scores(rr)
        vals.append([b[a] for a in axes])
    arr = np.array(vals, dtype=float)
    # robust scaling: normalize by 95th percentile to avoid single outlier
    scale = np.quantile(arr[arr > 0], 0.95) if np.any(arr > 0) else 1.0
    scale = max(scale, 1e-9)

    W, H = 1400, 180 + 120 * len(ranked)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ft = _load_font(28)
    fs = _load_font(18)
    fsm = _load_font(16)

    draw_title(d, W, "Repo Risk Small Multiples (normalized rate per file)", 20, ft)

    # legend
    lx, ly = 40, 90
    d.text((lx, ly), f"Bar length = min(rate/scale, 1). scale≈{scale:.3f}", fill=(90, 90, 90), font=fsm)

    start_y = 130
    bar_x = 360
    bar_w = 920
    row_h = 110

    colors = {
        "Repro": (31, 119, 180),
        "Leak/Eval": (214, 39, 40),
        "Pandas": (255, 127, 14),
        "Security": (148, 103, 189),
        "Dependency": (44, 160, 44),
    }

    for i, rr in enumerate(ranked):
        y = start_y + i * row_h
        d.text((40, y + 8), ellipsize(shorten_repo_id(rr.repo_id), 40), fill=(20, 20, 20), font=fs)

        b = bucket_scores(rr)
        # draw 5 bars stacked vertically
        for k, name in enumerate(axes):
            yy = y + 40 + k * 13
            rate = float(b[name])
            frac = min(rate / scale, 1.0)
            bw = int(bar_w * frac)

            d.rectangle([bar_x, yy, bar_x + bar_w, yy + 10], fill=(245, 245, 245))
            d.rectangle([bar_x, yy, bar_x + bw, yy + 10], fill=colors[name])
            d.text((bar_x - 120, yy - 2), name, fill=(60, 60, 60), font=fsm)

        # right numeric summary
        total = sum(b.values())
        d.text((bar_x + bar_w + 18, y + 46), f"sum={total:.3f}", fill=(60, 60, 60), font=fsm)

    save_img(img, out_path)


# ----------------------------
# Plot 3: Closed-loop funnel (replaces Sankey)
# ----------------------------
def make_closed_loop_funnel(repo: RepoRun, out_path: str):
    total_files = int(repo.file_count)

    finding_files = {
        normalize_code_ref(f.get("file", ""))
        for f in repo.findings
        if f.get("file") and not is_generated_artifact_path(f.get("file", ""))
    }
    n_find_files = min(len(finding_files), total_files)

    sources = set()
    gen = repo.testgen.get("generated") or []
    if isinstance(gen, list):
        for g in gen:
            src = g.get("source") or g.get("notebook_module") or g.get("module") or ""
            if src:
                sources.add(normalize_code_ref(src))
    n_test_targets = len(sources)
    n_test_targets = min(n_test_targets, n_find_files) if n_find_files > 0 else 0

    n_clean = max(total_files - n_find_files, 0)
    n_gap = max(n_find_files - n_test_targets, 0)

    W, H = 1200, 520
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ft = _load_font(26)
    fs = _load_font(18)
    fsm = _load_font(16)

    draw_title(d, W, f"Closed-loop Funnel (Review → TestGen)  {shorten_repo_id(repo.repo_id)}", 18, ft)

    # big bars
    x0, y0 = 120, 140
    full_w = 920
    bar_h = 34
    gap = 22

    def bar(y, label, val, frac, color):
        d.text((x0, y - 26), f"{label}: {val}", fill=(30, 30, 30), font=fs)
        d.rounded_rectangle([x0, y, x0 + full_w, y + bar_h], radius=12, fill=(245, 245, 245))
        d.rounded_rectangle([x0, y, x0 + int(full_w * frac), y + bar_h], radius=12, fill=color)
        d.text((x0 + full_w + 18, y + 6), f"{frac*100:.1f}%", fill=(70, 70, 70), font=fsm)

    frac_find = (n_find_files / total_files) if total_files > 0 else 0.0
    frac_test = (n_test_targets / max(n_find_files, 1)) if n_find_files > 0 else 0.0
    frac_gap = (n_gap / max(n_find_files, 1)) if n_find_files > 0 else 0.0
    frac_clean = (n_clean / total_files) if total_files > 0 else 0.0

    bar(y0 + 0*(bar_h+gap), "Files scanned", total_files, 1.0, (31, 119, 180))
    bar(y0 + 1*(bar_h+gap), "Files with findings", n_find_files, frac_find, (255, 127, 14))
    bar(y0 + 2*(bar_h+gap), "Findings covered by tests", n_test_targets, frac_test, (44, 160, 44))
    bar(y0 + 3*(bar_h+gap), "Findings NOT covered", n_gap, frac_gap, (214, 39, 40))
    bar(y0 + 4*(bar_h+gap), "Files clean", n_clean, frac_clean, (160, 160, 160))

    d.text((120, 460), "Goal: raise “covered by tests” and reduce “NOT covered”.", fill=(80, 80, 80), font=fsm)

    save_img(img, out_path)


# ----------------------------
# Plot 4: Hotspot treemap (aggregated & clean)
# ----------------------------
def make_treemap_hotspots(repo: RepoRun, out_path: str, depth: int = 2, top_n: int = 14):
    agg_count: Dict[str, int] = {}
    agg_sevsum: Dict[str, float] = {}

    for f in repo.findings:
        file_ = f.get("file") or ""
        if not file_ or is_generated_artifact_path(file_):
            continue
        file_ = normalize_code_ref(file_)
        sev = (f.get("severity") or "low").lower()
        w = SEV_WEIGHT.get(sev, 1)

        grp = dir_prefix(file_, depth=depth)
        agg_count[grp] = agg_count.get(grp, 0) + 1
        agg_sevsum[grp] = agg_sevsum.get(grp, 0.0) + float(w)

    if not agg_count:
        img = Image.new("RGB", (1200, 360), (255, 255, 255))
        d = ImageDraw.Draw(img)
        draw_title(d, 1200, f"Hotspot Treemap  {shorten_repo_id(repo.repo_id)}", 20, _load_font(28))
        d.text((40, 160), "No effective findings (after filtering artifacts).", fill=(40, 40, 40), font=_load_font(18))
        save_img(img, out_path)
        return

    items = sorted(agg_count.items(), key=lambda x: x[1], reverse=True)

    # top_n + Other
    top = items[:top_n]
    other = items[top_n:]
    if other:
        other_cnt = sum(v for _, v in other)
        top.append(("Other", other_cnt))

    labels = [k for k, _ in top]
    sizes = [float(v) for _, v in top]
    sev_avg = [agg_sevsum.get(k, 0.0) / max(agg_count.get(k, 1), 1) for k in labels]
    # "Other" severity: average of others if present
    if labels[-1] == "Other" and other:
        oth_sev = sum(agg_sevsum[k] for k, _ in other) / max(sum(agg_count[k] for k, _ in other), 1)
        sev_avg[-1] = oth_sev

    sev_arr = np.array(sev_avg, dtype=float)
    mn, mx = float(sev_arr.min()), float(sev_arr.max())
    sev_norm = (sev_arr - mn) / (mx - mn + 1e-9)

    rects = squarify(sizes, 0, 0, 100, 60)

    W, H = 1400, 720
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ft = _load_font(26)
    fs = _load_font(16)
    fsm = _load_font(14)

    draw_title(d, W, f"Hotspot Treemap (module-level)  {shorten_repo_id(repo.repo_id)}", 18, ft)
    d.text((40, 60), f"Grouping depth={depth}, top={top_n} (+Other). size=count, color=avg severity.",
           fill=(90, 90, 90), font=fs)

    ox, oy = 40, 110
    tw, th = 1320, 520

    for (x, y, dx, dy), lab, cnt, c in zip(rects, labels, sizes, sev_norm):
        px0 = ox + int(tw * (x / 100.0))
        py0 = oy + int(th * (y / 60.0))
        px1 = ox + int(tw * ((x + dx) / 100.0))
        py1 = oy + int(th * ((y + dy) / 60.0))
        color = viridis(float(c))

        d.rounded_rectangle([px0, py0, px1, py1], radius=16, fill=color, outline=(255, 255, 255), width=3)

        w = px1 - px0
        h = py1 - py0
        if w >= 220 and h >= 70:
            text = f"{ellipsize(lab, 42)}\n{int(cnt)} findings"
            draw_wrapped(d, (px0 + 14, py0 + 14), text, fsm, (20, 20, 20), max_w=w - 28, line_h=18, max_lines=3)

    # color legend
    lx, ly = 40, 650
    d.text((lx, ly), "Color: low severity → high severity", fill=(80, 80, 80), font=fs)
    ly += 26
    barw, barh = 320, 14
    for x in range(barw):
        t = x / max(barw - 1, 1)
        d.line([(lx + x, ly), (lx + x, ly + barh)], fill=viridis(t))
    d.rectangle([lx, ly, lx + barw, ly + barh], outline=(0, 0, 0))
    d.text((lx + barw + 14, ly - 2), f"min={mn:.2f}, max={mx:.2f}", fill=(80, 80, 80), font=fs)

    save_img(img, out_path)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--git_repo_dir", default="Git_repo")
    ap.add_argument("--out_dir", default="viz")
    ap.add_argument("--repo_focus", default="", help="focus repo_id for closed-loop + treemap")
    ap.add_argument("--top_rules", type=int, default=10)
    ap.add_argument("--normalize", choices=["none", "files", "kloc"], default="files")
    ap.add_argument("--treemap_depth", type=int, default=2)
    ap.add_argument("--include_root_run", action="store_true", help="include reports_dir root run if exists")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    runs = load_runs(args.reports_dir, args.git_repo_dir, include_root_run=args.include_root_run)
    print(f"[viz] runs={len(runs)}; out_dir={os.path.abspath(args.out_dir)}")

    # 1) heatmap
    make_heatmap_repo_rule(
        runs,
        out_path=os.path.join(args.out_dir, "01_repo_rule_heatmap.png"),
        top_rules=args.top_rules,
        normalize=args.normalize,
    )
    print("[viz] 01 heatmap done")

    # 2) risk bars
    make_risk_bars(
        runs,
        out_path=os.path.join(args.out_dir, "02_repo_risk_bars.png"),
        max_repos=6,
    )
    print("[viz] 02 risk bars done")

    # focus repo for funnel/treemap
    focus: Optional[RepoRun] = None
    repo_ids = {r.repo_id for r in runs}
    if args.repo_focus and args.repo_focus in repo_ids:
        focus = next(r for r in runs if r.repo_id == args.repo_focus)
    else:
        # pick the repo with most effective findings
        def eff_findings(rr: RepoRun) -> int:
            return sum(
                1 for f in rr.findings
                if f.get("file") and not is_generated_artifact_path(f.get("file", "")) and is_ds_finding(f)
            )
        focus = max(runs, key=eff_findings)

    # 3) closed-loop funnel
    make_closed_loop_funnel(
        focus,
        out_path=os.path.join(args.out_dir, f"03_closed_loop_funnel__{_safe_name(focus.repo_id)}.png"),
    )
    print("[viz] 03 closed-loop funnel done")

    # 4) treemap
    make_treemap_hotspots(
        focus,
        out_path=os.path.join(args.out_dir, f"04_hotspot_treemap__{_safe_name(focus.repo_id)}.png"),
        depth=args.treemap_depth,
        top_n=14,
    )
    print("[viz] 04 treemap done")

    print(f"\nDone. Images saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
