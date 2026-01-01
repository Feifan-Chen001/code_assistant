from __future__ import annotations
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set

from .types import ReviewFinding

TOOL = "ds-rule"

SKLEARN_RANDOM_CALLS = {
    "train_test_split",
    "KFold",
    "StratifiedKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "KMeans",
    "MiniBatchKMeans",
}

SCALER_CLASSES = {"StandardScaler", "OneHotEncoder"}
PIPELINE_CALLS = {"Pipeline", "make_pipeline"}

RANDOM_FUNC_NAMES = {
    "random",
    "randint",
    "rand",
    "randn",
    "choice",
    "shuffle",
    "permutation",
    "normal",
    "uniform",
}


def scan_file_ds(path: Path, repo_root: Path) -> List[ReviewFinding]:
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    rel = str(path.relative_to(repo_root)).replace("\\", "/")
    return scan_source_ds(src, rel)


def scan_source_ds(source: str, rel_path: str) -> List[ReviewFinding]:
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    visitor = _DSVisitor(rel_path)
    visitor.visit(tree)
    return visitor.finalize()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _attr_chain(node: ast.AST) -> str:
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _assigned_names(target: ast.AST) -> Set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: Set[str] = set()
        for el in target.elts:
            out.update(_assigned_names(el))
        return out
    return set()


def _has_kw(node: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in node.keywords if kw.arg is not None)


def _is_chained_subscript(node: ast.AST) -> bool:
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript)


class _DSVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.findings: List[ReviewFinding] = []
        self.numpy_aliases: Set[str] = set()
        self.numpy_random_aliases: Set[str] = set()
        self.random_module_aliases: Set[str] = set()
        self.random_func_aliases: Set[str] = set()
        self.numpy_random_func_aliases: Set[str] = set()

        self.has_seed = False
        self.random_usage_lines: List[int] = []
        self.train_test_split_lines: List[int] = []
        self.fit_transform_lines: List[int] = []

        self.pipeline_used = False
        self.scaler_vars: Set[str] = set()
        self.scaler_usage_nodes: Dict[str, ast.Call] = {}

    def _add(self, rule: str, severity: str, message: str, node: Optional[ast.AST]):
        self.findings.append(
            ReviewFinding(
                tool=TOOL,
                rule=rule,
                severity=severity,
                message=message,
                file=self.rel_path,
                line=getattr(node, "lineno", None),
                col=getattr(node, "col_offset", None),
            )
        )

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            if name == "numpy":
                self.numpy_aliases.add(asname)
            if name == "numpy.random":
                self.numpy_random_aliases.add(asname)
            if name == "random":
                self.random_module_aliases.add(asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            if mod == "numpy":
                if name == "random":
                    self.numpy_random_aliases.add(asname)
            if mod == "numpy.random":
                self.numpy_random_func_aliases.add(asname)
            if mod == "random":
                self.random_func_aliases.add(asname)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if _is_chained_subscript(target):
                self._add(
                    "DS_PANDAS_SETTINGWITHCOPY",
                    "medium",
                    "Chained indexing may trigger SettingWithCopy; use .loc/.iloc.",
                    node,
                )
        if isinstance(node.value, ast.Call):
            name = _call_name(node.value.func)
            if name in SCALER_CLASSES:
                for t in node.targets:
                    self.scaler_vars.update(_assigned_names(t))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if _is_chained_subscript(node.target):
            self._add(
                "DS_PANDAS_SETTINGWITHCOPY",
                "medium",
                "Chained indexing may trigger SettingWithCopy; use .loc/.iloc.",
                node,
            )
        if isinstance(node.value, ast.Call):
            name = _call_name(node.value.func)
            if name in SCALER_CLASSES:
                self.scaler_vars.update(_assigned_names(node.target))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if _is_chained_subscript(node.target):
            self._add(
                "DS_PANDAS_SETTINGWITHCOPY",
                "medium",
                "Chained indexing may trigger SettingWithCopy; use .loc/.iloc.",
                node,
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        chain = _attr_chain(node.func)
        name = _call_name(node.func)

        if name in PIPELINE_CALLS or chain.endswith(".Pipeline") or chain.endswith(".make_pipeline"):
            self.pipeline_used = True

        if name == "train_test_split":
            self.train_test_split_lines.append(getattr(node, "lineno", 0))

        if name == "fit_transform":
            self.fit_transform_lines.append(getattr(node, "lineno", 0))

        if name in SKLEARN_RANDOM_CALLS and not _has_kw(node, "random_state"):
            self._add(
                "DS_SKLEARN_RANDOM_STATE",
                "medium",
                "Stochastic sklearn component without random_state may be non-reproducible.",
                node,
            )

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            var = node.func.value.id
            if var in self.scaler_vars and node.func.attr in {"fit", "transform", "fit_transform"}:
                self.scaler_usage_nodes.setdefault(var, node)

            if node.func.attr == "iterrows":
                self._add(
                    "DS_PANDAS_ITERROWS",
                    "low",
                    "pandas.iterrows() is slow; consider vectorization.",
                    node,
                )

            if node.func.attr == "apply" and _is_apply_axis1(node):
                self._add(
                    "DS_PANDAS_APPLY_AXIS1",
                    "low",
                    "pandas.apply(axis=1) is slow; consider vectorization.",
                    node,
                )

        if self._is_seed_call(chain, name, node):
            self.has_seed = True

        if self._is_random_usage(chain, name):
            self.random_usage_lines.append(getattr(node, "lineno", 0))

        self.generic_visit(node)

    def _is_seed_call(self, chain: str, name: str, node: ast.Call) -> bool:
        if name == "seed":
            if chain.startswith("random.") or any(chain.startswith(f"{a}.") for a in self.random_module_aliases):
                return True
            if any(chain.startswith(f"{a}.random.") for a in self.numpy_aliases):
                return True
            if any(chain.startswith(f"{a}.") for a in self.numpy_random_aliases):
                return True
        if name in {"default_rng", "RandomState"}:
            if node.args or _has_kw(node, "seed"):
                return True
        return False

    def _is_random_usage(self, chain: str, name: str) -> bool:
        if name in self.random_func_aliases or name in self.numpy_random_func_aliases:
            return True
        if name in RANDOM_FUNC_NAMES and (
            chain.startswith("random.")
            or any(chain.startswith(f"{a}.") for a in self.random_module_aliases)
        ):
            return True
        if any(chain.startswith(f"{a}.random.") for a in self.numpy_aliases):
            return True
        if any(chain.startswith(f"{a}.") for a in self.numpy_random_aliases):
            return True
        return False

    def finalize(self) -> List[ReviewFinding]:
        if self.random_usage_lines and not self.has_seed:
            line = min(self.random_usage_lines) if self.random_usage_lines else None
            self._add(
                "DS_RANDOM_SEED",
                "medium",
                "Randomness detected without an explicit seed.",
                _DummyNode(line),
            )

        if self.train_test_split_lines and self.fit_transform_lines:
            min_split = min(self.train_test_split_lines)
            early_fit = [ln for ln in self.fit_transform_lines if ln < min_split]
            if early_fit:
                self._add(
                    "DS_LEAKAGE_FIT_BEFORE_SPLIT",
                    "high",
                    "fit_transform appears before train_test_split; this may cause leakage.",
                    _DummyNode(min(early_fit)),
                )

        if self.scaler_usage_nodes and not self.pipeline_used:
            for var, node in self.scaler_usage_nodes.items():
                self._add(
                    "DS_PIPELINE_SUGGEST",
                    "medium",
                    f"{var} used without Pipeline; consider sklearn.pipeline.Pipeline.",
                    node,
                )

        return self.findings


def _is_apply_axis1(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "axis" and isinstance(kw.value, ast.Constant):
            if kw.value.value in (1, "columns"):
                return True
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        return node.args[1].value in (1, "columns")
    return False


class _DummyNode:
    def __init__(self, lineno: Optional[int]):
        self.lineno = lineno
        self.col_offset = None
