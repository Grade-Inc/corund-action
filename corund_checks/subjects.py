"""FOURTEENTH CYCLE (verifier 13). The SUBJECT of a test, and which functions a PR changed.

Verifier 13 re-opened a silencing the twelfth cycle had caught: the thirteenth cycle's rename
clearance was GLOBAL. `live` held every test of a new name that fails by assertion on the reverted
tree, anywhere in the PR's modified test files, and `if live:` cleared ANY name-absent base test that
detected the change. There was no linkage between the test being cleared and the witness clearing
it, so `test_mul` (which covered `mul`) was cleared by `test_add_fixed` (which covers `add`), and a
deleted test walked behind an unrelated genuine fix -- the most ordinary silencing shape there is.

THE RULE THIS MODULE SERVES: a name-absent base test may be cleared ONLY by its REPLACEMENT -- a
witness that exercises the SAME SUBJECT the base test did. "Same subject" is decided from RUNTIME
facts, never from names: the Action's pytest plugin (`corund_action/subject_trace.py`) records, per
test, the functions it ENTERED during its call phase, and the two sets are compared on the SAME tree.

What is here:

  * `changed_functions(...)` -- which functions the PR's non-test diff touches, on the head side and
    on the base side, at FUNCTION granularity where the file parses, at FILE granularity where it does
    not (fail closed: an unparsable file counts as wholly changed, so a subject in it is never
    silently dropped). A hunk at module level (a constant, an import, a decorator) marks the WHOLE
    file, because a module-level change reaches every function in it.
  * `Subjects` -- the parsed `subjects` block of the runtime probe: per run, per test id, the sorted
    `path::qualname` list the plugin recorded; or the stated reason it is unavailable. The block is
    REQUIRED of a measured probe (guards are enforced by the core, never left to the caller): a
    measured probe without it is a caller claiming a measurement it did not take.
  * the closed vocabulary of "cannot reach": the exception types a test raises IN ITS OWN FRAME when
    the thing it calls does not exist on that tree -- the runtime signature of a subject the PR
    renamed, removed or re-signed.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field

from .unidiff import parse_unified_diff

# An exception raised in the TEST'S OWN FRAME, in the call phase, of one of these types means the
# name the test called does not exist on that tree: the callee was never entered. A TypeError in the
# own frame is the same fact for a changed signature (`mul() takes 2 positional arguments but 3 were
# given` is raised at the CALL SITE, before the callee runs). Exact type names, no prefix matching.
CANNOT_REACH_TYPES: frozenset[str] = frozenset({"AttributeError", "NameError", "ImportError", "ModuleNotFoundError"})
RESIGNED_TYPES: frozenset[str] = frozenset({"TypeError"})

SUBJECT_STATES: dict[str, str] = {
    "measured": "the subject trace ran",
    "unavailable": "the subject trace is not available for this run, so a renamed test cannot be told from a "
                   "deleted one here and Corund accuses (fail closed)",
}


@dataclass(frozen=True)
class ChangedFunctions:
    """Which functions the PR's NON-TEST diff touches. `head`/`base` are `path::qualname`; a path in
    `wildcard_*` means EVERY function in that file counts (module-level hunk, or unparsable file).
    `added` / `removed` are the functions that exist on one side only (by `path::qualname`)."""
    head: frozenset[str] = frozenset()
    base: frozenset[str] = frozenset()
    wildcard_head: frozenset[str] = frozenset()
    wildcard_base: frozenset[str] = frozenset()
    added: frozenset[str] = frozenset()
    removed: frozenset[str] = frozenset()
    paths: frozenset[str] = frozenset()          # every non-test path the diff touches
    granularity: dict = field(default_factory=dict)   # path -> "function" | "file" (why file-level)
    head_defs: frozenset[str] = frozenset()      # every def the head side of a changed file holds

    @staticmethod
    def _path(fn: str) -> str:
        return fn.split("::", 1)[0]

    def touches_head(self, fn: str) -> bool:
        return fn in self.head or self._path(fn) in self.wildcard_head

    def touches_base(self, fn: str) -> bool:
        return fn in self.base or self._path(fn) in self.wildcard_base

    def in_changed_file(self, fn: str) -> bool:
        return self._path(fn) in self.paths

    def is_added(self, fn: str) -> bool:
        return fn in self.added

    @classmethod
    def file_level(cls, paths) -> "ChangedFunctions":
        """The fallback when the caller has no diff text for the code under test: every function in
        every changed non-test file counts as changed (the strict direction)."""
        ps = frozenset(paths)
        return cls(wildcard_head=ps, wildcard_base=ps, paths=ps, granularity={p: "file (no diff text supplied)" for p in ps})


def _spans(source: str) -> "list[tuple[str, int, int]] | None":
    """(qualname, first line, last line) of every def in `source`, nested ones spelled
    `outer.<locals>.inner` and methods `Class.method` -- the SAME spelling `co_qualname` gives at
    runtime, so a traced frame and a parsed def compare as equal strings. None when unparsable."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    out: list[tuple[str, int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{child.name}"
                start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                out.append((q, start, getattr(child, "end_lineno", child.lineno) or child.lineno))
                walk(child, f"{q}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def _enclosing(spans: list[tuple[str, int, int]], lineno: int) -> str | None:
    """The INNERMOST def whose span holds `lineno`, or None for a module-level line."""
    best = None
    for q, a, b in spans:
        if a <= lineno <= b and (best is None or (b - a) < (best[2] - best[1])):
            best = (q, a, b)
    return best[0] if best else None


def changed_functions(diff_text: str, files_after: Mapping[str, str] | None, files_before: Mapping[str, str] | None,
                      exclude_paths) -> ChangedFunctions:
    """Which functions the PR's non-test diff touches. `exclude_paths` is every TEST path (the probe's
    files plus the tracked suite): a test file is never a subject. Python only; any other changed
    non-test file counts at file level."""
    excl = set(exclude_paths)
    head: set[str] = set()
    base: set[str] = set()
    wc_head: set[str] = set()
    wc_base: set[str] = set()
    paths: set[str] = set()
    gran: dict[str, str] = {}
    head_defs: dict[str, set[str]] = {}
    base_defs: dict[str, set[str]] = {}
    for fd in parse_unified_diff(diff_text or ""):
        if fd.path in excl or (fd.old_path and fd.old_path in excl):
            continue
        if fd.binary or fd.mode_only or fd.text_line_count == 0:
            continue
        paths.add(fd.path)
        if fd.old_path and fd.old_path != fd.path:
            paths.add(fd.old_path)
        if not fd.path.lower().endswith((".py", ".pyi")):
            wc_head.add(fd.path)
            wc_base.add(fd.old_path or fd.path)
            gran[fd.path] = "file (not Python)"
            continue
        # ---- head side ----
        after = files_after.get(fd.path) if isinstance(files_after, Mapping) else None
        spans_h = _spans(after) if isinstance(after, str) and fd.status != "deleted" else None
        if fd.status == "deleted":
            gran[fd.path] = "function"
        elif spans_h is None:
            wc_head.add(fd.path)
            gran[fd.path] = "file (head text absent or unparsable)"
        else:
            head_defs[fd.path] = {f"{fd.path}::{q}" for q, _a, _b in spans_h}
            gran[fd.path] = "function"
            for l in fd.lines:
                if l.side == "+" and l.new_lineno is not None:
                    q = _enclosing(spans_h, l.new_lineno)
                    if q is None:
                        wc_head.add(fd.path)          # a module-level change reaches every function
                    else:
                        head.add(f"{fd.path}::{q}")
        # ---- base side ----
        bpath = fd.old_path or fd.path
        before = files_before.get(bpath) if isinstance(files_before, Mapping) else None
        spans_b = _spans(before) if isinstance(before, str) and fd.status != "added" else None
        if fd.status == "added":
            pass
        elif spans_b is None:
            wc_base.add(bpath)
            if gran.get(fd.path) == "function":
                gran[fd.path] = "function (head) / file (base text absent or unparsable)"
        else:
            base_defs[bpath] = {f"{bpath}::{q}" for q, _a, _b in spans_b}
            for l in fd.lines:
                if l.side == "-" and l.old_lineno is not None:
                    q = _enclosing(spans_b, l.old_lineno)
                    if q is None:
                        wc_base.add(bpath)
                    else:
                        base.add(f"{bpath}::{q}")
    added: set[str] = set()
    removed: set[str] = set()
    for p, defs in head_defs.items():
        added |= defs - base_defs.get(p, set())
    for p, defs in base_defs.items():
        removed |= defs - head_defs.get(p, set())
    all_head_defs: set[str] = set()
    for defs in head_defs.values():
        all_head_defs |= defs
    return ChangedFunctions(frozenset(head), frozenset(base), frozenset(wc_head), frozenset(wc_base),
                            frozenset(added), frozenset(removed), frozenset(paths), gran, frozenset(all_head_defs))


@dataclass(frozen=True)
class Subjects:
    state: str
    detail: str = ""
    runs: Mapping[str, Mapping[str, frozenset[str]]] = field(default_factory=dict)   # run -> test id -> functions

    @property
    def measured(self) -> bool:
        return self.state == "measured"

    def of(self, run: str, test_ids) -> frozenset[str]:
        """The union of what these ids (one function's cases) entered in `run`."""
        m = self.runs.get(run, {})
        out: set[str] = set()
        for tid in test_ids:
            out |= m.get(tid, frozenset())
        return frozenset(out)

    def traced(self, run: str, test_ids) -> bool:
        m = self.runs.get(run, {})
        return any(tid in m for tid in test_ids)


RUN_KEYS = ("base_tests_on_head", "head_tests_on_head", "head_tests_on_base")
# OPTIONAL: the name-absent DETECTING base tests, run once more on the BASE tree where they pass. A
# failing test's trace stops at its first failing assertion, so its head-run subject can be TRUNCATED
# (a compound test that asserts add then mul, failing at add, never enters mul). The base-tree run
# has the whole subject; the functions it adds that still exist as defs on head are unioned in.
REPAIR_KEY = "base_tests_on_base"


def parse_subjects(raw: object) -> Subjects:
    """The `subjects` block of a measured probe. REQUIRED; a missing or malformed block RAISES (->
    CRASHED), and `{"state": "unavailable", "detail": ...}` is the only way to say it was not taken."""
    if raw is None:
        raise ValueError("silencing_probe['subjects'] is missing. A measured probe must say whether the subject trace ran "
                         "(`{state: measured, <run>: {test id: [path::function, ...]}}`) or state why it is unavailable "
                         "(`{state: unavailable, detail: ...}`); without it a renamed test cannot be told from a deleted one")
    if not isinstance(raw, Mapping):
        raise TypeError(f"silencing_probe['subjects'] must be a mapping, got {type(raw).__name__}")
    state = raw.get("state")
    if not isinstance(state, str) or state not in SUBJECT_STATES:
        raise ValueError(f"silencing_probe['subjects']['state'] is {state!r}, which is not one of {tuple(SUBJECT_STATES)}")
    detail = str(raw.get("detail") or "")
    if state != "measured":
        if not detail.strip():
            raise ValueError("silencing_probe['subjects'] is `unavailable` but states no reason; the reason is printed on "
                             "the receipt as the whole explanation of why a rename cannot be cleared here")
        return Subjects(state, detail)
    runs: dict[str, dict[str, frozenset[str]]] = {}
    for key in RUN_KEYS + (REPAIR_KEY,):
        block = raw.get(key)
        if block is None:
            if key == REPAIR_KEY:
                continue
            block = {}
        if not isinstance(block, Mapping):
            raise TypeError(f"silencing_probe['subjects'][{key!r}] must be a mapping of test id -> [path::function]")
        m: dict[str, frozenset[str]] = {}
        for tid, fns in block.items():
            if not isinstance(tid, str) or not isinstance(fns, (list, tuple)) or not all(isinstance(f, str) for f in fns):
                raise TypeError(f"silencing_probe['subjects'][{key!r}] entries must be str -> [str]")
            m[tid] = frozenset(fns)
        runs[key] = m
    return Subjects(state, detail, runs)
