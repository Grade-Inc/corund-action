"""What a pytest hook or an autouse fixture CAN DO, decided from its body — the ELEVENTH cycle's
decidability rule for test infrastructure (verifier 10, V10-F1; the owner's charter, item 3).

Until this cycle C2 reported EVERY outcome-affecting `pytest_*` hook and every autouse fixture a
PR added as a tier-B FINDING, and the receipt then said "a silent skip is a failure" about a
conftest whose whole body was

    def pytest_configure(config):
        config.addinivalue_line("markers", "slow: marks tests as slow to run")

— the pytest documentation's own idiom for registering a marker — and about `pytest_sessionstart`
printing a banner, `pytest_collection_modifyitems` sorting items, an autouse fixture seeding
`random`, and pytest-django's `def _db(db): pass`. Each was an ACCUSATION of mainstream code, and
the remedy on offer was "allowlist the conftest path", which the owner refused as a remedy for an
honest shape.

The rule now: **C2 says FAILED only when it can PROVE a silencing.** Where the body is decidably
harmless it is not a finding; where the body's effect cannot be determined from this file it is an
OBSERVATION that names what was seen and that nothing was concluded. Either way the receipt says
which. This is the V9-F1 pattern (decide what is visible, observe what is not) applied to the
conftest / plugin surface, and it changes NOTHING about C1: PR-authored infrastructure still runs
during the reverted phase, so every one of these — proven, benign or undecided — remains a C1
contamination (the owner's 2026-09-05 invariant, a safety refusal worded as one).

Three answers, and the words on the receipt for each:

  SILENCING   the body PROVABLY silences, drops, deselects or decides the outcome of a test:
              a skip/xfail raised or applied; a hook parameter's item list emptied, filtered,
              deleted from or popped; a value returned from a hook whose return value REPLACES
              pytest's default (pytest_pyfunc_call, pytest_runtest_protocol, pytest_ignore_collect,
              pytest_pycollect_makeitem, ...); a report's outcome/longrepr/excinfo assigned or
              forced; `session.shouldstop`/`exitstatus` assigned; an `addinivalue_line` that
              changes what is collected (`addopts`, `python_files`, `testpaths`, ...); a plugin
              registered through `config.pluginmanager`; a module attribute built from a string that
              folds to `collect_ignore`/`pytestmark`/`__test__`; a `raise` inside a hook that
              decides a test's outcome; the code under test PATCHED for every test by an autouse
              fixture (monkeypatch / mock.patch / setattr aimed at a module this PR ships).
              -> a FINDING, the same kinds as before, with the proof in the receipt line.
  BENIGN      every statement is in a CLOSED vocabulary that cannot touch collection or outcomes:
              marker registration, printing and logging, sorting/reversing the item list, seeding
              `random`, reading options, resetting an object that is not a hook parameter, `pass`,
              `yield`, a bare `return`.
              -> an OBSERVATION ("decided: ..."), never a finding.
  UNDECIDED   anything outside that vocabulary: a call to a function defined outside this file,
              file I/O, a computed attribute, a nested def, `sys.modules`, a patch of a module this
              PR does not ship (a third-party or stdlib name — mocking `requests`, freezing time).
              -> an OBSERVATION ("not decided: ..."), never a finding.

The vocabulary is a POSITIVE list. A shape nobody wrote down is UNDECIDED, and an undecided shape
is an observation — so the failure mode of an incomplete list is a missing sentence on a receipt,
never an accusation and never a proven silencing waved through: the silencing checks run FIRST,
over the whole body, before any benign verdict is possible.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from typing import Callable

# Hooks whose RETURN VALUE replaces pytest's default behaviour ("firstresult" hooks, plus the two
# whose truthy return short-circuits the protocol). A non-None return from one of these decides
# whether a test is collected or run at all.
RETURN_DECIDES_HOOKS: frozenset[str] = frozenset({
    "pytest_pyfunc_call", "pytest_runtest_protocol", "pytest_ignore_collect", "pytest_pycollect_makeitem",
    "pytest_pycollect_makemodule", "pytest_collect_file", "pytest_collect_directory", "pytest_make_collect_report",
    "pytest_runtest_makereport", "pytest_report_teststatus", "pytest_collection", "pytest_runtestloop",
    "pytest_cmdline_main", "pytest_load_initial_conftests", "pytest_fixture_setup",
})
# Hooks in which a `raise` decides the outcome of the test (or collection) the hook is about.
RAISE_DECIDES_HOOKS_PREFIXES: tuple[str, ...] = (
    "pytest_runtest_", "pytest_pyfunc_call", "pytest_collection", "pytest_pycollect_", "pytest_collect_",
    "pytest_make_collect_report", "pytest_itemcollected", "pytest_ignore_collect", "pytest_deselected",
    "pytest_report_teststatus", "pytest_runtestloop", "pytest_fixture_setup", "pytest_exception_interact",
)
# ini keys whose value changes WHAT IS COLLECTED or how it is run.
COLLECTION_INI_KEYS: frozenset[str] = frozenset({
    "addopts", "python_files", "python_functions", "python_classes", "testpaths", "norecursedirs",
    "collect_ignore_glob", "usefixtures", "required_plugins", "minversion",
})
# Module attributes that, assigned dynamically, silence or decollect.
DANGEROUS_MODULE_ATTRS: frozenset[str] = frozenset({"collect_ignore", "collect_ignore_glob", "pytestmark", "__test__", "pytest_plugins"})
# Attributes whose ASSIGNMENT decides an outcome.
OUTCOME_ATTRS: frozenset[str] = frozenset({
    "outcome", "longrepr", "wasxfail", "excinfo", "exitstatus", "shouldstop", "shouldfail", "testsfailed",
    "__test__", "skipped", "passed", "failed", "when",
})
# Skip / exit machinery, by canonical name (the resolver's spelling) or bare attribute.
SKIP_CANONS: frozenset[str] = frozenset({
    "pytest.skip", "pytest.xfail", "pytest.importorskip", "pytest.exit", "pytest.skip.Exception",
    "unittest.SkipTest", "unittest.case.SkipTest", "_pytest.outcomes.Skipped", "_pytest.outcomes.skip",
    "_pytest.outcomes.xfail", "_pytest.outcomes.exit", "_pytest.outcomes.Exit", "sys.exit", "os._exit",
    "os.abort", "os.kill", "builtins.exit", "builtins.quit",
})
SKIP_MARK_PREFIXES: tuple[str, ...] = ("pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail")
# Method names that DROP from a list (a hook parameter's items, args, session.items).
DROPPING_METHODS: frozenset[str] = frozenset({"remove", "pop", "clear", "discard"})
KEEPING_METHODS: frozenset[str] = frozenset({"sort", "reverse", "append", "extend", "insert", "index", "count", "copy"})
# The hook parameters whose mutation is a deselection / an argument injection.
LIST_PARAMS: frozenset[str] = frozenset({"items", "args", "collectors"})

# The BENIGN vocabulary of callees: callables that cannot touch collection or outcomes. Canonical
# names (module.attr) and bare builtins. A callee not here is UNDECIDED, never benign.
BENIGN_CALLEES: frozenset[str] = frozenset({
    "print", "len", "str", "repr", "int", "float", "bool", "list", "tuple", "dict", "set", "frozenset", "sorted",
    "reversed", "enumerate", "zip", "range", "min", "max", "sum", "abs", "round", "isinstance", "hasattr", "format",
    "any", "all", "id", "hash", "type", "iter", "next", "map", "filter", "divmod", "pow", "ord", "chr", "bin", "hex",
    "oct", "callable", "issubclass", "getattr",
    "random.seed", "random.shuffle", "random.getstate", "random.setstate", "random.random", "random.randint",
    "random.choice", "random.Random",
    "time.time", "time.monotonic", "time.perf_counter", "time.sleep", "time.strftime", "time.gmtime", "time.localtime",
    "datetime.datetime.now", "datetime.datetime.utcnow", "datetime.date.today", "datetime.timedelta",
    "os.getenv", "os.getcwd", "os.getpid", "os.environ.get", "os.environ.setdefault", "os.environ.pop",
    "os.environ.update", "os.environ.copy", "os.path.join", "os.path.dirname", "os.path.abspath", "os.path.exists",
    "os.path.isdir", "os.path.isfile", "os.path.basename", "os.path.realpath", "os.path.expanduser",
    "pathlib.Path", "pathlib.Path.exists", "pathlib.Path.is_dir", "pathlib.Path.is_file", "pathlib.Path.resolve",
    "pathlib.Path.joinpath", "pathlib.Path.read_text", "pathlib.Path.read_bytes",
    "json.loads", "json.dumps", "json.load", "math.floor", "math.ceil", "math.sqrt", "re.compile", "re.match",
    "re.search", "re.sub", "re.fullmatch", "copy.copy", "copy.deepcopy", "itertools.chain", "collections.defaultdict",
    "collections.OrderedDict", "collections.Counter", "functools.partial", "textwrap.dedent", "uuid.uuid4",
    "warnings.warn", "warnings.simplefilter", "warnings.filterwarnings", "warnings.resetwarnings", "warnings.catch_warnings",
    "logging.getLogger", "logging.basicConfig", "logging.disable", "logging.debug", "logging.info", "logging.warning",
    "logging.error", "logging.exception", "logging.critical", "logging.log", "logging.captureWarnings",
    "faker.Faker", "freezegun.freeze_time",
    # pytest's own read-only / registration surface
    "pytest.fixture", "pytest.hookimpl", "pytest.mark", "pytest.param", "pytest.approx", "pytest.warns", "pytest.raises",
    "pytest.deprecated_call", "pytest.register_assert_rewrite",
})
# Methods that are benign on ANY receiver that is not a hook parameter, sys.modules or builtins:
# reading, logging, and resetting an object between tests.
BENIGN_METHODS: frozenset[str] = frozenset({
    "get", "keys", "values", "items", "copy", "format", "join", "split", "strip", "lower", "upper", "startswith",
    "endswith", "replace", "encode", "decode", "read", "readline", "readlines", "info", "debug", "warning", "warn",
    "error", "exception", "critical", "log", "setLevel", "addHandler", "removeHandler", "write", "write_line",
    "write_sep", "line", "section", "ensure_newline", "flush", "clear", "reset", "reset_mock", "update", "setdefault",
    "add", "append", "extend", "insert", "sort", "reverse", "seed", "shuffle", "close", "disable", "enable",
    "getoption", "getini", "getvalue", "getgroup", "addoption", "addini", "get_closest_marker", "iter_markers",
    "listnames", "listchain", "getfixturevalue", "addinivalue_line", "parametrize", "seed_instance", "set_seed",
    "isoformat", "strftime", "total_seconds", "exists", "is_dir", "is_file", "resolve", "joinpath", "read_text",
    "read_bytes", "as_posix", "relative_to", "with_suffix", "with_name", "glob", "rglob", "iterdir", "mkdir",
    "touch", "count", "index", "pop", "discard", "remove", "invalidate_caches", "cache_clear", "configure",
    "setup", "teardown", "commit", "rollback", "flush_all", "flushdb", "delete", "truncate", "begin", "connect",
})
# Receivers on which even a "benign" method is NOT benign: interpreter and pytest internals.
SENSITIVE_RECEIVER_PREFIXES: tuple[str, ...] = (
    "sys.", "builtins", "pytest.", "_pytest", "importlib", "gc.", "ctypes", "signal", "threading", "subprocess",
    "multiprocessing", "shutil", "os.", "io.", "tempfile", "socket", "atexit", "inspect", "marshal", "pickle",
    "pluggy", "config.pluginmanager", "config.hook", "session.", "item.session", "request.session",
    "request.node", "request.config",
)
PATCHERS: frozenset[str] = frozenset({
    "monkeypatch.setattr", "monkeypatch.delattr", "monkeypatch.setitem", "monkeypatch.delitem",
    "mock.patch", "mock.patch.object", "mock.patch.dict", "unittest.mock.patch", "unittest.mock.patch.object",
    "unittest.mock.patch.dict", "mocker.patch", "mocker.patch.object", "mocker.patch.dict", "setattr", "delattr",
    "builtins.setattr", "builtins.delattr",
})
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | {"pytest", "_pytest", "pluggy", "py"}

MAX_HELPER_DEPTH = 3


@dataclass
class Decision:
    verdict: str                          # "silencing" | "benign" | "undecided"
    reasons: list[str] = field(default_factory=list)

    @property
    def why(self) -> str:
        return "; ".join(dict.fromkeys(self.reasons)) or self.verdict


def _hook_decides_on_raise(hook_name: str | None) -> bool:
    return bool(hook_name) and any(hook_name.startswith(p) for p in RAISE_DECIDES_HOOKS_PREFIXES)


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if not isinstance(node, ast.Call) else node.func
    return node.id if isinstance(node, ast.Name) else None


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def decide(body_owner: ast.AST, *, hook_name: str | None, is_fixture: bool, canon: Callable[[ast.AST], str | None],
           const: Callable[[ast.AST], object], first_party: frozenset[str], helpers: dict[str, ast.AST],
           _depth: int = 0) -> Decision:
    """Decide one hook / fixture / lambda body. `canon` and `const` are the resolver's canonical-name
    and constant-fold services; `first_party` the top-level module names this PR ships; `helpers`
    the module-level defs of the same file (same-file calls are followed, MAX_HELPER_DEPTH deep)."""
    if isinstance(body_owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
        params = {a.arg for a in body_owner.args.args + body_owner.args.posonlyargs + body_owner.args.kwonlyargs}
        if body_owner.args.vararg:
            params.add(body_owner.args.vararg.arg)
        stmts = list(body_owner.body)
        decorators = list(body_owner.decorator_list)
    elif isinstance(body_owner, ast.Lambda):
        params = {a.arg for a in body_owner.args.args}
        stmts = [ast.Expr(value=body_owner.body)]
        decorators = []
    else:
        return Decision("undecided", [f"a hook bound to a {type(body_owner).__name__}, not a def or a lambda"])

    silencing: list[str] = []
    undecided: list[str] = []
    list_params = {p for p in params if p in LIST_PARAMS}

    def cname(node: ast.AST) -> str:
        try:
            return canon(node) or _dotted(node) or ""
        except Exception:  # noqa: BLE001 — a resolver failure is an undecided read, never a crash
            return _dotted(node) or ""

    def is_first_party_target(target: ast.AST | None) -> bool | None:
        """True: a module this PR ships. False: stdlib / third-party. None: cannot tell."""
        if target is None:
            return None
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            root = target.value.split(".")[0]
        else:
            c = cname(target)
            root = (c or "").split(".")[0]
            if not root:
                return None
        if root in first_party:
            return True
        if root in _STDLIB:
            return False
        return None

    # ---- silencing scan: the whole body, decorators included, BEFORE any benign verdict
    for node in ast.walk(ast.Module(body=stmts + [ast.Expr(value=d) for d in decorators], type_ignores=[])):
        # skip / exit machinery, called or raised or referenced as a mark
        if isinstance(node, ast.Call):
            c = cname(node.func)
            if c in SKIP_CANONS or any(c == p or c.startswith(p + "(") for p in SKIP_MARK_PREFIXES):
                silencing.append(f"calls `{c}`")
            if c.endswith(".add_marker") or c.endswith(".applymarker") or c.endswith(".add_mark"):
                marks = [cname(a) for a in node.args] + [str(a.value) for a in node.args if isinstance(a, ast.Constant)]
                if any(m.startswith(("pytest.mark.skip", "pytest.mark.xfail")) or m in ("skip", "skipif", "xfail") for m in marks):
                    silencing.append(f"applies a skip/xfail marker through `{c}`")
                else:
                    undecided.append(f"applies a marker through `{c}` whose kind this file does not show")
            if c.endswith(".force_result") or c.endswith(".force_exception"):
                silencing.append(f"forces a hook result through `{c}`")
            if c.startswith("config.pluginmanager.") or c.endswith(".pluginmanager.register") or c.endswith(".import_plugin") \
                    or ".pluginmanager.consider_" in c:
                silencing.append(f"registers a plugin through `{c}`")
            if c.startswith("config.hook.") or ".hook.pytest_" in c:
                silencing.append(f"calls a pytest hook directly (`{c}`)")
            if c.endswith(".addinivalue_line") and node.args:
                key = const(node.args[0])
                if key == "markers":
                    pass                                                     # judged in the benign walk
                elif isinstance(key, str) and key in COLLECTION_INI_KEYS:
                    silencing.append(f"`addinivalue_line({key!r}, ...)` changes what is collected or how it runs")
                else:
                    undecided.append(f"`addinivalue_line` with a key this file does not spell as a constant collection key")
            if c in PATCHERS or c.endswith((".setattr", ".delattr", ".setitem", ".delitem")) and c.split(".")[0] in ("monkeypatch", "mocker", "mp", "mocker_"):
                target = node.args[0] if node.args else None
                fp = is_first_party_target(target)
                if fp is True:
                    silencing.append(f"patches the code under test for every test (`{c}` aimed at a module this PR ships)")
                elif fp is False:
                    undecided.append(f"patches a stdlib/third-party name through `{c}` — not the code under test; not decided")
                else:
                    undecided.append(f"patches through `{c}` a target this file does not resolve")
            if c.startswith(("mock.patch", "unittest.mock.patch", "mocker.patch")):
                target = node.args[0] if node.args else None
                fp = is_first_party_target(target)
                if fp is True:
                    silencing.append(f"patches the code under test for every test (`{c}`)")
                else:
                    undecided.append(f"patches through `{c}` a target that is not a module this PR ships")
            if c in ("setattr", "builtins.setattr") and len(node.args) >= 2:
                mod = cname(node.args[0])
                attr = const(node.args[1])
                if (mod.startswith("sys.modules") or mod == "<module>") and isinstance(attr, str) and attr in DANGEROUS_MODULE_ATTRS:
                    silencing.append(f"builds module attribute `{attr}` from a string")
                elif mod.startswith("sys.modules") or mod == "<module>":
                    undecided.append("builds a module attribute whose name this file does not spell as a constant")
            # method calls on a hook's list parameter
            if isinstance(node.func, ast.Attribute):
                recv = _root_name(node.func.value)
                if recv in list_params or cname(node.func.value) in ("session.items", "config.args"):
                    if node.func.attr in DROPPING_METHODS:
                        silencing.append(f"drops from `{cname(node.func.value)}` with `.{node.func.attr}()`")
        # raise
        if isinstance(node, ast.Raise):
            c = cname(node.exc) if node.exc is not None else ""
            if c in SKIP_CANONS or c.endswith(("Skipped", "SkipTest", "skip.Exception", "XFailed")):
                silencing.append(f"raises `{c}`")
            elif _hook_decides_on_raise(hook_name) and not is_fixture:
                silencing.append(f"raises inside `{hook_name}`, which decides the outcome of the test it is about")
            elif is_fixture:
                undecided.append("raises — every test using this fixture errors loudly; not a silence, not decided")
            else:
                undecided.append("raises outside an outcome hook")
        # assignments that decide outcomes / mutate hook parameters / build module attributes
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and t.attr in OUTCOME_ATTRS:
                    silencing.append(f"assigns `{_dotted(t) or t.attr}`, which decides a reported outcome")
                if isinstance(t, ast.Subscript):
                    root = _root_name(t.value)
                    dotted = cname(t.value)
                    if root in list_params or dotted in ("session.items", "config.args"):
                        v = node.value if not isinstance(node, ast.AugAssign) else None
                        if _is_keeping_rewrite(v, root or dotted):
                            pass                                             # items[:] = sorted(items) keeps every item
                        elif _is_empty_or_filter(v):
                            silencing.append(f"rewrites `{dotted or root}[...]` to a subset — a deselection")
                        else:
                            silencing.append(f"rewrites `{dotted or root}[...]` in place — pytest reads that list")
                    if dotted.startswith("sys.modules") or dotted == "<module>":
                        attr = const(t.slice)
                        if isinstance(attr, str) and attr in DANGEROUS_MODULE_ATTRS:
                            silencing.append(f"builds module attribute `{attr}` from a string")
                        else:
                            undecided.append("writes `sys.modules[...]` / a module namespace")
                    if dotted.startswith("globals()") or (isinstance(t.value, ast.Call) and cname(t.value.func) == "globals"):
                        attr = const(t.slice)
                        if isinstance(attr, str) and attr in DANGEROUS_MODULE_ATTRS:
                            silencing.append(f"builds module attribute `{attr}` through globals()")
                        else:
                            undecided.append("writes through globals()")
        if isinstance(node, ast.Delete):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and (_root_name(t.value) in list_params or cname(t.value) in ("session.items", "config.args")):
                    silencing.append(f"deletes from `{cname(t.value)}`")
        # a value returned from a hook whose return value replaces pytest's default
        if isinstance(node, ast.Return) and not is_fixture and hook_name in RETURN_DECIDES_HOOKS:
            if node.value is not None and not (isinstance(node.value, ast.Constant) and node.value.value is None):
                silencing.append(f"returns a value from `{hook_name}`, whose return value replaces pytest's default")

    if silencing:
        return Decision("silencing", silencing)

    # ---- benign walk: every statement must be in the vocabulary, else UNDECIDED
    def benign_expr(e: ast.AST | None) -> bool:
        if e is None or isinstance(e, (ast.Constant, ast.Name)):
            return True
        if isinstance(e, ast.Attribute):
            return benign_expr(e.value)
        if isinstance(e, (ast.BinOp,)):
            return benign_expr(e.left) and benign_expr(e.right)
        if isinstance(e, ast.BoolOp):
            return all(benign_expr(v) for v in e.values)
        if isinstance(e, ast.UnaryOp):
            return benign_expr(e.operand)
        if isinstance(e, ast.Compare):
            return benign_expr(e.left) and all(benign_expr(c) for c in e.comparators)
        if isinstance(e, ast.IfExp):
            return benign_expr(e.test) and benign_expr(e.body) and benign_expr(e.orelse)
        if isinstance(e, ast.JoinedStr):
            return all(benign_expr(v) for v in e.values)
        if isinstance(e, ast.FormattedValue):
            return benign_expr(e.value)
        if isinstance(e, ast.Subscript):
            return benign_expr(e.value) and benign_expr(e.slice)
        if isinstance(e, ast.Slice):
            return benign_expr(e.lower) and benign_expr(e.upper) and benign_expr(e.step)
        if isinstance(e, (ast.List, ast.Tuple, ast.Set)):
            return all(benign_expr(x) for x in e.elts)
        if isinstance(e, ast.Dict):
            return all(benign_expr(x) for x in e.keys if x is not None) and all(benign_expr(x) for x in e.values)
        if isinstance(e, ast.Starred):
            return benign_expr(e.value)
        if isinstance(e, ast.Lambda):
            return benign_expr(e.body)
        if isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return benign_expr(e.elt) and all(benign_expr(g.iter) and all(benign_expr(i) for i in g.ifs) for g in e.generators)
        if isinstance(e, ast.DictComp):
            return benign_expr(e.key) and benign_expr(e.value) and all(benign_expr(g.iter) and all(benign_expr(i) for i in g.ifs) for g in e.generators)
        if isinstance(e, ast.Yield):
            return benign_expr(e.value)
        if isinstance(e, ast.Await):
            return benign_expr(e.value)
        if isinstance(e, ast.Call):
            return benign_call(e)
        undecided.append(f"a `{type(e).__name__}` expression this rule does not classify")
        return False

    def benign_call(c: ast.Call) -> bool:
        name = cname(c.func)
        bare = c.func.id if isinstance(c.func, ast.Name) else None
        args_ok = all(benign_expr(a) for a in c.args) and all(benign_expr(k.value) for k in c.keywords)
        if not args_ok:
            return False
        if name in BENIGN_CALLEES or bare in BENIGN_CALLEES:
            return True
        # a same-file helper: follow it
        if bare in helpers and _depth < MAX_HELPER_DEPTH:
            sub = decide(helpers[bare], hook_name=hook_name, is_fixture=is_fixture, canon=canon, const=const,
                         first_party=first_party, helpers=helpers, _depth=_depth + 1)
            if sub.verdict == "silencing":
                silencing.extend(f"via same-file helper `{bare}`: {r}" for r in sub.reasons)
                return False
            if sub.verdict == "undecided":
                undecided.extend(f"via same-file helper `{bare}`: {r}" for r in sub.reasons)
                return False
            return True
        if isinstance(c.func, ast.Attribute):
            recv = c.func.value
            recv_name = cname(recv)
            method = c.func.attr
            root = _root_name(recv)
            if root in list_params or recv_name in ("session.items", "config.args"):
                return method in KEEPING_METHODS                              # items.sort() keeps every item
            if method == "addinivalue_line":
                key = const(c.args[0]) if c.args else None
                val_ok = len(c.args) >= 2 and _is_constant_str_or_loop_var(c.args[1], const)
                if key == "markers" and val_ok:
                    return True
                undecided.append("`addinivalue_line` whose key or value is not a constant string")
                return False
            if any(recv_name.startswith(p) for p in SENSITIVE_RECEIVER_PREFIXES) or root in ("sys", "builtins", "importlib"):
                undecided.append(f"calls `{recv_name}.{method}()` on an interpreter/pytest-internal receiver")
                return False
            if method in BENIGN_METHODS:
                return True
            undecided.append(f"calls `.{method}()`, a method this rule does not classify")
            return False
        undecided.append(f"calls `{name or bare or '<expr>'}`, which is not in the benign vocabulary and is not defined in this file")
        return False

    def benign_stmt(st: ast.stmt) -> bool:
        if isinstance(st, ast.Pass):
            return True
        if isinstance(st, ast.Expr):
            return benign_expr(st.value)
        if isinstance(st, ast.Assign):
            return all(isinstance(t, (ast.Name, ast.Tuple, ast.List)) or _is_local_or_environ(t, cname) for t in st.targets) \
                and benign_expr(st.value)
        if isinstance(st, ast.AnnAssign):
            return isinstance(st.target, ast.Name) and benign_expr(st.value)
        if isinstance(st, ast.AugAssign):
            return isinstance(st.target, ast.Name) and benign_expr(st.value)
        if isinstance(st, ast.Return):
            if st.value is None or (isinstance(st.value, ast.Constant) and st.value.value is None):
                return True
            if is_fixture or hook_name not in RETURN_DECIDES_HOOKS:
                return benign_expr(st.value)
            return False
        if isinstance(st, (ast.If, ast.While)):
            return benign_expr(st.test) and all(benign_stmt(s) for s in st.body) and all(benign_stmt(s) for s in st.orelse)
        if isinstance(st, (ast.For, ast.AsyncFor)):
            return isinstance(st.target, (ast.Name, ast.Tuple)) and benign_expr(st.iter) \
                and all(benign_stmt(s) for s in st.body) and all(benign_stmt(s) for s in st.orelse)
        if isinstance(st, (ast.With, ast.AsyncWith)):
            return all(benign_expr(i.context_expr) for i in st.items) and all(benign_stmt(s) for s in st.body)
        if isinstance(st, ast.Try):
            return all(benign_stmt(s) for s in st.body) and all(all(benign_stmt(s) for s in h.body) for h in st.handlers) \
                and all(benign_stmt(s) for s in st.orelse) and all(benign_stmt(s) for s in st.finalbody)
        if isinstance(st, ast.Assert):
            return benign_expr(st.test)
        if isinstance(st, ast.Delete):
            return all(isinstance(t, ast.Name) for t in st.targets)
        undecided.append(f"a `{type(st).__name__}` statement this rule does not classify")
        return False

    ok = all(benign_stmt(s) for s in stmts)
    if silencing:                                   # a followed helper proved a silencing
        return Decision("silencing", silencing)
    if ok and not undecided:
        return Decision("benign", [_benign_summary(stmts, hook_name, is_fixture)])
    return Decision("undecided", undecided or ["a body this rule could not classify"])


def _is_local_or_environ(t: ast.AST, cname) -> bool:
    if isinstance(t, ast.Subscript):
        d = cname(t.value)
        return d in ("os.environ",) or d.endswith(".environ")
    return False


def _is_keeping_rewrite(v: ast.AST | None, name: str | None) -> bool:
    """`items[:] = sorted(items, ...)` / `reversed(items)` / `list(items)` keeps every item."""
    if not isinstance(v, ast.Call) or not isinstance(v.func, ast.Name) or v.func.id not in ("sorted", "reversed", "list"):
        return False
    return bool(v.args) and isinstance(v.args[0], ast.Name) and v.args[0].id == name


def _is_empty_or_filter(v: ast.AST | None) -> bool:
    if isinstance(v, (ast.List, ast.Tuple)) and not v.elts:
        return True
    if isinstance(v, ast.ListComp) and any(g.ifs for g in v.generators):
        return True
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "filter":
        return True
    return False


def _is_constant_str_or_loop_var(node: ast.AST, const) -> bool:
    v = const(node)
    if isinstance(v, str):
        return True
    return isinstance(node, ast.Name)          # a loop variable over a literal list — the loop's iter is checked by benign_expr


def _benign_summary(stmts: list[ast.stmt], hook_name: str | None, is_fixture: bool) -> str:
    kinds: list[str] = []
    for node in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "addinivalue_line":
            kinds.append("marker registration only (`addinivalue_line('markers', ...)`)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("sort", "reverse", "shuffle"):
            kinds.append("reorders the item list without dropping any")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            kinds.append("prints")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("seed",):
            kinds.append("seeds a random source")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("clear", "reset", "reset_mock", "cache_clear"):
            kinds.append("resets an object between tests")
    if not kinds:
        kinds.append("a body of pass / yield / reads only" if not is_fixture else "a fixture body of pass / yield / reads only")
    return "body is " + ", ".join(dict.fromkeys(kinds)) + " — registers no plugin, drops no item, decides no outcome"
