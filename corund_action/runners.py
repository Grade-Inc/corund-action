"""Runner adapters — the launch half of the support matrix (checks/README.md, SUPPORT_MATRIX):
the pytest family and the jest/vitest family. Anything else raises UnsupportedRunner by name and
the caller reports NOT_RUN.

Parsing produces the C1 result vocabulary: {test_id: {"status", "phase", "type", "message"}} with
`fail` reserved for executed-and-failed-by-assertion and `error` for every other exception, plus
{path: message} for files that failed to collect/import.

Classification is by the HEAD TOKEN of the failure text, never by a class-name suffix: pytest's
junit `<failure message>` starts with `assert ...` for a rewritten assert, `AssertionError...` for an
explicit/unittest assertion, `Failed: ...` for pytest.fail / DID NOT RAISE, and `<Type>[: text]` for
any other exception (`pkg.core.NotFound: k`, `StopIteration`, `subprocess.TimeoutExpired: ...`,
`SystemExit: 3`). Only the first three are assertion reds. `[XPASS(strict)] ...` is an xfail
marker outcome, never an assertion.

Reconciliation (`reconcile`): a run's exit code, its report counts and the ids the same tree
collected must agree; the caller turns any disagreement into C1 CRASHED naming it — a forged report
never yields PROVEN. NEW module; the jest/vitest adapters are proven against fixture JSON and a
stand-in command, never against a real jest (no network to install one in this lane).
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET  # nosec B405 — see the B314 note at _parse_junit: self-inflicted blast radius only

SUPPORTED_FAMILIES = ("pytest", "jest", "vitest")

_HEAD_RE = re.compile(r"^([A-Za-z_][\w.]*)(?::|\s|$)")
_E_LINE_RE = re.compile(r"^E\s+(.*)$", re.M)
_XPASS_RE = re.compile(r"^\[XPASS\(strict\)\]")
_ASSERTION_HEADS = frozenset({"AssertionError", "Failed"})
_IMPORT_HEADS = frozenset({"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError"})
_COLLECT_ID_RE = re.compile(r"^([^\s:]+)::(.+)$")            # `<path>::<rest>`: no whitespace before the first `::`
_COLLECT_ERROR_RE = re.compile(r"^ERROR\s+(\S+)")
_COLLECT_SUMMARY_RE = re.compile(r"(\d+) tests? collected|no tests collected|(\d+)/(\d+) tests? collected")
_EXC_RE = re.compile(r"^([A-Za-z_][\w.]*?(?:Error|Exception|Exit|Interrupt|Warning))\b:?")   # jest only, first-token fallback


# The path part may itself contain a colon (`tests/te:st.py`) and may be absolute, so it is matched
# non-greedily up to the LAST `:<line>:` on the frame line rather than forbidden from holding a colon
# — the old `[^\n:]*?` read no origin at all for such a file, and a witness with no origin is refused
# (verifier 4, V4-15). A Windows `C:\x\y.py:3:` now parses for the same reason.
_ORIGIN_CRASH_RE = re.compile(r"^(\S[^\n]*?):(\d+): [A-Za-z_][\w.]*")               # --tb=auto/long: `path:line: Type`
_ORIGIN_SHORT_RE = re.compile(r"^(\S[^\n]*?):(\d+): in \w+")                        # --tb=short frames
_ORIGIN_NATIVE_RE = re.compile(r'^\s*File "([^"]+)", line (\d+)')                   # --tb=native frames


# The FIRST `def <name>(` line of a --tb=auto body is the function pytest entered to reach the raise
# (verified by probe: an own assert shows `def test_x():`; a pytest_runtest_call hook defined in the
# same test file shows `def pytest_runtest_call(item):`; a fixture shows the fixture's own def).
# `>` marks the raising line, so a one-line function body can carry the marker on the def itself.
_ENTRY_DEF_RE = re.compile(r"^\s*>?\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")


def origin_from_body(body: str) -> str | None:
    """The raising frame (`path:line`) of a junit failure body — the LAST frame line pytest printed.
    None when the body carries no frame (--tb=line / --tb=no)."""
    last = None
    for line in (body or "").splitlines():
        for rx in (_ORIGIN_CRASH_RE, _ORIGIN_SHORT_RE, _ORIGIN_NATIVE_RE):
            m = rx.match(line)
            if m:
                last = f"{m.group(1)}:{m.group(2)}"
                break
    return last


def entry_from_body(body: str) -> str | None:
    """The FIRST frame's function name in a junit failure body — what C1's own-frame allowlist
    compares against the test's own function name. None when the body prints no `def` line."""
    for line in (body or "").splitlines():
        m = _ENTRY_DEF_RE.match(line)
        if m:
            return m.group(1)
    return None


class UnsupportedRunner(Exception):
    pass


class ReportError(Exception):
    pass


def _check_family(family: str) -> None:
    if family not in SUPPORTED_FAMILIES:
        raise UnsupportedRunner(f"runner family {family!r} is NOT SUPPORTED (launch support: {', '.join(SUPPORTED_FAMILIES)}); "
                                f"C1 reports NOT_RUN for it")


def detect_family(repo_dir: str) -> str | None:
    for name in ("pytest.ini", "conftest.py", "tox.ini", "setup.cfg", "pyproject.toml"):
        p = os.path.join(repo_dir, name)
        if os.path.exists(p):
            if name in ("pytest.ini", "conftest.py"):
                return "pytest"
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                txt = ""
            if "pytest" in txt:
                return "pytest"
    pj = os.path.join(repo_dir, "package.json")
    if os.path.exists(pj):
        try:
            data = json.load(open(pj, encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        deps = {}
        for k in ("dependencies", "devDependencies"):
            if isinstance(data.get(k), dict):
                deps.update(data[k])
        if "vitest" in deps:
            return "vitest"
        if "jest" in deps:
            return "jest"
    for name in os.listdir(repo_dir) if os.path.isdir(repo_dir) else []:
        if name.startswith("vitest.config."):
            return "vitest"
        if name.startswith("jest.config."):
            return "jest"
    return None


def build_command(family: str, test_command: list[str], report_path: str, test_files: list[str]) -> list[str]:
    _check_family(family)
    if family == "pytest":
        # --continue-on-collection-errors: one file's import failure must not hide the others' results;
        # --tb=auto: the junit body ends with the raising frame (`path:line: Type`) so a witness's origin is attributable
        return [*test_command, "-p", "no:cacheprovider", "--continue-on-collection-errors", "--tb=auto",
                f"--junitxml={report_path}", *test_files]
    if family == "jest":
        return [*test_command, "--ci", "--json", f"--outputFile={report_path}", *test_files]
    return [*test_command, "run", "--reporter=json", f"--outputFile={report_path}", *test_files]


def build_collect_command(family: str, test_command: list[str], test_files: list[str]) -> list[str] | None:
    """The collect-only pre-pass in the SAME tree; None when the family has no test-level listing."""
    _check_family(family)
    if family == "pytest":
        return [*test_command, "-p", "no:cacheprovider", "--continue-on-collection-errors", "--collect-only", "-q", *test_files]
    return None


# ----------------------------------------------------------------------------- classification

def classify_failure_text(first_line: str) -> tuple[str, str]:
    """(status, type) from the first line of a failure's text. `fail` only for an assertion head."""
    s = (first_line or "").strip()
    if not s:
        return "error", "UnclassifiedFailure"
    if _XPASS_RE.match(s):
        return "error", "XPassStrict"
    if s.startswith("assert ") or s == "assert" or s.startswith("assert\t"):
        return "fail", "AssertionError"
    m = _HEAD_RE.match(s)
    if not m:
        return "error", "UnclassifiedFailure"
    head = m.group(1)
    if head in _ASSERTION_HEADS:
        return "fail", head
    return "error", head


def _first_line(s: str) -> str:
    for line in (s or "").splitlines():
        if line.strip():
            return line.strip()[:300]
    return ""


def _last_e_line(body: str) -> str:
    m = _E_LINE_RE.findall(body or "")
    return m[-1].strip() if m else ""


def _classify(message: str, body: str, kind: str) -> tuple[str, str, str]:
    """(status, phase, type) from a junit <failure>/<error> element's message and text."""
    msg = (message or "").strip()
    text = body or ""
    if kind == "error":
        low = msg.lower()
        if "collection" in low:
            phase = "collection"
        elif "setup" in low:
            phase = "setup"
        elif "teardown" in low:
            phase = "teardown"
        else:
            phase = "call"
        # the traceback's last `E ` line names the real exception; pytest's message is generic
        # ("test setup failure") or wraps it ('failed on setup with "AttributeError: ..."')
        last = _last_e_line(text)
        inner = re.search(r'with "(.*?)"?\s*$', msg, re.S)
        for cand in (last, _first_line(inner.group(1)) if inner else "", msg):
            if cand:
                _, typ = classify_failure_text(cand)
                if typ != "UnclassifiedFailure" and typ.lower() not in ("test", "failed", "collection"):
                    break
        else:
            typ = "UnclassifiedFailure"
        return "error", phase, typ            # an <error> is never an assertion red, whatever its type
    first = _first_line(msg)
    if not first:
        first = _last_e_line(text)
    # pytest's <failure message> is the exception's own text: its head token is authoritative. (An earlier
    # cross-check against the traceback's last `E` line reclassified honest multi-line assertion messages
    # such as numpy's `y: array([2])` — verifier 1; removed.)
    status, typ = classify_failure_text(first)
    return status, "call", typ


def _path_from_classname(classname: str, known_files: set[str]) -> str:
    """junit classname (`tests.v1.0.test_add.TestX`, rootdir-relative) -> `<known file>::TestX`. A known file's dotted
    form keeps dots inside directory names, and a rootdir below the repo root is matched by suffix."""
    parts = [p for p in classname.split(".") if p]
    if not parts:
        return classname
    # SIXTH CYCLE (verifier 5, F1, end-to-end repro V5E16). This map used to be built with
    # `f[:-3].replace("/", ".")` — the FORWARD slash only. But pytest's own junit `classname` is lossy
    # in exactly the same way and it does not stop at `/`: a test defined in a file literally named
    # `tests/test_x\y.py` is reported as `classname="tests.test_x.y"`, which is ALSO the dotted form of
    # the entirely different real module `tests/test_x/y.py`. Measured, not assumed — that XML is what
    # pytest writes on this platform. With only the `/` form in the map the crafted file's dotted name
    # was `tests.test_x\y`, the classname matched the REAL module uniquely, and the adapter handed C1
    # an id naming a file the test was not in: a helper's assertion affirmed as the test's own, with a
    # fully green receipt. Both separators are now folded to `.`, so the two files COLLIDE here and the
    # ambiguity refusal below fires, which is the honest answer — the classname genuinely cannot tell
    # them apart. A pre-existing collision of the same kind (`a.b/c.py` vs `a/b/c.py`) is caught too.
    dotted: list[tuple[str, str]] = [(re.sub(r"[\\/]", ".", f[:-3]), f) for f in known_files if f.endswith(".py")]
    for i in range(len(parts), 0, -1):
        mod = ".".join(parts[:i])
        hits = sorted({f for d, f in dotted if d == mod or d.endswith("." + mod)})
        if len(hits) > 1:
            # SIXTH CYCLE (verifier 5, F1b). A dotted classname is LOSSY: `a.b.test_m` is the dotted
            # form of `a/b/test_m.py` AND the rootdir-relative suffix of `x/a/b/test_m.py`, so this map
            # is many-to-one. It used to resolve that by `hits.sort(key=len)` — silently pick the
            # SHORTEST — which is a guess presented as a fact on the exact path that decides which file
            # C1 compares a raising frame against. There is no information here to break the tie with,
            # so the id keeps the classname's own dotted shape: it will not match any tracked file, the
            # own-frame allowlist will refuse to affirm, and the result is UNPROVEN rather than a
            # PROVEN built on a coin flip. FAILS CLOSED, and the loss is confined to a repo that really
            # does track two test files with the same dotted name.
            #
            # The id keeps the classname VERBATIM (a dotted name, not a path), rather than falling
            # through to the `"/".join(parts) + ".py"` reconstruction below — that reconstruction can
            # rebuild one of the very candidates it just refused to choose between, which would hand
            # back the same guess through a different door.
            return classname
        if hits:
            rest = parts[i:]
            return hits[0] + ("::" + "::".join(rest) if rest else "")
    return "/".join(parts) + ".py"


def repo_relative(path: str, known_files: set[str]) -> str:
    """A rootdir-relative path (`tests/test_x.py` under backend/) -> the known repo-relative file, by suffix."""
    if path in known_files:
        return path
    hits = [f for f in known_files if f.endswith("/" + path)]
    if hits:
        return min(hits, key=len)
    # The other direction, for an ABSOLUTE path only: an adapter that prints `/tmp/wt/tests/test_x.py`
    # is naming the repo's own `tests/test_x.py` under the execution tree's prefix. Restricted to
    # absolute paths on purpose — a RELATIVE path that merely ends with a known file's path
    # (`helpers/tests/test_x.py`) is a different file and must not be folded onto it.
    if path.startswith("/") or _DRIVE_RE.match(path):
        under = [f for f in known_files if path.endswith("/" + f)]
        if under:
            return max(under, key=len)
    # A case-insensitive filesystem (macOS, Windows) can report a path whose case differs from the one
    # git tracks. Mapped ONLY when exactly one known file matches case-insensitively: to git, two files
    # differing only by case are two different files, so an ambiguous match keeps the path as it came.
    low = path.lower()
    ci = [f for f in known_files if f.lower() == low or f.lower().endswith("/" + low)]
    return ci[0] if len(ci) == 1 else path


def repo_relative_origin(origin: str, known_files: set[str]) -> str:
    """An `origin` (`path:line`) normalised the way test ids are: the PATH through repo_relative, the
    line number kept as it came.

    FIFTH CYCLE (verifier 4). `origin` was the ONE field `_parse_junit` handed to C1 raw. Test ids go
    through `_path_from_classname` or `repo_relative`; origins went through nothing — so C1's
    own-frame allowlist compared a rootdir-relative or absolute frame path against a repo-relative
    module path and refused an honest witness (V4-05, V4-15, V4-75). The comparison stays EXACT; only
    the normalisation is shared. The path is split on the LAST `:` before the digits, so a filename
    that legitimately contains a colon survives."""
    path, sep, line = str(origin).rpartition(":")
    if not sep or not line.isdigit():
        path, line = str(origin), ""
    path = path.replace("\\", "/").strip()
    while path.startswith("./"):                       # the same two normalisations normalize_test_id
        path = path[2:]                                # applies to every id: `./` prefixes and
    path = re.sub(r"(?<!:)/{2,}", "/", path)           # doubled slashes are not identity
    rel = repo_relative(path, known_files)
    return f"{rel}:{line}" if line else rel


_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")
_MODULE_PATH_RE = re.compile(r"importing test module '([^']+)'")


def _collection_path(name: str, file_attr: str | None, body: str, known_files: set[str]) -> str:
    cands = []
    if file_attr:
        cands.append(file_attr)
    if name.endswith(".py"):
        cands.append(name)
    else:
        cands.append(name.replace(".", "/") + ".py")
    m = _MODULE_PATH_RE.search(body)
    if m:
        abs_path = m.group(1).replace("\\", "/")
        for k in known_files:
            if abs_path.endswith("/" + k) or abs_path == k:
                return k
    for c in cands:
        if c in known_files:
            return c
    return cands[0]


def _int_attr(el, name: str) -> int | None:
    v = el.get(name)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def _parse_junit(text: str, known_files: set[str]) -> tuple[dict, dict, dict]:
    try:
        root = ET.fromstring(text)  # nosec B314 — the junit report is the CUSTOMER'S OWN runner output on the CUSTOMER'S OWN runner: a malicious report DoSes only that job. ElementTree does not expand external entities. See SECURITY.md
    except ET.ParseError as exc:
        raise ReportError(f"junit report is not well-formed XML: {exc}") from exc
    results: dict[str, dict] = {}
    coll: dict[str, str] = {}
    # ISSUE #217. `failures`/`errors`/`skipped` count the TESTCASES carrying one (what the per-test
    # results say); `*_elements` count the ELEMENTS, which is what the `<testsuite>` attributes count.
    # They differ exactly when one test reports more than one outcome -- stdlib `unittest.subTest()`
    # puts two <failure> children on ONE <testcase> -- and the attribute comparison needs the element
    # count or it calls mainstream Python a forged report.
    meta = {"testcases": 0, "failures": 0, "errors": 0, "skipped": 0, "module_skips": {},
            "failure_elements": 0, "error_elements": 0, "skipped_elements": 0,
            "suite_tests": None, "suite_failures": None, "suite_errors": None, "suite_skipped": None}
    suites = list(root.iter("testsuite"))
    if suites:
        for k in ("tests", "failures", "errors", "skipped"):
            vals = [_int_attr(s, k) for s in suites]
            if all(v is not None for v in vals):
                meta[f"suite_{k}"] = sum(vals)
    for tc in root.iter("testcase"):
        meta["testcases"] += 1
        classname = tc.get("classname") or ""
        name = tc.get("name") or ""
        failures_el, errors_el, skipped_el = tc.findall("failure"), tc.findall("error"), tc.findall("skipped")
        failure = failures_el[0] if failures_el else None
        error = errors_el[0] if errors_el else None
        skipped = skipped_el[0] if skipped_el else None
        meta["failure_elements"] += len(failures_el)
        meta["error_elements"] += len(errors_el)
        meta["skipped_elements"] += len(skipped_el)
        if failure is not None:
            meta["failures"] += 1
        if error is not None:
            meta["errors"] += 1
        if skipped is not None:
            meta["skipped"] += 1
        if error is not None and "collection" in (error.get("message") or "").lower():
            body = error.text or ""
            path = _collection_path(name, tc.get("file"), body, known_files)
            last = _last_e_line(body)
            coll[path] = last or (_first_line(body) or "collection failure")
            continue
        if not classname and skipped is not None and failure is None and error is None:
            # a MODULE-level skip (pytest.importorskip / pytest.skip(allow_module_level=True) at import): pytest emits
            # <testcase classname="" name="tests.test_opt"> with <skipped>; no test id exists, nothing was collected
            path = _collection_path(name, tc.get("file"), skipped.text or "", known_files)
            meta["module_skips"][path] = _first_line(skipped.get("message") or "") or "module-level skip"
            continue
        prefix = _path_from_classname(classname, known_files) if classname else (tc.get("file") or "")
        test_id = f"{prefix}::{name}" if prefix else name
        origin = entry = None
        if failure is not None:
            status, phase, typ = _classify(failure.get("message") or "", failure.text or "", "failure")
            msg = _first_line(failure.get("message") or "") or _first_line(failure.text or "")
            origin = origin_from_body(failure.text or "")
            if origin:
                origin = repo_relative_origin(origin, known_files)
            entry = entry_from_body(failure.text or "")
        elif error is not None:
            status, phase, typ = _classify(error.get("message") or "", error.text or "", "error")
            msg = _last_e_line(error.text or "") or _first_line(error.get("message") or "")
        elif skipped is not None:
            status, phase, typ, msg = "skip", None, skipped.get("type"), _first_line(skipped.get("message") or "")
        else:
            status, phase, typ, msg = "pass", "call", None, None
        row = {"status": status}
        if phase:
            row["phase"] = phase
        if typ:
            row["type"] = typ
        if msg:
            row["message"] = msg
        if origin:
            row["origin"] = origin
        if entry:
            row["entry"] = entry
        results[test_id] = row
    if meta["testcases"] == 0 and root.tag not in ("testsuites", "testsuite"):
        raise ReportError(f"junit report has no <testsuite>/<testcase> (root <{root.tag}>)")
    return results, coll, meta


def _rel(path: str, repo_dir: str) -> str:
    repo_dir = repo_dir.rstrip("/") + "/"
    return path[len(repo_dir):] if path.startswith(repo_dir) else path


def _parse_jest(text: str, repo_dir: str) -> tuple[dict, dict, dict]:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ReportError(f"jest/vitest JSON report is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("testResults"), list):
        raise ReportError("jest/vitest JSON report has no testResults list")
    results: dict[str, dict] = {}
    coll: dict[str, str] = {}
    meta = {"testcases": 0, "failures": 0, "errors": 0, "skipped": 0,
            "suite_tests": data.get("numTotalTests") if isinstance(data.get("numTotalTests"), int) else None,
            "suite_failures": data.get("numFailedTests") if isinstance(data.get("numFailedTests"), int) else None,
            "suite_errors": None, "suite_skipped": None}
    for suite in data["testResults"]:
        if not isinstance(suite, dict):
            continue
        path = _rel(str(suite.get("name") or suite.get("testFilePath") or ""), repo_dir)
        ars = suite.get("assertionResults") or []
        if not ars and suite.get("status") == "failed":
            msg = str(suite.get("message") or suite.get("failureMessage") or "suite failed to run")
            meaningful = [l.strip() for l in msg.splitlines() if l.strip() and l.strip() != "Test suite failed to run"]
            coll[path] = (meaningful[0] if meaningful else _first_line(msg))[:300]
            meta["errors"] += 1
            continue
        for ar in ars:
            if not isinstance(ar, dict):
                continue
            meta["testcases"] += 1
            tid = f"{path}::{ar.get('fullName') or ar.get('title') or ''}"
            st = str(ar.get("status") or "")
            if st == "passed":
                results[tid] = {"status": "pass", "phase": "call"}
            elif st in ("pending", "skipped", "todo", "disabled"):
                meta["skipped"] += 1
                results[tid] = {"status": "skip", "message": st}
            elif st == "failed":
                msgs = ar.get("failureMessages") or []
                first = _first_line(str(msgs[0])) if msgs else ""
                if first.startswith(("Error: expect(", "expect(", "AssertionError")):
                    meta["failures"] += 1
                    results[tid] = {"status": "fail", "phase": "call", "type": "AssertionError", "message": first}
                else:
                    meta["errors"] += 1
                    m = _EXC_RE.match(first)
                    results[tid] = {"status": "error", "phase": "call", "type": m.group(1) if m else "Error",
                                    "message": first or "failed without a message"}
            else:
                meta["errors"] += 1
                results[tid] = {"status": "error", "phase": "call", "type": "UnknownStatus", "message": st}
    return results, coll, meta


def parse_report_full(family: str, text: str, *, known_files: set[str], repo_dir: str) -> tuple[dict, dict, dict]:
    _check_family(family)
    if family == "pytest":
        return _parse_junit(text, known_files)
    return _parse_jest(text, repo_dir)


def parse_report(family: str, text: str, *, known_files: set[str], repo_dir: str) -> tuple[dict, dict]:
    results, coll, _ = parse_report_full(family, text, known_files=known_files, repo_dir=repo_dir)
    return results, coll


# ----------------------------------------------------------------------------- collect-only

_TREE_LINE_RE = re.compile(r"^(\s*)<(\w+) (.*?)>\s*$")
_COUNT_LINE_RE = re.compile(r"^(\S.*?\.(?:py|pyi)): (\d+)$")
_COLLECTORS = {"Session", "Dir", "Package", "Module", "Class", "UnitTestCase", "DoctestModule", "DoctestTextfile", "YamlFile"}


class CollectResult:
    """What the collect-only pre-pass listed. `ids` when pytest printed node ids (verbosity -1),
    `counts` per file when it printed counts (-qq), both None when nothing parseable came back.
    `fmt` names the shape on the receipt; reconciliation is id-level, count-level, or absent."""

    def __init__(self, ids=None, counts=None, errors=None, summary=None, fmt="none"):
        self.ids: list[str] | None = ids
        self.counts: dict[str, int] | None = counts
        self.errors: list[str] = errors or []
        self.summary: int | None = summary
        self.fmt = fmt

    @property
    def n(self) -> int | None:
        if self.ids is not None:
            return len(self.ids)
        if self.counts is not None:
            return sum(self.counts.values())
        return self.summary


def _resolve_module(dirs: list[str], module: str, known_files: set[str]) -> str:
    for i in range(0, len(dirs) + 1):
        cand = "/".join(dirs[i:] + [module])
        if cand in known_files:
            return cand
    return "/".join(dirs[1:] + [module]) if dirs else module


def parse_collect_only(text: str, known_files: set[str] | None = None) -> CollectResult:
    """pytest `--collect-only`: the flat id list (`-q`), the per-file counts (`-qq`), or the tree
    (verbosity >= 0, e.g. a repo whose addopts carry `-v`) — whichever the repo's own addopts produced."""
    known = set(known_files or ())
    ids: list[str] = []
    counts: dict[str, int] = {}
    errors: list[str] = []
    summary: int | None = None
    tree_ids: list[str] = []
    stack: list[tuple[int, str, str]] = []           # (indent, kind, name)
    for line in (text or "").splitlines():
        tm = _TREE_LINE_RE.match(line)
        if tm:
            indent, kind, name = len(tm.group(1)), tm.group(2), tm.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if kind in _COLLECTORS:
                stack.append((indent, kind, name))
                continue
            dirs = [n for _, k, n in stack if k in ("Dir", "Package")]
            modules = [n for _, k, n in stack if k in ("Module", "DoctestModule", "DoctestTextfile", "YamlFile")]
            classes = [n for _, k, n in stack if k in ("Class", "UnitTestCase")]
            if not modules:
                continue
            path = _resolve_module(dirs, modules[-1], known)
            tree_ids.append("::".join([path, *classes, name]))
            continue
        im = _COLLECT_ID_RE.match(line)
        if im and not line.startswith(("ERROR ", "FAILED ", "WARNING ", "E ")):
            ids.append(repo_relative(im.group(1), known) + "::" + im.group(2).strip())
            continue
        cm = _COUNT_LINE_RE.match(line)
        if cm:
            counts[repo_relative(cm.group(1), known)] = int(cm.group(2))
            continue
        m = _COLLECT_ERROR_RE.match(line)
        if m and m.group(1).rstrip(" -") not in errors:
            errors.append(m.group(1).rstrip(" -"))
        em = re.match(r"^_+ ERROR collecting (\S+) _+$", line)      # the ERRORS section header (present even under -qq)
        if em and em.group(1) not in errors:
            errors.append(em.group(1))
        sm = _COLLECT_SUMMARY_RE.search(line)
        if sm:
            if sm.group(1):
                summary = int(sm.group(1))
            elif sm.group(2):
                summary = int(sm.group(2))
            elif "no tests collected" in line:
                summary = 0
    if ids:
        return CollectResult(ids=ids, errors=errors, summary=summary, fmt="ids")
    if tree_ids:
        return CollectResult(ids=tree_ids, errors=errors, summary=summary, fmt="tree")
    if counts:
        return CollectResult(counts=counts, errors=errors, summary=summary, fmt="counts")
    if summary == 0 or (errors and not ids and not counts):
        # nothing collected — either pytest said so, or every file given errored during collection
        return CollectResult(ids=[], errors=errors, summary=0, fmt="ids")
    return CollectResult(errors=errors, summary=summary, fmt="none")


# ----------------------------------------------------------------------------- reconciliation

class Reconciliation:
    """What one run's reconciliation found. `disagreements` non-empty -> C1 CRASHED naming them.
    `observations` name something the reader should see that is NOT a disagreement (ISSUE #217: a
    `<testsuite tests=N>` surplus accounted for by subtests); they never change a verdict."""

    def __init__(self, disagreements=None, observations=None):
        self.disagreements: list[str] = disagreements or []
        self.observations: list[str] = observations or []


def _itemised(results: dict, coll: dict, collected: "CollectResult | None") -> tuple[list[str], bool]:
    """The report's OWN testcases against the ids/counts the collect-only pre-pass listed on this same
    tree. Returns (disagreements, ran) -- `ran` False means there was nothing independent to compare
    against, which is never read as agreement."""
    out: list[str] = []
    if collected is None:
        return out, False
    if collected.ids is not None:
        cset, reported = set(collected.ids), set(results)
        missing, extra = sorted(cset - reported), sorted(reported - cset)
        if missing:
            out.append(f"{len(missing)} collected id(s) absent from the report: {', '.join(missing[:5])}")
        if extra:
            out.append(f"{len(extra)} reported id(s) were never collected: {', '.join(extra[:5])}")
        return out, True
    if collected.counts is not None:
        per_file: dict[str, int] = {}
        for tid in results:
            f = tid.split("::", 1)[0]
            per_file[f] = per_file.get(f, 0) + 1
        for f, n in collected.counts.items():
            if per_file.get(f, 0) != n:
                out.append(f"{f}: {n} collected, {per_file.get(f, 0)} reported")
        for f, n in per_file.items():
            if f not in collected.counts and f not in coll:
                out.append(f"{f}: {n} reported id(s) but the file was never collected")
        return out, True
    if collected.fmt == "none" and results:
        out.append("the collect-only pre-pass produced no parseable listing while the run reported tests")
    return out, False


def reconcile(family: str, exit_code: int | None, results: dict, coll: dict, meta: dict,
              collected: "CollectResult | list[str] | None") -> list[str]:
    """The disagreements alone -- see `reconcile_full`. Empty = consistent."""
    return reconcile_full(family, exit_code, results, coll, meta, collected).disagreements


def reconcile_full(family: str, exit_code: int | None, results: dict, coll: dict, meta: dict,
                   collected: "CollectResult | list[str] | None") -> Reconciliation:
    """Disagreements between the exit code, the report and what the same tree collected (ids when the
    pre-pass listed them, per-file counts when it only counted), plus the observations that are not
    disagreements. Empty disagreements = consistent."""
    out: list[str] = []
    obs: list[str] = []
    if isinstance(collected, list):
        collected = CollectResult(ids=collected, fmt="ids")
    itemised, itemised_ran = _itemised(results, coll, collected)
    n_fail, n_err, n_coll = meta.get("failures", 0), meta.get("errors", 0), len(coll)
    n_reported = len(results)
    if family == "pytest":
        if exit_code == 0:
            if n_fail or n_err or n_coll:
                out.append(f"exit 0 but the report carries {n_fail} failure(s), {n_err} error(s), {n_coll} collection error(s)")
            if n_reported == 0:
                out.append("exit 0 with no testcase in the report (pytest exits 5 when nothing is collected)")
        elif exit_code == 1:
            if not (n_fail or n_err or n_coll):
                out.append("exit 1 but the report carries no failure, error or collection error")
        elif exit_code == 5:
            if n_reported or n_coll:
                out.append(f"exit 5 (no tests collected) but the report carries {n_reported} testcase(s)")
        elif exit_code in (2, 3, 4):
            what = {2: "interrupted", 3: "internal error", 4: "usage error"}[exit_code]
            out.append(f"exit {exit_code} ({what}): the run did not complete; its report is partial")
        else:
            out.append(f"abnormal exit {exit_code}: the runner did not finish normally")
        # ISSUE #217, the crash that was 54 of the 55 crashes in the 1,440-PR field replay. pytest emits
        # one <testcase> per COLLECTED TEST but counts every OUTCOME in the <testsuite> attributes, and
        # a stdlib `unittest.subTest()` produces more outcomes than tests. A PASSING subtest leaves no
        # element in the report at all, so the surplus cannot be itemised from the report -- which is
        # why the decision is made against an INDEPENDENT oracle instead: the ids the collect-only
        # pre-pass listed on this same tree. When every <testcase> reconciles one-for-one with those,
        # nothing was dropped from or invented in the itemised report, and a `tests` SURPLUS is the
        # runner's own outcome count, not a forgery. A DEFICIT, a surplus with no itemised comparison
        # available, and a surplus over an itemised comparison that DISAGREES all stay disagreements,
        # as do all three other attributes in both directions -- a <failure> deleted to fake a pass is
        # still caught by `failures`, which is the forgery this comparison exists for.
        counts = {"tests": meta.get("testcases", 0),
                  "failures": meta.get("failure_elements", n_fail),
                  "errors": meta.get("error_elements", n_err),
                  "skipped": meta.get("skipped_elements", meta.get("skipped", 0))}
        for k in ("tests", "failures", "errors", "skipped"):
            sv = meta.get(f"suite_{k}")
            counted = counts[k]
            if sv is None or sv == counted:
                continue
            if k == "tests" and sv > counted and itemised_ran and not itemised:
                obs.append(f"<testsuite tests={sv}> with {counted} <testcase>(s): {sv - counted} outcome(s) counted "
                           f"with no <testcase> of their own — the shape stdlib unittest.subTest() produces")
                continue
            out.append(f"<testsuite {k}={sv}> but {counted} <testcase>(s) counted as {k}")
    else:
        if exit_code == 0 and (n_fail or n_err or n_coll):
            out.append(f"exit 0 but the report carries {n_fail} failure(s), {n_err} error(s), {n_coll} suite error(s)")
        elif exit_code == 1 and not (n_fail or n_err or n_coll):
            out.append("exit 1 but the report carries no failure or error")
        elif exit_code not in (0, 1):
            out.append(f"abnormal exit {exit_code}: the runner did not finish normally")
    out.extend(itemised)
    return Reconciliation(disagreements=out, observations=obs)
