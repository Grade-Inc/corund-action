"""C1's gathering half: run the PR's changed tests WITH the change and with the PR's NON-TEST,
NON-INFRASTRUCTURE diff REVERTED onto the base, each phase in its own fresh, git-less tree, and hand
the pure core its inputs.

Mechanism (all local git, no network):
  1. changed files = git diff --name-status --no-renames -z <merge-base> <head>; split three ways:
     TEST files (the globs), TEST-INFRASTRUCTURE files (conftest.py, pytest.ini, pyproject.toml,
     setup.cfg, tox.ini, jest/vitest/vite config, package.json, .github/workflows/**, __init__.py under
     test dirs — NEVER reverted; their changes are C2 tier B-D / C3 findings) and the NON-TEST diff.
  2. reverse patch = git diff --binary <head> <merge-base> -- <non-test files>; its COMPOSITION is
     read (text lines, mode-only, binary, symlink) — no text line to revert is NOT_RUN.
  3. PHASE "with-change": fresh `git worktree` at <head>, `.git` link REMOVED, fresh TMPDIR, a
     collect-only pre-pass, then the run with a report written outside the tree; the tree is deleted.
  4. PHASE "without-change": a SECOND fresh worktree at <head>, the reverse patch applied, `.git`
     removed, and the same collect + run.
  5. if anything is red WITHOUT the change, PHASE "without-change-rerun": a THIRD fresh reverted tree.
  6. every phase is RECONCILED: exit code vs report counts vs collected ids (runners.reconcile); any
     disagreement raises Inconsistent -> the caller reports C1 CRASHED naming it.

No state, marker file, `git status` or `git log` oracle can leak between phases: each phase sees a
tree that has never been run in, has no `.git`, and has its own TMPDIR; the environment is the same
in every phase (no phase label is exported; PYTEST_ADDOPTS is dropped).

Outcomes: `inputs` (the C1 snapshot + execution notes), `not_run` with the reason, or RunnerError
(test command missing, no report on the WITH run, revert did not apply, timeout, an inconsistent
run) which the caller turns into CRASHED with the text. A reverted run that exits WITHOUT a report
is a fact about the reverted tree: the snapshot carries `test_failure_kinds_without_change["<session>"]`
and the core reports NOT_RUN with that reason (owner's ruling; never PROVEN). NEW module.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess  # nosec B404 — the Action runs the caller's own declared test command by design
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

from corund_checks.c2_skip_audit import is_test_infrastructure
from corund_checks.unidiff import parse_unified_diff

from . import gitio, runners, subject_trace

_SOURCE_EXT = (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")


class RunnerError(Exception):
    pass


class NoReport(RunnerError):
    """The test command exited without writing a report. On the WITH run this is the Action's problem
    (CRASHED); on the reverted run it is a fact about the reverted tree (a `<session>` entry)."""

    def __init__(self, label: str, exit_code: int | None, tail: list[str]):
        self.exit_code, self.tail = exit_code, tail
        super().__init__(f"{label}: the test command produced no report (exit {exit_code}); "
                         f"output tail: {' | '.join(tail[-5:])[:400]}")


class Inconsistent(RunnerError):
    """Exit code, report and collected ids disagree for one run: the measurement cannot be trusted."""

    def __init__(self, label: str, disagreements: list[str]):
        self.label, self.disagreements = label, disagreements
        super().__init__(f"{label}: report inconsistent — " + "; ".join(disagreements)[:600])


@dataclass
class C1Outcome:
    kind: str                                  # inputs | not_run
    inputs: dict = field(default_factory=dict)
    reason: str = ""
    runs: list[dict] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    non_test_files: list[str] = field(default_factory=list)
    infra_files: list[str] = field(default_factory=list)
    # TWELFTH CYCLE. The runtime silencing probe C2 consumes. It is ADAPTER output, not C1 output --
    # `run_c1` has no C2 input of any kind -- so the entrypoint gathers it BEFORE either check runs
    # and hands it to C2 as data, with no cycle between the checks. Every return path sets it; the
    # value always speaks corund_checks.runtime_silencing's closed vocabulary, so "we did not measure"
    # is a stated reason on the receipt and never an absent key.
    probe: dict = field(default_factory=lambda: {"state": "runner-error",
                                                 "detail": "the runner set no probe on this path"})


def split_globs(test_globs: str) -> list[str]:
    return [g.strip() for g in test_globs.replace(";", ",").split(",") if g.strip()]


def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """Glob -> regex where `*` and `?` never cross `/` and `**` spans directories (so `**/test_*.py` does
    not match `src/test_utils/helper.py` — verifier 1)."""
    out = "^"
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
            continue
        if pattern.startswith("**", i):
            out += ".*"
            i += 2
            continue
        if ch == "*":
            out += "[^/]*"
        elif ch == "?":
            out += "[^/]"
        else:
            out += re.escape(ch)
        i += 1
    return re.compile(out + "$")


def is_test_path(path: str, globs: list[str]) -> bool:
    return any(_glob_regex(g).match(path) for g in globs)


def split_changed(changed: list[tuple[str, str]], globs: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(test files to run, test-infrastructure files (never reverted), non-test files to revert)."""
    tests, infra, non_test = [], [], []
    for st, p in changed:
        if is_test_path(p, globs):
            if st != "D":
                tests.append(p)
        elif is_test_infrastructure(p):
            infra.append(p)
        else:
            non_test.append(p)
    return tests, infra, non_test


# ---------------------------------------------------------- ISSUE #202: the repository's own scope
#
# The launch gate is "C1 installs and works on a STRANGER'S pytest repo in under 10 minutes", and the
# shape that broke it is the most ordinary one there is: pallets/flask keeps `examples/javascript/`,
# its own project with its own pyproject.toml, beside the suite. A changed file there matches the
# default glob `**/test_*.py`, so C1 ran a test the REPOSITORY ITSELF NEVER RUNS, in a sub-package
# nothing had installed, and the whole check CRASHED on `ImportError while loading conftest`.
#
# The repository already says which files are its suite, in its own pytest configuration. C1 reads
# that declaration and does not select outside it. Three rules bound the change:
#   (a) a repo that declares nothing selects exactly what it selected before -- no entries, no filter;
#   (b) a caller who passed their OWN --test-globs asked for those files, so the declaration is read
#       and REPORTED but not applied;
#   (c) a crash is never swallowed: selection narrows what runs, and whatever still runs and cannot
#       is CRASHED with its cause named, exactly as before.
# In every case an excluded file is NAMED on the receipt with the reason -- a file dropped in silence
# is the absence of a question, not the answer to one.

# The DEFAULT the Action ships (action.yml `test-globs`). It lives HERE, next to the code that has to
# know whether the caller chose their own globs, and `entrypoint.DEFAULT_GLOBS` is this same object:
# the decision is the core's, and an adapter cannot weaken it by omitting a flag, because "not stated"
# is the STRICTER branch.
DEFAULT_TEST_GLOBS = ("tests/**/test_*.py,**/test_*.py,**/*_test.py,**/*.test.js,**/*.test.ts,**/*.test.tsx,"
                      "**/*.test.jsx,**/*.spec.ts,**/*.spec.js,**/__tests__/**")

# pytest's OWN documented order for finding its configfile ("Initialization: determining rootdir and
# configfile"): the first file that MATCHES wins and the candidates are never merged. pytest.ini (and
# its dotted alias) match always, even when empty; the other three match only when they carry pytest's
# section. Derived from pytest's documentation, not guessed, and asserted by unit tests.
PYTEST_CONFIG_ORDER = (
    ("pytest.ini", "pytest"),
    (".pytest.ini", "pytest"),
    ("pyproject.toml", "tool.pytest.ini_options"),
    ("tox.ini", "pytest"),
    ("setup.cfg", "tool:pytest"),
)
NESTED_PROJECT_MARKERS = ("pyproject.toml", "setup.py")


class TestPaths:
    """What the repository declares its own test suite to be. `entries` is None when it declares
    nothing -- which is not the same as declaring an empty list, and takes rule (a)'s path."""

    def __init__(self, entries=None, source="", note=""):
        self.entries: list[str] | None = entries
        self.source = source
        self.note = note

    def __repr__(self) -> str:
        return f"TestPaths(entries={self.entries!r}, source={self.source!r}, note={self.note!r})"


def _ini_testpaths(text: str, section: str) -> "list[str] | None":
    import configparser
    cp = configparser.RawConfigParser(strict=False)
    cp.read_string(text)
    if not cp.has_section(section):
        raise KeyError(section)
    raw = cp.get(section, "testpaths", fallback=None)
    return raw.split() if raw is not None else None


def _toml_testpaths(text: str) -> "list[str] | None":
    """pytest matches a pyproject.toml only when it carries pytest's own table: the long-standing
    `[tool.pytest.ini_options]`, or (pytest 9) the native-TOML `[tool.pytest]`. Neither present means
    pytest does not read this file at all, and neither do we."""
    import tomllib
    pytest_table = tomllib.loads(text).get("tool", {}).get("pytest", {})
    if not isinstance(pytest_table, dict):
        raise KeyError("tool.pytest")
    if "ini_options" in pytest_table:
        table = pytest_table["ini_options"]
    elif set(pytest_table) - {"ini_options"}:
        table = pytest_table                          # native TOML: [tool.pytest] itself
    else:
        raise KeyError("tool.pytest.ini_options")
    raw = table.get("testpaths") if isinstance(table, dict) else None
    if raw is None:
        return None
    return [str(x) for x in raw] if isinstance(raw, (list, tuple)) else str(raw).split()


def read_testpaths(repo_dir: str, ref: str) -> TestPaths:
    """The repository's `testpaths`, read from ITS OWN configuration at `ref` in pytest's documented
    precedence. A file that cannot be parsed excludes NOTHING and says so on the receipt: we do not
    know what the repo declared, so we run what we would have run before, and whatever cannot run is
    still CRASHED with its cause named."""
    for name, section in PYTEST_CONFIG_ORDER:
        text = gitio.show_file(repo_dir, ref, name)
        if text is None:
            continue
        try:
            entries = _toml_testpaths(text) if name == "pyproject.toml" else _ini_testpaths(text, section)
        except KeyError:
            continue                  # the file exists but carries no pytest section: pytest does not match it
        except Exception as exc:      # noqa: BLE001 — any parse failure at all is "we could not read it"
            return TestPaths(None, name, f"{name} could not be parsed ({type(exc).__name__}: {str(exc)[:120]}), "
                                         f"so no changed test file was excluded on its account")
        if entries is None:
            return TestPaths(None, name, f"{name} is this repository's pytest configuration and declares no testpaths")
        return TestPaths([e.strip("/") for e in entries if e.strip("/")] or None, name, "")
    return TestPaths(None, "", "this repository declares no pytest configuration file")


def within_testpaths(path: str, entries: list[str]) -> bool:
    """Is `path` inside one of the repository's declared testpaths? An entry may name a directory, a
    single file, or (pytest 8+) a glob. `.` is the whole tree."""
    for e in entries:
        e = e.strip("/")
        if e in ("", "."):
            return True
        if any(ch in e for ch in "*?["):
            if _glob_regex(e).match(path) or _glob_regex(e + "/**").match(path):
                return True
            continue
        if path == e or path.startswith(e + "/"):
            return True
    return False


def nested_project_dir(path: str, tracked: "set[str] | None") -> str:
    """The nearest ancestor directory (never the repo root) that carries its own project marker, named
    as `<dir>/<marker>` -- flask's `examples/javascript/pyproject.toml`. Reported, never decisive on
    its own: excluding on a marker alone would change what a repo that declares NOTHING selects, and
    rule (a) forbids that."""
    if not tracked:
        return ""
    parts = path.split("/")[:-1]
    while parts:
        d = "/".join(parts)
        for m in NESTED_PROJECT_MARKERS:
            if f"{d}/{m}" in tracked:
                return f"{d}/{m}"
        parts.pop()
    return ""


class Selection:
    """The repository-scope filter applied to C1's candidate test files, and what it has to say."""

    def __init__(self, decl: TestPaths, globs_are_default: bool, tracked: "set[str] | None" = None,
                 family: str = "pytest"):
        self.decl, self.globs_are_default, self.tracked = decl, globs_are_default, tracked or set()
        self.family = family
        self.excluded: dict[str, str] = {}
        self.observed: dict[str, str] = {}

    @property
    def active(self) -> bool:
        # `testpaths` is PYTEST'S declaration about PYTEST'S suite. It says nothing about a jest or
        # vitest run, and narrowing one with it would exclude a JS test on the strength of a Python
        # config file -- an accusation-shaped mistake against an ordinary monorepo.
        return bool(self.decl.entries) and self.globs_are_default and self.family == "pytest"

    def keep(self, paths: list[str]) -> list[str]:
        out = []
        for q in paths:
            nested = nested_project_dir(q, self.tracked)
            if self.active and not within_testpaths(q, self.decl.entries or []):
                why = (f"outside this repository's own testpaths ({', '.join(self.decl.entries or [])}) "
                       f"declared in {self.decl.source}")
                if nested:
                    why += f"; it is a separate project with its own {nested}"
                self.excluded[q] = why
                continue
            if nested and not self.active:
                self.observed[q] = (f"it sits under a separate project ({nested}), but this repository declares no "
                                    f"testpaths, so nothing was excluded on that account")
            out.append(q)
        return out

    def note(self) -> str:
        if self.decl.entries and self.globs_are_default and self.family != "pytest":
            return (f"test selection: this repository declares pytest testpaths ({', '.join(self.decl.entries)} in "
                    f"{self.decl.source}) and it was NOT applied — this run's runner family is {self.family}, and a "
                    f"pytest declaration says nothing about a {self.family} suite. Nothing was excluded.")
        if self.excluded:
            return ("test selection: " + f"{len(self.excluded)} changed test file(s) EXCLUDED — "
                    + "; ".join(f"{q} ({why})" for q, why in sorted(self.excluded.items()))
                    + ". C1 runs the repository's own suite: a file the repo itself never runs cannot be a "
                      "red-on-revert witness, and running it is how this check CRASHED on an uninstalled "
                      "sub-package. Pass your own `test-globs` to include it.")
        if self.decl.entries and not self.globs_are_default:
            return (f"test selection: this repository declares testpaths {', '.join(self.decl.entries)} in "
                    f"{self.decl.source}, and it was NOT applied — the caller passed their own test-globs, which win. "
                    f"Every changed test file matching those globs was selected.")
        if self.decl.entries:
            return (f"test selection: every changed test file is inside this repository's own testpaths "
                    f"({', '.join(self.decl.entries)}, declared in {self.decl.source})")
        if self.observed:
            return ("test selection: no testpaths declared, so nothing was excluded; OBSERVED — "
                    + "; ".join(f"{q}: {why}" for q, why in sorted(self.observed.items())))
        return f"test selection: {self.decl.note or 'this repository declares no testpaths'}; nothing was excluded"


def build_selection(repo_dir: str, head_sha: str, test_globs: str, tracked: "list[str] | None" = None,
                    family: str = "pytest") -> Selection:
    return Selection(read_testpaths(repo_dir, head_sha),
                     split_globs(test_globs) == split_globs(DEFAULT_TEST_GLOBS),
                     set(tracked or []), family)


def patch_composition(patch) -> dict:
    """What the reverse patch would change: per file text lines, mode-only / binary / symlink flags.
    The patch is BYTES (it must apply byte-for-byte); it is decoded with replacement for reading only."""
    text = patch.decode("utf-8", errors="replace") if isinstance(patch, bytes) else patch
    comp = {"files": 0, "text_lines": 0, "source_text_lines": 0, "mode_only": [], "binary": [], "symlink": [], "text_files": []}
    if not text.strip():
        return comp
    for fd in parse_unified_diff(text):
        comp["files"] += 1
        if fd.mode_only:
            comp["mode_only"].append(fd.path)
            continue
        if fd.binary:
            comp["binary"].append(fd.path)
            continue
        if fd.symlink:
            comp["symlink"].append(fd.path)
            continue
        n = fd.text_line_count
        comp["text_lines"] += n
        if n:
            comp["text_files"].append(f"{fd.path} (+{fd.added_count}/-{fd.deleted_count})")
            if fd.path.lower().endswith(_SOURCE_EXT):
                comp["source_text_lines"] += n
    return comp


# Environment variables that register plugins or inject flags into every pytest run. They are DROPPED
# from every phase, not merely reported: a plugin registered by the environment runs on the reverted
# tree too, so leaving it in place would put outcome-affecting code C1 never reverted into the
# comparison. When one was present the runner says so on the receipt (see `dropped_env`).
OUTCOME_AFFECTING_ENV = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")


def dropped_env(environ: "dict[str, str] | None" = None) -> list[str]:
    """The names of OUTCOME_AFFECTING_ENV that were actually set — for the receipt."""
    src = os.environ if environ is None else environ
    return [k for k in OUTCOME_AFFECTING_ENV if src.get(k)]


def _env(run_tmp: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in OUTCOME_AFFECTING_ENV}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "CI": "true", "TMPDIR": run_tmp, "TEMP": run_tmp, "TMP": run_tmp})
    return env


def _run(cmd: list[str], cwd: str, timeout_s: int, label: str, env: dict[str, str]) -> tuple[dict, str]:
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout_s, env=env)  # nosec B603 — argv list, shell=False; the command is the caller's own declared test command
    except FileNotFoundError as exc:
        raise RunnerError(f"{label}: test command not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"{label}: test command timed out after {timeout_s}s: {' '.join(cmd)[:200]}") from exc
    # never a traceback on non-UTF-8 output: decode with replacement (verifier 1)
    stdout = r.stdout.decode("utf-8", errors="replace")
    stderr = r.stderr.decode("utf-8", errors="replace")
    tail = (stdout + "\n" + stderr).strip().splitlines()[-15:]
    return {"label": label, "command": cmd, "exit_code": r.returncode,
            "seconds": round(time.monotonic() - t0, 2), "tail": tail}, stdout


@dataclass
class Phase:
    results: dict
    coll: dict
    collected: "runners.CollectResult | None"
    # FOURTEENTH CYCLE: the subject trace this phase recorded (test id -> [path::qualname]), or None
    # with `subjects_note` saying why (the plugin could not be loaded, no sidecar was written, ...).
    subjects: "dict[str, list[str]] | None" = None
    subjects_note: str = ""


def _fn_name(test_id: str) -> str:
    """The function name of a test id: the last `::` segment minus its `[parametrisation]`."""
    tail = test_id.split("::")[-1]
    cut = tail.find("[")
    return tail if cut < 0 else tail[:cut]


def _with_plugin(cmd: list[str], plugin_args: list[str], test_files: list[str]) -> list[str]:
    """`-p corund_subject_trace` inserted BEFORE the test file arguments (pytest reads options first)."""
    if not plugin_args:
        return list(cmd)
    head = cmd[:len(cmd) - len(test_files)] if test_files else list(cmd)
    return [*head, *plugin_args, *test_files]


def _plugin_import_failed(exit_code: int | None, output: str) -> bool:
    """pytest could not import the `-p` plugin: a non-zero exit whose output carries pytest's own
    `Error importing plugin "<name>"` line naming this plugin (exit 4 as a usage error on older
    pytest, exit 1 with a traceback on newer ones -- the message is the stable part)."""
    text = output or ""
    return bool(exit_code) and f'Error importing plugin "{subject_trace.MODULE_NAME}"' in text


def _phase(label: str, *, repo_dir: str, head_sha: str, patch: "bytes | None", family: str, test_command: list[str],
           test_files: list[str], timeout_s: int, workdir: str | None, runs: list[dict],
           trace: "bool | Callable[[runners.CollectResult | None], set[str] | None]" = False) -> Phase:
    """One isolated run. `trace` (FOURTEENTH CYCLE): False = load the subject-trace plugin but trace
    nothing; True = trace every test; a callable is given the collect-only listing and returns the
    node ids to trace (None = every test). The plugin is loaded in EVERY pytest phase so the plugin
    list and the environment are the same in each; only the targets differ."""
    tree = tempfile.mkdtemp(prefix="corund-tree-", dir=workdir)
    os.rmdir(tree)
    run_tmp = tempfile.mkdtemp(prefix="corund-run-tmp-")
    report_dir = tempfile.mkdtemp()                                   # random name, outside the tree and the run's TMPDIR
    isolation = {"tree": "fresh worktree", "git": "", "tmpdir": "fresh per run", "report": "outside the tree", "patch": "none"}
    try:
        try:
            gitio.worktree_add(repo_dir, tree, head_sha)
        except gitio.GitError as exc:
            raise RunnerError(f"{label}: could not create the execution tree: {exc}") from exc
        if patch:
            try:
                gitio.apply_patch(tree, patch)
            except gitio.GitError as exc:
                raise RunnerError(f"{label}: revert of the non-test diff did not apply cleanly: {exc}") from exc
            isolation["patch"] = "reverse patch applied, then .git removed"
        isolation["git"] = gitio.strip_git_link(tree)
        env = _env(run_tmp)
        env["PWD"] = tree                 # the copied environment must not name the original checkout
        plugin_args: list[str] = []
        subjects_out = None
        subjects_note = ""
        if family == "pytest":
            subject_trace.install(report_dir)
            subjects_out = os.path.join(report_dir, secrets.token_hex(8) + ".subjects.json")
            env["PYTHONPATH"] = report_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env[subject_trace.OUT_VAR] = subjects_out
            env[subject_trace.ROOT_VAR] = tree
            env[subject_trace.TARGETS_VAR] = subject_trace.write_targets(report_dir, set())   # nothing yet
            plugin_args = ["-p", subject_trace.MODULE_NAME]
        collected: runners.CollectResult | None = None
        ccmd = runners.build_collect_command(family, test_command, test_files)
        if ccmd is not None:
            cinfo, cout = _run(_with_plugin(ccmd, plugin_args, test_files), tree, timeout_s, f"{label}:collect", env)
            if plugin_args and _plugin_import_failed(cinfo["exit_code"], "\n".join(cinfo["tail"]) + cout):
                # The wrapper scrubbed PYTHONPATH (or pytest could not import the plugin for another
                # reason): run WITHOUT the trace rather than not at all, and SAY so. The core fails closed
                # on an unavailable trace (a rename is then accused), which is the safe direction.
                subjects_note = (f"pytest could not import the subject-trace plugin in this environment (collect exited "
                                 f"{cinfo['exit_code']}); the phase ran without it")
                plugin_args = []
                for k in (subject_trace.OUT_VAR, subject_trace.ROOT_VAR, subject_trace.TARGETS_VAR):
                    env.pop(k, None)
                subjects_out = None
                cinfo["subject_trace"] = "plugin import failed; retried without it"
                runs.append(cinfo)
                cinfo, cout = _run(ccmd, tree, timeout_s, f"{label}:collect", env)
            collected = runners.parse_collect_only(cout, set(test_files))
            cinfo["collected"] = collected.n
            cinfo["collected_format"] = collected.fmt
            cinfo["collection_errors"] = len(collected.errors)
            runs.append(cinfo)
        targets: "set[str] | None" = set()
        if plugin_args:
            if trace is True:
                targets = None
            elif callable(trace):
                targets = trace(collected)
            subject_trace.write_targets(report_dir, targets)
        report = os.path.join(report_dir, secrets.token_hex(8) + (".xml" if family == "pytest" else ".json"))
        cmd = _with_plugin(runners.build_command(family, test_command, report, test_files), plugin_args, test_files)
        info, _ = _run(cmd, tree, timeout_s, label, env)
        info["isolation"] = isolation
        if plugin_args:
            info["subject_trace"] = ("every test" if targets is None else f"{len(targets)} targeted test(s)")
        elif family == "pytest":
            info["subject_trace"] = "unavailable: " + (subjects_note or "not loaded")
        runs.append(info)
        if not os.path.exists(report) or os.path.getsize(report) == 0:
            info["reconciled"] = ["no report written"]
            raise NoReport(label, info["exit_code"], info["tail"])
        with open(report, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        results, coll, meta = runners.parse_report_full(family, text, known_files=set(test_files), repo_dir=tree)
        info["tests"] = len(results)
        info["collection_errors"] = len(coll)
        if collected is not None:
            info["collected"] = collected.n
            info["collected_format"] = collected.fmt
        rec = runners.reconcile_full(family, info["exit_code"], results, coll, meta, collected)
        info["reconciled"] = "ok" if not rec.disagreements else rec.disagreements
        if rec.observations:
            info["reconcile_observations"] = rec.observations
        if rec.disagreements:
            raise Inconsistent(label, rec.disagreements)
        subjects: "dict[str, list[str]] | None" = None
        if plugin_args and subjects_out is not None:
            raw_subjects, errors, found = subject_trace.read(subjects_out)
            if not found:
                subjects_note = "the subject-trace plugin wrote no sidecar (the session did not reach its end hook)"
            else:
                known = set(test_files)
                subjects = {}
                for nodeid, fns in raw_subjects.items():
                    path, sep, rest = nodeid.partition("::")
                    tid = runners.repo_relative(path, known) + sep + rest
                    subjects[tid] = fns
                if errors:
                    subjects_note = f"{len(errors)} note(s) from the trace: " + " | ".join(errors[:3])[:300]
                if targets is not None:
                    # a targeted test that ran and left no record was not traced: say so, rather than
                    # hand the core an empty set that reads as "entered nothing"
                    missing = [t for t in targets if t in results and t not in subjects and results[t].get("status") != "skip"]
                    if missing:
                        subjects_note = (f"{len(missing)} targeted test(s) left no trace record"
                                         + (f" ({', '.join(missing[:3])})" if missing else ""))
                        subjects = None
        elif family == "pytest" and not subjects_note:
            subjects_note = "not loaded"
        return Phase(results, coll, collected, subjects, subjects_note)
    finally:
        shutil.rmtree(tree, ignore_errors=True)
        shutil.rmtree(run_tmp, ignore_errors=True)
        shutil.rmtree(report_dir, ignore_errors=True)
        gitio.worktree_prune(repo_dir)


def _failure_kinds(results: dict) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for tid, r in results.items():
        st = r.get("status")
        if st == "fail":
            kinds[tid] = "assertion"
        elif st == "error":
            typ = str(r.get("type") or "")
            phase = r.get("phase")
            if typ in ("ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError"):
                kinds[tid] = "import"
            elif "Timeout" in typ or "TimedOut" in typ:
                kinds[tid] = "timeout"
            elif phase in ("collection", "setup", "teardown"):
                kinds[tid] = "collection"
            else:
                kinds[tid] = "error"
    return kinds


# C1's own-frame ALLOWLIST is STRICT BY DEFAULT: a witness must affirm that its assertion was raised
# in the test's own function in the test's own module. An adapter whose report carries no frames at
# all cannot affirm it and must SAY SO — the declaration is printed on every receipt that uses it.
# pytest is absent from this map on purpose: its junit body carries the frames (the Action forces
# `--tb=auto` on the command line, which outranks any `--tb=` a repo puts in its addopts), so a
# pytest witness with no frame is refused rather than waived.
FRAME_ATTRIBUTION = {
    "jest": "unavailable: the jest JSON report gives no frame for a failed assertion, so C1 cannot check that the "
            "red came from the test's own function rather than a helper, a reporter or a setup file",
    "vitest": "unavailable: the vitest JSON report gives no frame for a failed assertion, so C1 cannot check that the "
              "red came from the test's own function rather than a helper, a reporter or a setup file",
}


def modified_test_files(changed: list[tuple[str, str]], globs: list[str]) -> list[str]:
    """Test files the PR MODIFIES **or DELETES**: a base version of them exists to run against this
    PR's code. A test file the PR ADDS has no base version to compare against and is out of scope by
    construction; that is a stated residual of the runtime silencing rule.

    THIRTEENTH CYCLE (owner's ruling 2026-09-07, verifier 12's H3/H11). DELETED files used to be
    excluded on the ground that they were "already `[A/test-deleted]`" -- and `[A/test-deleted]` is
    exactly the detector that accuses ordinary work: deleting a feature together with its test, and
    relocating a test to another file. The runtime rule can DECIDE those, and only if it is given the
    file: restore the deleted test, run it against this PR's code, and read what happens. A test
    deleted with the feature it covered ERRORS (the symbol is gone) and proves nothing; a test
    deleted to hide a regression FAILS BY ASSERTION and is accused. The syntactic tier then defers to
    that fact instead of accusing both."""
    return sorted(p for st, p in changed if st != "A" and is_test_path(p, globs))


def _outside_reds(results: dict, scope: set[str]) -> set[str]:
    """Test ids in the WIDE base-tests run that fail by assertion and live in a file the PR does NOT
    modify or delete. `fail` is the adapter's word and it is only ever granted after positive evidence
    that an assertion executed and failed (`runners._classify`)."""
    out = set()
    for tid, r in results.items():
        if isinstance(r, dict) and r.get("status") == "fail" and tid.split("::", 1)[0] not in scope:
            out.add(tid)
    return out


def _full_revert(repo_dir: str, mb: str, head_sha: str, non_test: list[str], modified_tests: list[str]):
    """The reverse patch that takes the head tree ALL the way back: the non-test diff AND the PR's own
    test-file diff. The `suite-base` phase runs it to answer the one question no other phase can --
    was this test ALREADY failing before this PR? A red that predates the diff is not a red the diff
    caused, and accusing there would accuse a PR that FIXES cross-test pollution, which is honest work."""
    return gitio.diff(repo_dir, head_sha, mb, paths=sorted(set(non_test) | set(modified_tests)),
                      binary=True, renames=False, as_bytes=True)



def subject_trace_repair_key() -> str:
    """The probe key for the base-tree trace of the deleted-and-detecting tests (the core's REPAIR_KEY)."""
    from corund_checks.subjects import REPAIR_KEY
    return REPAIR_KEY


def _subjects_block(*phases: "tuple[str, Phase | None]") -> dict:
    """The probe's `subjects` block from the phases that were traced: measured only when EVERY traced
    phase produced a trace; otherwise `unavailable` naming the phase and why (the core fails closed)."""
    block: dict = {"state": "measured"}
    for key, run in phases:
        if run is None:
            block[key] = {}
            continue
        if run.subjects is None:
            return {"state": "unavailable", "detail": f"{key}: {run.subjects_note or 'no trace'}"}
        block[key] = run.subjects
    return block


def _absent_targets(modified_tests: list[str], head_results: dict):
    """The base-phase trace targets: base tests in the modified files whose FUNCTION NAME no head test
    carries -- the only ones the accounting needs a subject for. Falls back to tracing every test when
    the listing has no ids (a repo whose addopts make pytest print counts)."""
    head_names = {_fn_name(t) for t in head_results}
    mod = set(modified_tests)

    def pick(collected: "runners.CollectResult | None"):
        if collected is None or collected.ids is None:
            return None
        return {t for t in collected.ids if t.split("::", 1)[0] in mod and _fn_name(t) not in head_names}
    return pick


def _base_probe(*, common: dict, modified_tests: list[str], all_tests: list[str], changed, globs: list[str],
                with_: "Phase | None", without: "Phase | None", notes: list[str], repo_dir: str, mb: str,
                head_sha: str, non_test: list[str], test_only: bool) -> dict:
    """The runtime silencing probe: the WIDE base-tests phase (thirteenth cycle) plus, this cycle, the
    subject trace of every test the accounting can ask about. `test_only` is the PR that changes no
    non-test code: there is no reverted run (nothing to revert), the with-change results stand in for
    both head runs, and the core is told `no_non_test_change` so it accounts without detecting."""
    if not modified_tests:
        return {"state": "no-modified-test-files"}
    tpatch = gitio.diff(repo_dir, head_sha, mb, paths=modified_tests, binary=True, renames=False, as_bytes=True)
    if not tpatch.strip():
        return {"state": "no-modified-test-files",
                "detail": f"{len(modified_tests)} modified test file(s) whose reverse patch changes no text"}
    # THIRTEENTH CYCLE (verifier 12, D3). The base-tests phase is WIDE: it runs EVERY test file git
    # tracks, with the PR's own test files reverted to their base content. It used to run the modified
    # test files ALONE, and that scope is what D3 walked through -- a modified test file that rebinds a
    # global inside an UNMODIFIED one silences a test the probe never ran, and no comparison over the
    # modified files can see it, because the two files never met in one process.
    suite = sorted(set(all_tests) | set(modified_tests))
    head_results = with_.results if with_ is not None else {}
    try:
        base_tests = _phase("base-tests", patch=tpatch, trace=_absent_targets(modified_tests, head_results),
                            **{**common, "test_files": suite})
        probe = {"state": "measured", "files": modified_tests,
                 "changed_files": sorted(p for _st, p in changed),
                 "suite_files": suite,
                 "deleted_files": sorted(q for st, q in changed if st == "D" and is_test_path(q, globs)),
                 "base_tests_on_head": base_tests.results,
                 # FOURTEENTH CYCLE (verifier 13, D06): a modified/deleted test file whose BASE content does
                 # not even collect against this PR's code (the module it imports was renamed) ran no test
                 # at all; the core needs to know that rather than see an empty enumeration.
                 "base_collection_errors": {k: str(v) for k, v in base_tests.coll.items() if k in set(modified_tests)},
                 "head_tests_on_head": head_results,
                 "head_tests_on_base": (without.results if without is not None else head_results),
                 "no_non_test_change": test_only,
                 "subjects": _subjects_block(("base_tests_on_head", base_tests), ("head_tests_on_head", with_),
                                             ("head_tests_on_base", without if not test_only else with_))}
        notes.append(f"runtime silencing probe: a {'base-tests' if test_only else 'FOURTH'} phase ran {len(suite)} tracked "
                     f"test file(s) with the {len(modified_tests)} the PR modifies or deletes ({', '.join(modified_tests[:6])}"
                     f"{' ...' if len(modified_tests) > 6 else ''}) RESTORED to their base content, against "
                     f"this PR's code, in its own fresh tree; {len(base_tests.results)} result(s) read"
                     + ("; this PR changes no non-test code, so the probe accounts for each base test (relocated, "
                        "still exercised, deleted with its subject) and detects nothing" if test_only else ""))
        # FOURTEENTH CYCLE: the TRUNCATION repair. A base test that FAILS on this PR's code stops at its
        # first failing assertion, so its trace is only what it reached before that; a compound test
        # (assert add, then mul) that fails at add never enters mul, and a replacement of add alone
        # would clear it. The name-absent DETECTING base tests -- the only ones a replacement is ever
        # asked for -- are run once more on the BASE tree (the full revert), where they pass and run
        # to the end, and that trace is handed to the core beside the head one. Targeted: it runs only
        # when such tests exist, and only their files.
        head_names = {_fn_name(t) for t in head_results}
        absent_red = sorted(t for t, r in base_tests.results.items()
                            if t.split("::", 1)[0] in set(modified_tests) and _fn_name(t) not in head_names
                            and isinstance(r, dict) and r.get("status") == "fail")
        if absent_red and not test_only and probe["subjects"].get("state") == "measured":
            repair_files = sorted({t.split("::", 1)[0] for t in absent_red})
            try:
                repair = _phase("base-tests-on-base", patch=_full_revert(repo_dir, mb, head_sha, non_test, modified_tests),
                                trace=lambda _c, _ids=set(absent_red): _ids, **{**common, "test_files": repair_files})
                if repair.subjects is not None:
                    probe["subjects"][subject_trace_repair_key()] = repair.subjects
                    notes.append(f"subject trace: {len(absent_red)} base test(s) with no head test of their name that FAIL "
                                 f"against this PR's code were run once more on the BASE tree, where they pass, so their "
                                 f"whole subject (not only what they reached before failing) is what a replacement is measured "
                                 f"against")
                else:
                    notes.append(f"subject trace: the base-tree run of {len(absent_red)} deleted-and-detecting test(s) left no "
                                 f"trace ({repair.subjects_note}); their subject is what they reached on this PR's code before failing")
            except RunnerError as exc:
                notes.append(f"subject trace: the base-tree run of the deleted-and-detecting tests did not complete "
                             f"({str(exc)[:160]}); their subject is what they reached on this PR's code before failing")
        sb = probe["subjects"]
        notes.append("subject trace: " + ("measured -- the Action's own pytest plugin recorded, per test, the functions "
                                          "it entered during its call phase (both runs of the PR's test files in full; "
                                          "in the base-tests phase only the base tests no head test carries by name)"
                                          if sb.get("state") == "measured" else f"UNAVAILABLE ({sb.get('detail')}); a "
                                          f"renamed test cannot be told from a deleted one and C2 accuses there"))
        # The out-of-scope half. A test in a file the PR does NOT touch that fails BY ASSERTION in that
        # run is a test this PR's CODE breaks -- ordinary, and the repo's own suite is about to go red on
        # it. It becomes a QUESTION only if the PR's TEST diff is what makes it stop failing, and
        # answering that needs two more runs. They are CONDITIONAL: in a repo whose suite passes against
        # this PR's code the list is empty and neither runs. On a test-only PR the code is the same on
        # both sides, so the two phases would compare a tree with itself: not needed by construction.
        outside = _outside_reds(base_tests.results, set(modified_tests)) if not test_only else set()
        probe["suite_state"] = "not-needed"
        if outside:
            notes.append(f"runtime silencing probe: {len(outside)} test(s) in file(s) this PR does NOT touch "
                         f"fail by assertion against this PR's code ({', '.join(sorted(outside)[:4])}"
                         f"{' ...' if len(outside) > 4 else ''}) - two further phases decide whether this "
                         f"PR's TEST diff is what stops them failing")
            # FOURTEENTH CYCLE (verifier 13's X4 exposed it): the head tree does not HAVE the test
            # files this PR deletes -- pytest exits 4 on a missing path -- so the head-content phase
            # runs the suite minus them; the full-revert phase restores them and runs the whole suite.
            deleted_here = {q for st, q in changed if st == "D"}
            suite_head = _phase("suite-head", patch=None, **{**common, "test_files": [q for q in suite if q not in deleted_here]})
            suite_base = _phase("suite-base",
                                patch=_full_revert(repo_dir, mb, head_sha, non_test, modified_tests),
                                **{**common, "test_files": suite})
            probe["suite_state"] = "measured"
            probe["suite_head_on_head"] = suite_head.results
            probe["suite_base_on_base"] = suite_base.results
    except NoReport as nr:
        probe = {"state": "no-report", "detail": f"the base-tests phase exited {nr.exit_code} with no report"}
    except RunnerError as exc:
        probe = {"state": "runner-error", "detail": str(exc)[:300]}
    if probe["state"] != "measured":
        notes.append(f"runtime silencing probe: DID NOT RUN ({probe['state']}: {probe.get('detail', '')}) — "
                     f"C2 is told so and narrows its own success sentence accordingly")
    return probe


def _test_only_probe(*, repo_dir: str, mb: str, head_sha: str, family: str, test_command: list[str], timeout_s: int,
                     workdir: str | None, runs: list[dict], test_files: list[str], modified_tests: list[str],
                     all_tests: list[str], changed, globs: list[str]) -> dict:
    """FOURTEENTH CYCLE (verifier 13, F / D09 / D03 / item 2a). A PR that changes NO non-test code used
    to get NO probe at all, so a pure `git mv` of a test file was three syntactic accusations with
    nothing to answer them. What is TRUE of such a PR: it cannot hide a regression in code it does not
    change, but it CAN remove a test from the suite. What is MEASURABLE: whether every base test in the
    files it touches still runs, is relocated by name, or is still exercised by a test it runs. So the
    with-change run of its own test files (traced) and the wide base-tests phase run; no revert exists,
    so the with-change results stand in for both head runs and the core is told `no_non_test_change`."""
    if family not in runners.SUPPORTED_FAMILIES or not modified_tests:
        return {"state": "no-non-test-change"}
    common = dict(repo_dir=repo_dir, head_sha=head_sha, family=family, test_command=test_command,
                  test_files=test_files, timeout_s=timeout_s, workdir=workdir, runs=runs)
    notes: list[str] = []
    try:
        with_ = _phase("with-change", patch=None, trace=True, **common) if test_files else None
    except NoReport as nr:
        return {"state": "no-report", "detail": f"the with-change run exited {nr.exit_code} with no report"}
    except RunnerError as exc:
        return {"state": "runner-error", "detail": str(exc)[:300]}
    probe = _base_probe(common=common, modified_tests=modified_tests, all_tests=all_tests, changed=changed, globs=globs,
                        with_=with_, without=None, notes=notes, repo_dir=repo_dir, mb=mb, head_sha=head_sha,
                        non_test=[], test_only=True)
    if probe.get("state") == "measured":
        probe["detail"] = "; ".join(notes)[:400]
    return probe


SUBTEST_OBSERVATION_TAIL = (
    "pytest counts every OUTCOME in the <testsuite> attributes but writes one <testcase> per COLLECTED TEST, so a run "
    "using stdlib unittest.subTest() counts more than it itemises. This is an OBSERVATION, not a disagreement: no "
    "verdict turns on it, and it is not read as a forged report BECAUSE every <testcase> in each of those runs "
    "reconciles one-for-one with the ids the collect-only pre-pass listed on that same tree. RESIDUAL: a PASSING "
    "subtest leaves no element in the junit report at all, so the surplus is counted here, never itemised; a surplus "
    "is REFUSED as inconsistent whenever that id comparison is unavailable or disagrees, and the other three "
    "<testsuite> attributes are compared in both directions unchanged."
)


def reconcile_observation_notes(runs: list[dict]) -> list[str]:
    """ISSUE #217. Every phase's reconciliation observations, on the receipt, with the phase named.
    A surplus that was TOLERATED is a thing the reader is told about, not a thing quietly dropped: a
    comparison that can pass without saying what it excused is the absence of a question."""
    seen = [(r.get("label"), line) for r in runs for line in (r.get("reconcile_observations") or [])]
    if not seen:
        return []
    return ["reconciliation OBSERVATION — " + "; ".join(f"{label}: {line}" for label, line in seen)
            + ". " + SUBTEST_OBSERVATION_TAIL]


def run_c1(*, repo_dir: str, base_sha: str, head_sha: str, test_globs: str, family: str,
           test_command: list[str], timeout_s: int, workdir: str | None = None) -> C1Outcome:
    globs = split_globs(test_globs)
    if family not in runners.SUPPORTED_FAMILIES:
        return C1Outcome("not_run", reason=f"runner family {family!r} is NOT SUPPORTED (launch support: "
                                            f"{', '.join(runners.SUPPORTED_FAMILIES)}) — see the support matrix",
                            probe={"state": "runner-unsupported"})
    mb = gitio.merge_base(repo_dir, base_sha, head_sha)
    changed = gitio.changed_files(repo_dir, mb, head_sha)
    test_files, infra, non_test = split_changed(changed, globs)
    modified_tests = modified_test_files(changed, globs)
    # SCOPE IS DERIVED, NEVER SEARCHED: the suite is whatever git tracks that matches the
    # caller's own test globs, at the merge-base or at head -- never a hand-typed list.
    tracked = gitio.tracked_paths(repo_dir, mb, head_sha)
    all_tests = sorted(q for q in tracked if is_test_path(q, globs))
    # ISSUE #202. The repository's own declaration narrows all three, and the SAME filter narrows each:
    # the wide base-tests phase runs `all_tests`, so a file excluded from the PR's run but left in the
    # suite would crash the probe on exactly the sub-package C1 just declined to run.
    selection = build_selection(repo_dir, head_sha, test_globs, tracked, family)
    test_files = selection.keep(test_files)
    modified_tests = selection.keep(modified_tests)
    all_tests = selection.keep(all_tests)
    selection_note = selection.note()
    infra_note = (f"; {len(infra)} test-infrastructure file(s) changed and kept at head on every tree (never reverted; "
                  f"audited by C2): {', '.join(infra)}") if infra else ""
    infra_note += "; " + selection_note
    if not test_files:
        # ISSUE #202. The reason must name the REAL cause. "changed no test files matching the globs"
        # is false when files matched and the repository's own declaration is what put them out of
        # scope, and a receipt that contradicts its own body is the one thing this product cannot ship.
        if selection.excluded:
            reason = (f"every changed test file matching the globs {', '.join(globs)} "
                      f"({len(selection.excluded)} of them) is outside the suite this repository declares for "
                      f"itself — nothing IN THIS REPOSITORY'S OWN SUITE for red-on-revert to run "
                      f"({len(changed)} file(s) changed){infra_note}")
        else:
            reason = (f"the PR changed no test files matching the globs {', '.join(globs)} "
                      f"({len(changed)} file(s) changed) — nothing for red-on-revert to run{infra_note}")
        if not modified_tests or not non_test:
            return C1Outcome("not_run", reason=reason, non_test_files=non_test, infra_files=infra,
                             probe={"state": "no-modified-test-files"})
        # FOURTEENTH CYCLE: the PR DELETES test files and changes code, and adds or modifies no test. C1
        # has nothing to run with the change, but the runtime probe has its whole question: does the
        # deleted tests' BASE content detect this PR's code? The wide base-tests phase runs on its own.
        runs = []
        notes: list[str] = []
        probe = _base_probe(common=dict(repo_dir=repo_dir, head_sha=head_sha, family=family, test_command=test_command,
                                        test_files=[], timeout_s=timeout_s, workdir=workdir, runs=runs),
                            modified_tests=modified_tests, all_tests=all_tests, changed=changed, globs=globs,
                            with_=None, without=None, notes=notes, repo_dir=repo_dir, mb=mb, head_sha=head_sha,
                            non_test=non_test, test_only=False)
        if probe.get("state") == "measured":
            probe["detail"] = "; ".join(notes)[:400]
        return C1Outcome("not_run", reason=reason, runs=runs, non_test_files=non_test, infra_files=infra, probe=probe)
    if not non_test:
        reason = (f"the PR changed no non-test files ({len(test_files)} test file(s)"
                  f"{' and ' + str(len(infra)) + ' test-infrastructure file(s)' if infra else ''} only) — "
                  f"nothing to revert, so no test can be red-on-revert{infra_note}")
        runs: list[dict] = []
        probe = _test_only_probe(repo_dir=repo_dir, mb=mb, head_sha=head_sha, family=family, test_command=test_command,
                                 timeout_s=timeout_s, workdir=workdir, runs=runs, test_files=test_files,
                                 modified_tests=modified_tests, all_tests=all_tests, changed=changed, globs=globs)
        return C1Outcome("not_run", reason=reason, runs=runs, test_files=test_files, infra_files=infra, probe=probe)

    patch = gitio.diff(repo_dir, head_sha, mb, paths=non_test, binary=True, renames=False, as_bytes=True)
    comp = patch_composition(patch)
    if not patch.strip() or comp["text_lines"] == 0:
        what = []
        if comp["mode_only"]:
            what.append("mode-only: " + ", ".join(comp["mode_only"]))
        if comp["binary"]:
            what.append("binary: " + ", ".join(comp["binary"]))
        if comp["symlink"]:
            what.append("symlink: " + ", ".join(comp["symlink"]))
        reason = ("nothing executable to revert: the reverse patch of the non-test files changes no "
                  "text line (" + ("; ".join(what) or "empty patch") + f"){infra_note}")
        runs = []
        probe = _test_only_probe(repo_dir=repo_dir, mb=mb, head_sha=head_sha, family=family, test_command=test_command,
                                 timeout_s=timeout_s, workdir=workdir, runs=runs, test_files=test_files,
                                 modified_tests=modified_tests, all_tests=all_tests, changed=changed, globs=globs)
        return C1Outcome("not_run", reason=reason, runs=runs, test_files=test_files, non_test_files=non_test,
                         infra_files=infra, probe=probe)

    notes = [
        "isolation: a fresh git worktree per phase (with-change / without-change / rerun), its .git link removed before "
        "anything runs, a fresh TMPDIR per run, the report written outside the tree; "
        + ", ".join(OUTCOME_AFFECTING_ENV) + " dropped from the environment. NOT isolated "
        "(stated residual): state a test persists outside the tree and TMPDIR — HOME / XDG dirs, the original checkout path, "
        "the network — can still carry between phases",
        f"reverted diff: {comp['files']} file(s), {comp['text_lines']} text line(s) ({comp['source_text_lines']} in Python/JS source): "
        + ", ".join(comp["text_files"][:8]) + (" ..." if len(comp["text_files"]) > 8 else "")
        + ("; mode-only (not reverted as text): " + ", ".join(comp["mode_only"]) if comp["mode_only"] else "")
        + ("; binary: " + ", ".join(comp["binary"]) if comp["binary"] else "")
        + ("; symlink: " + ", ".join(comp["symlink"]) if comp["symlink"] else ""),
    ]
    if comp["source_text_lines"] == 0:
        notes.append("note: no Python/JS source changes in the reverted diff (data/config/template only)")
    was_set = dropped_env()
    if was_set:
        notes.append(f"environment: {', '.join(was_set)} WAS SET in this job's environment and was dropped from every phase — it "
                     f"registers plugins or injects flags into every run, and C1 never reverts it, so it would have run on the "
                     f"reverted tree too")
    if infra:
        notes.append(f"test infrastructure kept at head on every tree (never reverted; audited by C2): {', '.join(infra)}")
    notes.append(selection_note)

    runs: list[dict] = []
    common = dict(repo_dir=repo_dir, head_sha=head_sha, family=family, test_command=test_command,
                  test_files=test_files, timeout_s=timeout_s, workdir=workdir, runs=runs)
    # FOURTEENTH CYCLE: both runs of the PR's OWN test files are subject-traced in full (bounded by
    # the PR's changed test files), so a replacement can be linked to the test it replaces.
    with_ = _phase("with-change", patch=None, trace=True, **common)
    try:
        without = _phase("without-change", patch=patch, trace=True, **common)
    except NoReport as nr:
        # TENTH CYCLE: the `<session>` entry is `<kind>: <reason>` and the kind comes from the core's
        # closed vocabulary (c1_red_on_revert.SESSION_NOTE_KINDS), which the CORE enforces — this
        # adapter no longer decides the shape of the one line a reader uses to judge a NOT_RUN.
        session = (f"no-report: reverted-tree run exited {nr.exit_code} with no per-test results; output tail: "
                   f"{' | '.join(nr.tail[-5:])[:300]}")
        notes.append("reconciliation: with-change run consistent (exit code, report counts, collected ids agree); "
                     "without-change run produced no report")
        return C1Outcome("inputs", inputs={
            "base_sha": mb, "head_sha": head_sha,
            "test_results_with_change": with_.results, "test_results_without_change": {},
            "collection_errors_with_change": with_.coll, "collection_errors_without_change": {},
            "test_failure_kinds_without_change": {"<session>": session},
            # The reverted run produced nothing, so "is this test still a witness?" — the half of the
            # runtime rule that keeps it off an honest bugfix-with-updated-test — cannot be answered.
            # Say so rather than accuse from half a comparison.
            "execution_notes": notes,
            "runner_family": family,
            "tracked_files": gitio.tracked_paths(repo_dir, mb, head_sha),
            **({"frame_attribution": FRAME_ATTRIBUTION[family]} if FRAME_ATTRIBUTION.get(family) else {}),
        }, runs=runs, test_files=test_files, non_test_files=non_test, infra_files=infra,
           probe={"state": "no-report", "detail": "the reverted-tree run wrote no report, so a silenced test "
                                                  "could not be told from a witness"})
    inputs = {
        "base_sha": mb, "head_sha": head_sha,
        "test_results_with_change": with_.results, "test_results_without_change": without.results,
        "collection_errors_with_change": with_.coll, "collection_errors_without_change": without.coll,
        "test_failure_kinds_without_change": _failure_kinds(without.results),
        # SIXTH CYCLE (verifier 5, F8 / F1). Both are inputs to guards that live in the CORE, not
        # here. `runner_family` lets check_c1 refuse a frame waiver for pytest ITSELF instead of
        # trusting this module's FRAME_ATTRIBUTION map to never offer one; `tracked_files` lets the
        # own-frame allowlist detect an origin string that names more than one real file, which is how
        # a helper module's assertion affirmed as the test's own (end-to-end repro V5E16).
        #
        # SEVENTH CYCLE (verifier 6, F1, second half). This passed `sorted(test_files)` — the PR's
        # CHANGED TEST FILES — under the name `tracked_files`, so the core's ambiguity test was asking
        # about the wrong set entirely. The set it needs is the REPOSITORY's: an origin is ambiguous
        # because two files git tracks could both answer to it, and a changed-file list can almost
        # never contain the second one. The guard was reachable and toothless, which is worse than
        # absent, because the receipt claimed it had been applied.
        "runner_family": family,
        "tracked_files": gitio.tracked_paths(repo_dir, mb, head_sha),
    }
    if FRAME_ATTRIBUTION.get(family):
        inputs["frame_attribution"] = FRAME_ATTRIBUTION[family]
    any_red = any(v.get("status") in ("fail", "error") for v in without.results.values()) or bool(without.coll)
    phases = 2
    if any_red:
        rerun = _phase("without-change-rerun", patch=patch, **common)
        inputs["test_results_without_change_rerun"] = rerun.results
        phases = 3
    if with_.collected is not None and without.collected is not None:
        level = "collected ids" if (with_.collected.ids is not None and without.collected.ids is not None) else \
                "per-file collected counts (the repo's addopts made pytest print counts, not ids)"
        def _n(cr):
            if cr.n is None:
                return "no listing (nothing collected and nothing parseable)"
            return f"{cr.n}" + (f" ({len(cr.errors)} file(s) errored at collection)" if cr.errors else "")
        notes.append(f"reconciliation: {phases} run(s) consistent — exit code, report counts and {level} agree in each; "
                     f"collected {_n(with_.collected)} with the change, {_n(without.collected)} on the reverted tree")
    else:
        notes.append(f"reconciliation: {phases} run(s) consistent — exit code and report counts agree in each; "
                     f"collected-id reconciliation not available for {family}")
    # ---- PHASE "base-tests": the run nobody had -------------------------------------------------
    # The BASE content of the test files this PR MODIFIES, executed against THIS PR's code. It answers
    # the one question C1's two trees never asked: did the test, as it stood before the diff, catch
    # this change? A test that caught it before and does not catch it after is silenced, and that is
    # true of every spelling because it reads no spelling. See corund_checks/runtime_silencing.py.
    probe = _base_probe(common=common, modified_tests=modified_tests, all_tests=all_tests, changed=changed, globs=globs,
                        with_=with_, without=without, notes=notes, repo_dir=repo_dir, mb=mb, head_sha=head_sha,
                        non_test=non_test, test_only=False)

    notes.extend(reconcile_observation_notes(runs))
    inputs["execution_notes"] = notes
    return C1Outcome("inputs", inputs=inputs, runs=runs, test_files=test_files, non_test_files=non_test,
                     infra_files=infra, probe=probe)
