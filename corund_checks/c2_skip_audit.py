"""C2 — skip-audit over a unified diff and the BASE tree's loud-skip allowlist.

A silent test skip becomes a failure unless it is on a loud-skip allowlist committed in the repo,
with a reason. This module audits what the PR's diff ADDS (skips, xfails, dead gates, constant-true
assertions, runner escapes, discovery narrowing) and what it DELETES or SILENCES (assertions, whole
tests, collectability), across four NAMED detection tiers (every pattern's coverage is stated;
unstated = not covered):

  A  in the test file itself           B  relocated into conftest.py
  C  relocated into runner config      D  relocated into the CI workflow command / matrix
     (pytest.ini, pyproject, setup.cfg, tox.ini; jest/vitest config, package.json)

Three detection layers, each published per pattern (`via` in the matrix):
  regex   a line pattern over ADDED lines (always runs; the cross-check)
  ast     Python `ast` over the NEW side of each changed Python file (aliases resolved, constants
          folded, exact targets, before/after collectability) — `corund_checks.pyast_audit`
  token   a tokenizer pass over JS/TS test files (member chains through .skip/['skip']/.each(...))
          — `corund_checks.js_audit`
The ast/token layers read `files_after` / `files_before` when the caller supplies them (the Action
does); without them they read the diff's NEW side where it parses, and the receipt says which.

Verdicts:  FAILED         any finding outside the allowlist
           GAMED_SUSPECT  an aimable marker (skip/xfail/constant-true/runner-escape/loosened) targets a
                          test C1 needed — `c1_needed_tests` if the caller supplied it, else the
                          tests this same diff adds or modifies
           PROVEN         the diff was audited and adds nothing silent (the evidence says how much)
           NOT_RUN        the diff is empty, or has no text line to audit (binary / mode-only / a
                          pure rename with nothing to report) — nothing to audit is not a pass
           CRASHED        (runner) the text is not a diff, the allowlist is malformed, ...

Test-infrastructure files (conftest.py, pytest.ini, pyproject.toml, setup.cfg, tox.ini, jest/vitest
config, package.json, CI workflows) are REVERT-PROTECTED SURFACE (owner's ruling): C1 never reverts
them, and every change to their test-affecting content is a C2 finding here (tiers B-D).

DETECTION_MATRIX and NOT_COVERED are DATA; checks/README.md's matrix is checked against them by
checks/tests/test_readme_matrix.py, so the published coverage can never drift from the code.
The allowlist must be the BASE tree's: entries a PR adds in the same diff do not count until they
are merged, and `skip_allowlist_path` lets this module name such an edit. An allowlist entry is
honoured only if it names a path (a literal path segment, optionally `::test` / `:line`) AND
carries a reason; `*`, `**`, `?*`, a bare kind name or a reason-less entry is REFUSED and reported.
NEW module.
"""
from __future__ import annotations

import fnmatch
import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from . import js_audit, pyast_audit, runtime_silencing
from .unidiff import DiffLine, FileDiff, parse_unified_diff
from .verdict import MissingInput, Verdict

# ------------------------------------------------------------------------------ file classes

TEST_PY, CONFTEST, PY_CONFIG, TEST_JS, JS_CONFIG, WORKFLOW, OTHER_PY, OTHER_JS, OTHER = (
    "test-py", "conftest", "py-config", "test-js", "js-config", "workflow", "other-py", "other-js", "other")
_PY_CONFIG_NAMES = {"pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", ".pytest.ini"}
_JS_CONFIG_RE = re.compile(r"^(jest\.config\.(js|cjs|mjs|ts|json)|vitest\.config\.(js|cjs|mjs|ts|mts)|"
                           r"vite\.config\.(js|cjs|mjs|ts|mts)|package\.json|babel\.config\.(js|cjs|json))$")
_JS_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")
_TEST_JS_RE = re.compile(r"(\.(test|spec)\.(js|jsx|ts|tsx|mjs|cjs|mts|cts)$)|(^|/)__tests__/")
_TEST_DIRS_ANY = {"tests", "test", "__tests__"}
_TEST_DIRS_TOP = {"testing", "spec", "specs", "unit_tests", "integration_tests", "functional_tests", "e2e", "acceptance"}


def _in_test_dir(parts: list[str]) -> bool:
    """`tests/`, `test/`, `__tests__/` anywhere; `testing/`, `spec/`, `e2e/`... only as the repository's top-level
    directory (pkg/testing/ is a public helper module, numpy.testing-style)."""
    dirs = parts[:-1]
    return any(p in _TEST_DIRS_ANY for p in dirs) or (bool(dirs) and dirs[0] in _TEST_DIRS_TOP)
_PY_TEST_FILE_RE = re.compile(r"(^test_.*\.py$)|(_tests?\.py$)|(^tests?\.py$)|(_spec\.py$)|(^spec_.*\.py$)")
_PY_COLLECTABLE_FILE_RE = re.compile(r"(^test_.*\.py$)|(_test\.py$)")      # pytest's default python_files
_DEF_TEST_TOP_RE = re.compile(r"^(async\s+)?def\s+test\w*\s*\(|^class\s+Test\w*\b")


def classify_path(path: str) -> str:
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    parts = low.split("/")
    if low.startswith(".github/workflows/") and base.endswith((".yml", ".yaml")):
        return WORKFLOW
    if base == "conftest.py":
        return CONFTEST
    if base in _PY_CONFIG_NAMES:
        return PY_CONFIG
    if _JS_CONFIG_RE.match(base):
        return JS_CONFIG
    if base.endswith((".py", ".pyi")):
        in_tests_dir = _in_test_dir(parts)
        if _PY_TEST_FILE_RE.search(base) or in_tests_dir:
            return TEST_PY
        return OTHER_PY
    if base.endswith(_JS_EXT):
        return TEST_JS if _TEST_JS_RE.search(low) else OTHER_JS
    return OTHER


def is_test_infrastructure(path: str) -> bool:
    """The revert-protected surface: files C1 never reverts and whose test-affecting changes are
    C2 (tiers B-D) / C3 findings. Also exported to the Action."""
    cls = classify_path(path)
    if cls in (CONFTEST, PY_CONFIG, JS_CONFIG, WORKFLOW):
        return True
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    parts = low.split("/")
    return base == "__init__.py" and _in_test_dir(parts)


def _tier_for(cls: str) -> str:
    return {CONFTEST: "B", PY_CONFIG: "C", JS_CONFIG: "C", WORKFLOW: "D"}.get(cls, "A")


# ------------------------------------------------------------------------- detection matrix

@dataclass(frozen=True)
class Pattern:
    tier: str                      # A | B | C | D
    kind: str                      # short slug printed on the receipt
    family: str                    # pytest | jest | any
    files: frozenset[str]          # file classes the pattern applies to
    side: str                      # "+" scans added lines; "-" is a deletion rule (net-computed); "=" structural
    regex: re.Pattern | None
    aimable: bool                  # can be aimed at one test (GAMED_SUSPECT candidate)
    marker: str | None             # what C1 should hear about it: skip | xfail | constant-true | runner-escape | loosened
    description: str
    via: str = "regex"             # regex | ast | token | regex+ast | regex+token | parser
    coverage: str = "covered"
    observation: bool = False      # printed as an OBSERVATION line; never a finding, never blocks


def _p(tier, kind, family, files, rx, aimable=False, marker=None, side="+", description="", via="regex", coverage="covered",
       observation=False):
    return Pattern(tier, kind, family, frozenset(files), side, re.compile(rx) if rx else None, aimable,
                   marker, description, via, coverage, observation)


_PY_ANY = (TEST_PY, OTHER_PY, CONFTEST)
_JS_ANY = (TEST_JS, OTHER_JS)

DETECTION_MATRIX: tuple[Pattern, ...] = (
    # ---- Tier A: in the test file (also fires for helper modules and conftest, re-tiered by file class)
    _p("A", "skip-call", "pytest", _PY_ANY, r"\bpytest\.skip\s*\(|\bpytest\s*\.\s*skip\s*\\?\s*$", True, "skip",
       description="pytest.skip(...) call added (aliases `import pytest as p`, `from pytest import skip`, getattr, "
                   "`raise pytest.skip.Exception`, `_pytest.outcomes.Skipped` resolved by the AST tier)", via="regex+ast"),
    _p("A", "importorskip", "pytest", _PY_ANY, r"\bpytest\.importorskip\b", True, "skip",
       description="pytest.importorskip(...) — optional dependency the runner may never install (aliases via AST)", via="regex+ast"),
    _p("A", "xfail-call", "pytest", _PY_ANY, r"\bpytest\.xfail\b", True, "xfail",
       description="pytest.xfail(...) call added (aliases via AST)", via="regex+ast"),
    _p("A", "mark-skip", "pytest", _PY_ANY, r"@\s*\(?\s*pytest\s*\.\s*mark\s*\.\s*skip\b(?!if)|=\s*pytest\.mark\.skip\s*\(\s*test", True, "skip",
       description="@pytest.mark.skip decorator added (parenthesised, spaced, line-continued, `mark = pytest.mark`, "
                   "getattr, and `test_a = pytest.mark.skip(test_a)` resolved by the AST tier)", via="regex+ast"),
    _p("A", "mark-skipif", "pytest", _PY_ANY, r"@\s*\(?\s*pytest\s*\.\s*mark\s*\.\s*skipif\b", True, "skip",
       description="@pytest.mark.skipif decorator added (fires on the condition CI always satisfies)", via="regex+ast"),
    _p("A", "mark-xfail", "pytest", _PY_ANY, r"@\s*\(?\s*pytest\s*\.\s*mark\s*\.\s*xfail\b", True, "xfail",
       description="@pytest.mark.xfail decorator added — including xfail(strict=True) on a condition the reverted tree "
                   "satisfies (its XPASS is not an assertion red for C1)", via="regex+ast"),
    _p("A", "pytestmark", "pytest", _PY_ANY, r"^\s*pytestmark\s*=.*pytest\.mark\.(skip|skipif|xfail)\b", True, "skip",
       description="module-level pytestmark skip/xfail — silences every test in the file (multi-line lists via AST)", via="regex+ast"),
    _p("A", "unittest-skip", "pytest", _PY_ANY,
       r"(@\s*(unittest\.)?(skip|skipIf|skipUnless|expectedFailure)\s*\(?)|(\.skipTest\s*\()|(raise\s+(unittest\s*\.\s*)?SkipTest\b)", True, "skip",
       description="unittest skip decorator / skipTest / SkipTest / expectedFailure added (aliases via AST)", via="regex+ast"),
    _p("A", "empty-parametrize", "pytest", (TEST_PY, CONFTEST), r"parametrize\s*\([^)]*[\[(]\s*[\])]", True, "skip",
       description="@pytest.mark.parametrize(..., []) — zero cases, reported as skipped", via="regex+ast"),
    _p("A", "dead-gate", "pytest", (TEST_PY,), r"^\s*(if|while)\s+(False|0|None|\(\s*\)|\[\s*\]|\"\"|''|1\s*==\s*2|TYPE_CHECKING)\s*(:|\band\b)", True, "constant-true",
       description="`if False:` / `if 0:` / `if None:` / `if 1 == 2:` / `if TYPE_CHECKING:` gate added around test code "
                   "(any constant-false expression via AST)", via="regex+ast"),
    _p("A", "constant-true", "pytest", (TEST_PY,), r"^\s*assert\s+\(?\s*(True|1|not\s+(False|0|None)|\.\.\.)\s*\)?\s*(,.*)?(#.*)?$", True, "constant-true",
       description="`assert True`-shaped assertion added — passes on every tree. The AST tier folds a GRAMMAR, not a "
                   "list of shapes: literals and displays, `*`-unpacking, operators, comparison CHAINS, f-strings with "
                   "specs and conversions, subscripts and slices, `:=`, ternaries, `x == x`, `assertTrue(True)`, "
                   "`assertEqual(1, 1)`, and the pure builtins `bool`/`len`/`str`/`repr`/`any`/`all`. It is NOT total: "
                   "the expression node types it refuses are named one by one in the Not-covered block below, and so is "
                   "its size budget", via="regex+ast"),
    _p("A", "early-return", "pytest", (TEST_PY,), None, True, "constant-true",
       description="a `return` before the first assertion of a test body (unconditional or under a condition) — the test passes by not "
                   "running. TWO tiers: the AST tier sees every shape; a narrow regex tier sees a bare `return` as the FIRST statement "
                   "of a `def test*` body (blank lines, comments and a docstring skipped) and still fires when a pathological "
                   "expression has disarmed the AST",
       via="ast+regex"),
    _p("A", "runner-escape", "any", (TEST_PY, TEST_JS), r"\b(os\._exit|sys\.exit|pytest\.exit|process\.exit|pytest\.main|atexit\.register)\s*\(|raise\s+SystemExit\b|junitxml|xmlpath|--outputFile|PYTEST_ADDOPTS|\bsys\.argv\b",
       True, "runner-escape",
       description="a test that can end the runner or reach its report: os._exit / sys.exit / pytest.exit / process.exit / "
                   "raise SystemExit / pytest.main / atexit (also via `from os import *`, sys.modules['os'], os.__dict__[...], "
                   "getattr on a sensitive module), a report-path token (junitxml, xmlpath, outputFile, PYTEST_ADDOPTS) or a "
                   "sys.argv read in test code. C1 refuses such a test as a witness (GAMED-SUSPECT)", via="regex+ast+token"),
    _p("A", "dynamic-code", "pytest", (TEST_PY, CONFTEST), None, True, "runner-escape",
       description="exec / eval / compile, or an attribute resolved from a non-constant string, in test code — the call cannot be audited",
       via="ast"),
    _p("A", "jest-skip", "jest", _JS_ANY, r"\b(test|it|describe)\.skip\s*\(|\bx(it|test|describe)\s*\(", True, "skip",
       description="test.skip / it.skip / describe.skip / xit / xtest added (chains: test.concurrent.skip, it.each(...).skip, "
                   "describe.skip.each, newline before the call, `const t = it` aliases via the token tier). A COMPUTED member is "
                   "resolved when it is a constant string: `it['skip']`, `it[`skip`]`, `it['sk' + 'ip']`, and escaped forms "
                   "(`it['\\x73kip']`, `it['\\u0073kip']`). A non-constant computed member is left unresolved, never guessed",
       via="regex+token"),
    _p("A", "jest-todo", "jest", _JS_ANY, r"\b(test|it)\.todo\s*\(", True, "todo",
       description="test.todo / it.todo added — a test that never runs", via="regex+token"),
    _p("A", "jest-only", "jest", _JS_ANY, r"\b(test|it|describe)\.only\s*\(|\bf(it|describe|test)\s*\(", False, None,
       description="test.only / describe.only / fit added — focuses the run, silently skipping everything else", via="regex+token"),
    _p("A", "jest-failing", "jest", _JS_ANY, None, True, "xfail",
       description="test.failing — passes only when the body throws; an inverted assertion", via="token"),
    _p("A", "jest-zero-assertions", "jest", _JS_ANY, r"\bexpect\.assertions\s*\(\s*0\s*\)", True, "constant-true",
       description="expect.assertions(0) added"),
    _p("A", "assertion-deleted", "any", (TEST_PY, TEST_JS), None, side="-",
       description="net loss of assertion lines in a test file (deleted minus added, per file; expect.hasAssertions / "
                   "expect.assertions count as assertion lines)"),
    _p("A", "test-deleted", "any", (TEST_PY, TEST_JS), None, side="-",
       description="a test function/case removed (a rename — deleted def immediately re-added — is not a deletion)"),
    _p("A", "test-file-renamed-away", "any", (TEST_PY, TEST_JS), None, side="=",
       description="a test file renamed to a name the runner no longer collects (tests/test_x.py -> tests/x_helpers.py; "
                   "git rename detection emits no hunks, so this is read from the rename header)", via="parser"),
    _p("A", "test-package-removed", "pytest", (TEST_PY,), None, side="=",
       description="an `__init__.py` under a test directory deleted (changes rootdir-relative module names and what collects)",
       via="parser"),
    # TWELFTH CYCLE (verifier 11's nine spellings). The ONLY row in this matrix that reads no syntax
    # at all: `via="runtime"`. Every other row answers a spelling, and a twelfth spelling always
    # existed; this one compares two RUNS of the same test and is therefore the same rule for all of
    # them. See corund_checks/runtime_silencing.py for the standard of proof and why it does NOT fire
    # on a bugfix that updates its test, on an environment-gated skip, or on a test the PR adds.
    _p("A", "test-silenced-at-runtime", "any", (TEST_PY, TEST_JS), None, False, None, side="=", via="runtime",
       description="MEASURED, not parsed: the BASE version of a test in a MODIFIED test file fails BY ASSERTION "
                   "against this PR's code -- it detects the change -- and this PR's version of that same test does "
                   "not (it is skipped, never collected, or green on both trees), while being no witness and no "
                   "caught regression. Catches any silencing spelling, including ones no pattern here names: "
                   "`del test_x`, a shadowing second def, a lambda or `__code__` rebind, `globals().pop`, a "
                   "swallowing decorator, an assertion moved into a nested def nobody calls or into an unconsumed "
                   "generator expression, a `pytest.skip` reached through a container. Where this PR holds more than "
                   "one test of that name and the counterpart cannot be identified, C2 says so as an "
                   "OBSERVATION and makes no finding. Needs `silencing_probe`"),
    _p("A", "test-decollected", "pytest", (TEST_PY,), None, True, "skip", side="=",
       description="a test that was collectable before and is defined but NOT collectable after: class Test* renamed, "
                   "def indented into a function, moved into a non-Test class, `__test__ = False`, `if TYPE_CHECKING:` "
                   "(needs `files_before`; the diff alone shows the def lines)", via="ast"),
    _p("A", "test-not-collectable", "pytest", (TEST_PY,), None, True, "skip", side="=",
       description="in a NEW test file, a `def test*` NO class anywhere can collect: nested inside a function, "
                   "behind a constant-false gate, or in a module or class switched off with `__test__ = False`. "
                   "A `def test*` in a plain class is NOT this — pytest collects it through any `Test*` or "
                   "`unittest.TestCase` subclass that inherits it, and C2 resolves that across the PR's files "
                   "before it accuses (`test-collection-unresolved` is what it says when it cannot)", via="ast"),
    _p("A", "test-collection-unresolved", "pytest", (TEST_PY,), None, False, None, side="=", observation=True,
       description="in a NEW test file, a `def test*` in a class pytest does not collect BY NAME, where no class "
                   "in this PR's files inherits it either. C2 cannot see files the PR did not change, so it "
                   "reports what it knows and does NOT conclude the test never runs — the mainstream contract-test "
                   "base whose `class TestAdd(AddContract)` lives in an unchanged file lands here. An OBSERVATION: "
                   "printed, never a finding, never blocking", via="ast"),
    _p("A", "test-collection-resolved", "pytest", (TEST_PY,), None, False, None, side="=", observation=True,
       description="in a NEW test file, a `def test*` in a class pytest does not collect BY NAME that a `Test*` or "
                   "`unittest.TestCase` class elsewhere in this PR's files inherits: the receipt NAMES the collecting "
                   "class and its file. An OBSERVATION: the resolution stated as a checkable fact (eleventh cycle, "
                   "V10-F2 — a resolution that left no trace could not be proven alive)", via="ast"),
    _p("A", "test-under-condition", "pytest", (TEST_PY,), None, True, "skip", side="=",
       description="a test defined under a non-constant module-level `if` (defined only when the condition holds)", via="ast"),
    _p("A", "assertion-loosened", "any", (TEST_PY, TEST_JS), None, True, "loosened", side="-",
       description="an equality assertion on X replaced by a weaker assertion that still mentions X: membership, truthiness, "
                   "not-None, `>=`, `!=`, isinstance, str(X), assertTrue/In, mock assert_called_once_with -> assert_called, "
                   "toBeTruthy/toBeDefined/toContain/not.toBeNull/.not.toBe(<other>), expect.anything()", via="regex+ast+token"),
    _p("A", "raises-widened", "pytest", (TEST_PY,), None, True, "loosened", side="-",
       description="pytest.raises(Specific) -> pytest.raises(Exception/BaseException), or a `match=` dropped", via="ast"),
    _p("A", "assertion-swallowed", "pytest", (TEST_PY,), r"contextlib\.suppress\(\s*(AssertionError|Exception|BaseException)", True, "loosened",
       description="an assertion inside try/except (AssertionError | Exception | bare) that does not re-raise, or inside "
                   "contextlib.suppress(...) — also in a helper module under a test directory", via="regex+ast"),
    _p("A", "assertion-moved-under-condition", "pytest", (TEST_PY,), None, True, "loosened", side="-",
       description="a deleted unconditional assertion re-added under a non-constant `if` (e.g. an environment flag)", via="ast"),
    _p("A", "files-after-mismatch", "any", (TEST_PY, TEST_JS, CONFTEST, OTHER_PY), None, False, None, side="=",
       description="`files_after` disagrees with the diff (an added line of the diff is absent from the supplied file text). The "
                   "diff's added lines are ground truth: the AST/token tier read the diff's reconstructed NEW side instead, and the "
                   "disagreement itself is reported", via="parser"),
    # ---- Tier B: conftest relocation (ANY hook is a finding — the owner's ruling: conftest is revert-protected surface)
    _p("B", "conftest-hook", "pytest", (CONFTEST,),
       r"^\s*(async\s+)?def\s+pytest_(?!(generate_tests|addoption|report_header|terminal_summary|make_parametrize_id|addhooks|plugin_registered|report_collectionfinish|html_\w+)\b)\w+"
       r"|^\s*pytest_(?!(generate_tests|addoption|report_header|terminal_summary|make_parametrize_id|addhooks|plugin_registered|report_collectionfinish|html_\w+)\b)\w+\s*=",
       False, None,
       description="an OUTCOME-AFFECTING pytest_* hook defined (def or assignment/lambda) or modified in conftest.py — it can change "
                   "collection or outcomes (runtest_setup skip, makereport forcing passed, pycollect_makeitem -> [], collection_finish "
                   "clearing items, load_initial_conftests injecting args, configure registering plugins). Also a C1 CONTAMINATION: "
                   "the changed conftest runs during the reverted phase, so C1 refuses every witness (UNPROVEN-contaminated)", via="regex+ast"),
    _p("B", "infra-observed", "pytest", (CONFTEST, TEST_PY, OTHER_PY), None, False, None, side="=", observation=True,
       description="a `pytest_*` hook, an autouse fixture or a `pytest_plugins` assignment whose BODY C2 either DECIDED "
                   "cannot silence, drop or decide the outcome of a test (marker registration only, a reorder, a print, "
                   "`pass`/`yield`, seeding `random`, resetting an object) or COULD NOT DECIDE from this file (a call into "
                   "another file, file I/O, a computed attribute, a patch of a third-party name, `pytest_plugins` itself). "
                   "An OBSERVATION that says which — never a finding (eleventh cycle, verifier 10 V10-F1: FAILED only on a "
                   "PROVEN silencing). Still a C1 CONTAMINATION: PR-authored infrastructure runs during the reverted phase",
       via="ast"),
    _p("B", "conftest-parametrize-hook", "pytest", (CONFTEST,),
       r"^\s*(async\s+)?def\s+pytest_(generate_tests|addoption|report_header|terminal_summary|make_parametrize_id|addhooks|plugin_registered|report_collectionfinish|html_\w+)\b"
       r"|^\s*pytest_(generate_tests|addoption|report_header|terminal_summary|make_parametrize_id|addhooks|plugin_registered|report_collectionfinish|html_\w+)\s*=",
       False, None,
       description="OBSERVATION only: a parametrization / reporting hook (pytest_generate_tests, pytest_addoption, report_header, "
                   "terminal_summary, make_parametrize_id) added in conftest.py — noted on the receipt, not a finding, not a C1 contamination",
       via="regex+ast", observation=True),
    _p("B", "conftest-autouse", "pytest", (CONFTEST,), r"autouse\s*=\s*True", False, None,
       description="an autouse fixture added in conftest.py (runs for every test; can monkeypatch the code under test). "
                   "A C1 contamination: witnesses are refused while the changed conftest runs during the reverted phase", via="regex+ast"),
    _p("B", "conftest-plugins", "pytest", (CONFTEST,), r"^\s*pytest_plugins\s*=", False, None,
       description="pytest_plugins assignment — loads arbitrary plugin modules", via="regex+ast"),
    _p("B", "conftest-collect-ignore", "pytest", (CONFTEST,), r"\bcollect_ignore(_glob)?\b", False, None,
       description="collect_ignore / collect_ignore_glob added in conftest.py (also via setattr on the module)", via="regex+ast"),
    _p("B", "conftest-add-marker", "pytest", (CONFTEST,), r"add_marker\s*\(\s*(pytest\.mark\.(skip|skipif|xfail)\b|['\"](skip|skipif|xfail)['\"])", False, None,
       description="item.add_marker(pytest.mark.skip/xfail) or add_marker('skip') added in conftest.py", via="regex+ast"),
    _p("B", "conftest-mark-skip", "pytest", (CONFTEST,), r"\bpytest\.mark\.(skip|skipif|xfail)\b(?!.*add_marker)|\bpytest\.skip\b|\bSkipped\b|skip\.Exception", False, None,
       description="pytest.mark.skip/skipif/xfail, pytest.skip, raise Skipped referenced in conftest.py", via="regex+ast"),
    _p("B", "conftest-dynamic-attr", "pytest", (CONFTEST,), r"setattr\s*\(\s*sys\.modules|globals\s*\(\s*\)\s*\[", False, None,
       description="setattr(sys.modules[__name__], ...) / globals()[...] in conftest.py — a module attribute built from a string", via="regex+ast"),
    _p("B", "conftest-internal-import", "pytest", (CONFTEST,), r"^\s*(from|import)\s+_pytest\b", False, None,
       description="an import from pytest's private `_pytest` package (outcomes, mark generator) in conftest.py", via="regex+ast"),
    # ---- Tier B, fourth cycle: the same infrastructure OUTSIDE conftest.py. pytest calls a pytest_* hook
    # from ANY loaded module, and a kept test file is loaded on every tree including the reverted one.
    _p("B", "plugin-hook", "pytest", (TEST_PY, OTHER_PY),
       r"^\s*(async\s+)?def\s+pytest_(?!(generate_tests|addoption|report_header|terminal_summary|make_parametrize_id|addhooks|plugin_registered|report_collectionfinish|html_\w+)\b)\w+"
       r"|^\s*pytest_(?!(generate_tests|addoption|report_header|terminal_summary|make_parametrize_id|addhooks|plugin_registered|report_collectionfinish|html_\w+|plugins\b))\w+\s*=",
       False, None,
       description="an OUTCOME-AFFECTING pytest_* hook defined or modified in a file that is NOT a conftest (a kept test file, a "
                   "helper module). Once the module is loaded — as a test module, via `-p <mod>`, `pytest_plugins`, PYTEST_PLUGINS or "
                   "a pytest11 entry point — pytest calls the hook for every test, and the file is never reverted. This is the shape "
                   "that forged a C1 witness in the third adversarial pass (EN3): its origin IS the test's own module. A C1 CONTAMINATION",
       via="regex+ast"),
    _p("B", "plugin-parametrize-hook", "pytest", (TEST_PY, OTHER_PY),
       r"^\s*(async\s+)?def\s+pytest_(generate_tests|addoption|report_header|terminal_summary|make_parametrize_id|addhooks|plugin_registered|report_collectionfinish|html_\w+)\b",
       False, None,
       description="OBSERVATION only: a parametrization / reporting hook in a non-conftest file — the same carve-out conftest.py gets",
       via="regex+ast", observation=True),
    _p("B", "plugin-autouse", "pytest", (TEST_PY, OTHER_PY), r"autouse\s*=\s*True", False, None,
       description="an autouse fixture in a file that is NOT a conftest — it runs for every test in its scope without any test asking, "
                   "and can monkeypatch the code under test on the reverted tree too. A C1 CONTAMINATION", via="regex+ast"),
    _p("B", "plugin-plugins", "pytest", (TEST_PY, OTHER_PY), r"^\s*pytest_plugins\s*=", False, None,
       description="a pytest_plugins assignment outside conftest.py — loads arbitrary plugin modules. A C1 CONTAMINATION", via="regex+ast"),
    # ---- Tier C: runner-config relocation (values are TOKENISED: INI continuation lines, TOML lists, `-pno:x`, `-k=expr`)
    _p("C", "config-ignore", "pytest", (PY_CONFIG,), None, False, None,
       description="--ignore / --ignore-glob added to addopts", via="token"),
    _p("C", "config-deselect", "pytest", (PY_CONFIG,), None, False, None,
       description="--deselect added to addopts", via="token"),
    _p("C", "config-k-m", "pytest", (PY_CONFIG,), None, False, None,
       description="-k / -m expression added to addopts (string, TOML list, `-k=expr`, a bare `-m`)", via="token"),
    _p("C", "config-no-plugin", "pytest", (PY_CONFIG,), None, False, None,
       description="-p no:<plugin> added to addopts (`-p no:x` and `-pno:x`)", via="token"),
    _p("C", "config-register-plugin", "pytest", (PY_CONFIG,), None, False, None,
       description="-p <module> added to addopts — REGISTERS a plugin (the opposite of `-p no:`). Whatever that module hooks runs on "
                   "every tree, the reverted one included, and no conftest.py names it. A C1 CONTAMINATION", via="token"),
    _p("C", "config-entrypoint-plugin", "pytest", (PY_CONFIG,), r"pytest11", False, None,
       description="a `pytest11` entry point declared in the packaging metadata (pyproject/setup.cfg/setup.py) — the installed package "
                   "registers a plugin with nothing in addopts or a conftest naming it. A C1 CONTAMINATION"),
    _p("C", "config-plugins-env", "pytest", (PY_CONFIG, WORKFLOW, JS_CONFIG, OTHER), r"\bPYTEST_PLUGINS\b", False, None,
       description="the PYTEST_PLUGINS environment variable set anywhere the run can see it (a workflow env, a config, a script) — it "
                   "registers plugin modules with no file in the repo naming them. A C1 CONTAMINATION"),
    _p("C", "config-collect-only", "pytest", (PY_CONFIG,), None, False, None,
       description="--collect-only / --co in addopts — collects, runs nothing", via="token"),
    _p("C", "config-narrowing-flag", "pytest", (PY_CONFIG,), None, False, None,
       description="a run-narrowing or config-overriding flag in addopts: -x / --exitfirst / --maxfail / --lf / --ff / --sw / "
                   "--stepwise / -o / --override-ini / -c / --rootdir / --confcutdir / --noconftest / --continue-on-collection-errors off",
       via="token"),
    _p("C", "config-discovery", "pytest", (PY_CONFIG,),
       r"^\s*(norecursedirs|python_files|python_classes|python_functions)\s*=", False, None,
       description="norecursedirs / python_files / python_classes / python_functions changed"),
    _p("C", "config-testpaths-shrink", "pytest", (PY_CONFIG,), r"^\s*testpaths\s*=", False, None,
       description="testpaths lost an entry (tokenised: quoted or unquoted, INI or TOML list; compared with the deleted line in the same hunk)",
       via="token"),
    _p("C", "config-option-changed", "pytest", (PY_CONFIG,),
       r"^\s*(pythonpath|usefixtures|required_plugins|minversion|xfail_strict|empty_parameter_set_mark|collect_ignore(_glob)?|"
       r"consider_namespace_packages|rootdir|confcutdir|addopts)\s*=", False, None,
       description="a collection- or import-affecting pytest option changed (pythonpath, usefixtures, required_plugins, minversion, "
                   "xfail_strict, empty_parameter_set_mark, collect_ignore, any addopts line) — test infrastructure is revert-protected, so its changes are reported"),
    _p("C", "jest-config-ignore", "jest", (JS_CONFIG,),
       r"\b(testPathIgnorePatterns|modulePathIgnorePatterns|testIgnore|coveragePathIgnorePatterns)\b", False, None,
       description="jest/vitest ignore pattern added in config"),
    _p("C", "jest-config-match", "jest", (JS_CONFIG,), r"\b(testMatch|testRegex|include|exclude|roots|projects|testEnvironment|setupFiles(AfterEach)?|globalSetup)\s*[:=]", False, None,
       description="testMatch / testRegex / include / exclude / roots / projects / setupFiles / globalSetup changed in config"),
    _p("C", "npm-test-script-changed", "jest", (JS_CONFIG,), r"\"(pre)?test(:\w+)?\"\s*:\s*\"", False, None,
       description="package.json `test` script line changed (narrowing flags, a different runner, `|| true`)"),
    # ---- Tier D: CI workflow relocation
    _p("D", "workflow-ignore", "any", (WORKFLOW,), None, False, None,
       description="--ignore / --ignore-glob / --deselect / --testPathIgnorePatterns / --testNamePattern in a workflow command", via="token"),
    _p("D", "workflow-k-m", "any", (WORKFLOW,), None, False, None,
       description="-k / -m (pytest) or -t (jest) expression in a workflow command, `-k expr` or `-k=expr`", via="token"),
    _p("D", "workflow-no-plugin", "pytest", (WORKFLOW,), None, False, None,
       description="-p no:<plugin> in a workflow command", via="token"),
    _p("D", "workflow-collect-only", "pytest", (WORKFLOW,), None, False, None,
       description="--collect-only / --co in a workflow command (collects, runs nothing)", via="token"),
    _p("D", "workflow-path-narrowing", "any", (WORKFLOW,), None, False, None,
       description="a runner command that had no positional paths gained some (pytest -> pytest tests/test_other.py)", via="token"),
    _p("D", "workflow-env-addopts", "any", (WORKFLOW,), r"\bPYTEST_ADDOPTS\b|\bPYTEST_PLUGINS\b|\bJEST_\w+\b|\bVITEST_\w+\b", False, None,
       description="PYTEST_ADDOPTS / PYTEST_PLUGINS / JEST_* / VITEST_* set in `env:` or inline before the runner"),
    _p("D", "workflow-swallowed-exit", "any", (WORKFLOW,),
       r"(pytest|jest|vitest|npm\s+(run\s+)?test|yarn\s+test|pnpm\s+test|npx\s+\w+|tox|nox|make\s+test).*(\|\|\s*(true|:|echo)|;\s*exit\s+0|\|\s*tee\b|2>\s*/dev/null)|^\s*set\s+\+e\b",
       False, None,
       description="the runner's exit code swallowed: `|| true`, `|| :`, `; exit 0`, `| tee` (pipe status lost), `set +e`"),
    _p("D", "workflow-matrix-exclude", "any", (WORKFLOW,), r"^\s*exclude:\s*(\[|$)", False, None,
       description="matrix `exclude:` block or inline list added"),
    _p("D", "workflow-continue-on-error", "any", (WORKFLOW,), r"^\s*-?\s*continue-on-error\s*:\s*(?!(false|no|off|0|'false'|\"false\")\s*$)\S", False, None,
       description="continue-on-error with any value that is not literally false (`true`, `True`, `'true'`, `${{ true }}`, an "
                   "expression) added (reported here; C3 decides whether it folds a gate)"),
    _p("D", "workflow-if-false", "any", (WORKFLOW,), r"^\s*-?\s*if:\s*(false|0|'false'|\"false\"|\$\{\{\s*false\s*\}\})\s*$", False, None,
       description="`if: false` (any constant-false spelling) on a job or step in a workflow"),
)

NOT_COVERED_BASE: tuple[str, ...] = (
    "narrowing flags placed in a Makefile, tox target, or shell wrapper the workflow calls (only pytest.ini/pyproject/setup.cfg/tox.ini [pytest] sections and .github/workflows are read)",
    "a test step removed from a workflow (C3 covers gate-job deletion)",
    "an `if:` on a step whose expression is not a constant (reported by C3 only at job level, as an OBSERVATION)",
    "a decorator whose target `def` is outside the hunk AND `files_after` is absent: the finding is attributed to the file, not the test",
    "a git-quoted path is decoded; a path git could not represent (invalid UTF-8) is kept as decoded-with-replacement and may not match an allowlist entry",
    "a test that persists state OUTSIDE the execution tree and its per-run TMPDIR (HOME, the original checkout via GITHUB_WORKSPACE, the network) can still carry state between C1's runs — a sandbox is required to close this",
    "a test that forges a self-consistent report, exit code and collected id WITHOUT any runner-escape token in the diff (the token list is stated in the `runner-escape` row) — C1's reconciliation cannot distinguish it from a real run",
    "`test-silenced-at-runtime` judges the test files the PR MODIFIES or DELETES; for the CROSS-FILE question it also runs every other test file git tracks under the caller's test globs. A test the PR ADDS has no base version to compare against and is out of scope by construction. A test file git does not track, or one outside those globs, is never run at all -- and this residual was REWRITTEN in the thirteenth cycle because the twelfth cycle's version claimed a test in an untouched file `was not silenced by this diff`, which is a factual claim and a false one (verifier 12's D3 falsified it by silencing exactly such a test)",
    "`test-silenced-at-runtime` proves nothing about a test whose base version does not fail BY ASSERTION against the PR's code -- one skipped by its environment (a docker/platform gate, importorskip), one whose red fires in the SETUP phase (a fixture's assertion) and one whose red fires in a frame belonging to a file THIS PR ALSO CHANGES (that red could have been authored to manufacture the accusation). That is the standard of proof, and the price of it is that a silencing of such a test is not caught by this rule. A red in a frame belonging to a file the PR does NOT change IS accepted -- the twelfth cycle refused those too, and a shared assertion helper one file over walked straight through",
    "`test-silenced-at-runtime` says NOT DECIDED, by name, about a test whose base version ERRORS in its own frame because what it called no longer exists on the PR's code (the PR renamed or removed the subject -- verifier 12's D5) or because the signature changed (a TypeError at the call site): a test of the new name cannot be run on the reverted tree at all, so whether it still pins behaviour cannot be measured, and a renamed subject cannot be told from an unrelated addition. It neither accuses nor clears there. This line was NARROWED in the fourteenth cycle: the thirteenth cycle folded `renamed` into the standard-of-proof line above in the same cycle that failed to close D5, and that widening is withdrawn here and stated on its own",
    "`test-silenced-at-runtime` clears a base test with no counterpart of its own name ONLY by a REPLACEMENT: a test of a new name that detects the change (fails by assertion on the reverted tree, or cannot run there because what it calls does not exist there) and ENTERS the changed functions the base test entered, compared on the same tree via the Action's subject trace. The thirteenth cycle's GLOBAL clearance (any live witness anywhere cleared any absent test) is withdrawn: it re-opened the twelfth cycle's N2 (verifier 13, X1/X2/X5/X6). Stated residuals of the linked rule: a witness that merely CALLS the deleted test's subject without asserting on it reads as its replacement; a base test that entered no changed function (a constant, a module attribute, a thread, a fixture) is NOT DECIDED; where the subject trace is unavailable (jest/vitest, or a pytest run the plugin could not join) a rename cannot be told from a deletion and Corund accuses, with the allowlist as the answer",
    "the tier-A kinds `test-deleted`, `assertion-deleted`, `skip-call`, `mark-skipif` and `importorskip` DEFER to the runtime probe where it can point at a positive fact, PER TEST: the test is relocated by name; replaced by a witness of the same subject; still exercised by a test the PR runs; deleted with the subject it covered (its base version errors because the symbol is gone and the PR adds nothing in that file); or an UNDECIDED account, which defers to a NOT DECIDED observation; and for the skip family, every base test in the file still executes with the change. A PR that changes NO non-test code is measured too (the base-tests phase runs; nothing can detect, so only relocation / exercise / subject-deleted apply). `mark-xfail` / `xfail-call` do NOT defer: an xfail keeps the test running and discards its verdict, so the loud form is the base-tree allowlist, and the receipt says so beside the finding. Where the probe did not run, or has no such fact, the syntactic finding stands -- so a repo whose probe cannot run keeps the syntactic answer and is told so on the receipt",
    "`test-silenced-at-runtime` does not run at all unless the caller supplies `silencing_probe` (only the Action does); every other C2 caller is told so on the receipt, and C2's success sentence is downgraded to a claim about the diff's text alone",
    "the test command must import the code under test from the execution tree (rootdir / pythonpath / a src layout); an editable install of the original checkout is not reverted and every such test is green-on-revert",
)
NOT_COVERED: tuple[str, ...] = tuple(dict.fromkeys(NOT_COVERED_BASE + pyast_audit.NOT_COVERED_AST + js_audit.NOT_COVERED_JS))

_ASSERT_PY = re.compile(r"^\s*(assert\b|self\.assert\w*\(|pytest\.raises\(|with\s+pytest\.raises\(|\w+\.assert_(called|awaited|any_call|has_calls|not_called)\w*\()")
_ASSERT_JS = re.compile(r"^\s*(expect\s*\(|expect\.(hasAssertions|assertions)\s*\(|assert\.\w+\(|assert\()")
_DEF_PY = re.compile(r"^\s*(async\s+)?def\s+(test\w*)\s*\(")
_DEF_JS = re.compile(r"^\s*(it|test)\s*\(\s*([\"'`])(.*?)\2")
_DEF_JS_ANY = re.compile(r"^\s*(it|test)(\.\w+)?\s*\(\s*([\"'`])(.*?)\3")
_DECORATOR = re.compile(r"^\s*@")
_NOISE = re.compile(r"^\s*($|#|//|import\b|from\b|\"\"\"|''')")
_QUOTED = re.compile(r"[\"']([^\"']+)[\"']")
_RUNNER_WORD_RE = re.compile(r"\b(pytest|py\.test|jest|vitest|tox|nox)\b|\bnpm\s+(run\s+)?test\b|\byarn\s+test\b|\bpnpm\s+test\b|\bnpx\s+(jest|vitest)\b")
_PY_KEY_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*=\s*(.*)$")
_YAML_RUN_RE = re.compile(r"^\s*-?\s*run\s*:\s*(.*)$")
_YAML_ENV_ADDOPTS_RE = re.compile(r"^\s*(PYTEST_ADDOPTS|PYTEST_PLUGINS)\s*:\s*(.*)$")
_VALUE_FLAGS = {"-k", "-m", "-p", "-o", "-c", "-n", "-W", "-r", "--deselect", "--ignore", "--ignore-glob", "--rootdir", "--confcutdir",
                "--junitxml", "--maxfail", "--tb", "--durations", "--override-ini", "--basetemp", "--import-mode", "--testNamePattern", "-t",
                "--testPathIgnorePatterns", "--outputFile", "--reporter", "--config"}


@dataclass(frozen=True)
class Finding:
    path: str
    lineno: int | None
    tier: str
    kind: str
    family: str
    snippet: str
    target: str | None            # path::name when known, None = file-level
    aimable: bool
    marker: str | None
    observation: bool = False

    @property
    def keys(self) -> tuple[str, ...]:
        """What an allowlist pattern may match: the path, `path:line`, `path::test`. NEVER the bare kind."""
        ks = [self.path, f"{self.path}:{self.lineno}" if self.lineno else self.path]
        if self.target:
            ks.append(f"{self.path}::{self.target}")
        return tuple(ks)

    def line(self) -> str:
        # TENTH CYCLE: an OBSERVATION names the test it concerns, but `aimed at` is the vocabulary
        # of an accusation (it is what a GAMED-SUSPECT marker does), and an observation accuses
        # nobody. A receipt that misdescribes its own work is the thing this product sells against.
        rel = "about" if self.observation else "aimed at"
        aim = f" -> {rel} {self.path}::{self.target}" if self.target else ""
        where = f"{self.path}:{self.lineno}" if self.lineno else self.path
        return f"  {where} [{self.tier}/{self.kind}] {self.snippet.strip()[:140]}{aim}"


# ------------------------------------------------------------------------- per-file walkers

def _new_side_defs(fd: FileDiff, cls: str) -> list[tuple[int, str, str]]:
    """(index-in-lines, name, side) for every test def on the NEW side (context or added)."""
    out = []
    for i, l in enumerate(fd.lines):
        if l.side == "-":
            continue
        if cls == TEST_PY:
            m = _DEF_PY.match(l.text)
            if m:
                out.append((i, m.group(2), l.side))
        elif cls == TEST_JS:
            m = _DEF_JS.match(l.text)
            if m:
                out.append((i, m.group(3), l.side))
    return out


def _target_for(fd: FileDiff, cls: str, idx: int, is_decorator: bool, defs) -> str | None:
    if cls not in (TEST_PY, TEST_JS):
        return None
    if cls == TEST_JS:
        m = _DEF_JS_ANY.match(fd.lines[idx].text)
        return m.group(4) if m else None
    if is_decorator:
        for i, name, _ in defs:
            if i > idx:
                between = fd.lines[idx + 1:i]
                if all(_DECORATOR.match(b.text) or not b.text.strip() or b.text.strip().startswith("#") for b in between):
                    return name
                return None
        return None
    prev = [name for i, name, _ in defs if i < idx]
    return prev[-1] if prev else None


def _needed_from_diff(fd: FileDiff, cls: str, marker_idx: set[int]) -> set[str]:
    """Tests C1 would need, derived from the diff itself: new test defs, plus existing tests whose
    body gained non-marker lines."""
    needed: set[str] = set()
    if cls not in (TEST_PY, TEST_JS):
        return needed
    defs = _new_side_defs(fd, cls)
    for i, name, side in defs:
        if side == "+":
            needed.add(f"{fd.path}::{name}")
    for idx, l in enumerate(fd.lines):
        if l.side != "+" or idx in marker_idx or _NOISE.match(l.text) or _DECORATOR.match(l.text):
            continue
        if cls == TEST_PY and _DEF_PY.match(l.text):
            continue
        prev = [name for i, name, _ in defs if i < idx]
        if prev:
            needed.add(f"{fd.path}::{prev[-1]}")
    return needed


def _deletion_findings(fd: FileDiff, cls: str, before_source: str | None = None,
                       after_source: str | None = None) -> list[Finding]:
    """`assertion-deleted` and `test-deleted`, read from the hunks.

    FOURTEENTH CYCLE (verifier 13, D14 / 2b). `assertion-deleted` used to be ONE file-level finding
    (net deleted assertions over the whole file), so a file that lost one test's assertions together
    with that test could never be answered per test by the runtime probe: the probe accounts for
    TESTS, and a file-level finding names none. When the OLD text is supplied the deleted assertion
    lines are attributed to the base function they belonged to (an AST span lookup on the base text,
    nothing more), the net is computed PER FUNCTION, and each function with a net loss gets its own
    finding carrying its target -- which the probe can then defer, or not, one test at a time. Without
    the old text the file-level net stands, exactly as before."""
    out: list[Finding] = []
    assert_re = _ASSERT_PY if cls == TEST_PY else _ASSERT_JS
    deleted_assert = [l for l in fd.lines if l.side == "-" and assert_re.match(l.text)]
    added_assert = [l for i, l in enumerate(fd.lines) if l.side == "+" and assert_re.match(l.text)]
    spans = _def_spans(before_source) if (cls == TEST_PY and isinstance(before_source, str)) else None
    spans_after = _def_spans(after_source) if (spans is not None and isinstance(after_source, str)) else None
    file_net = len(deleted_assert) - len(added_assert)
    # The FILE-level net is the gate, exactly as before: a file that lost no assertions overall (a
    # method moved between classes, an assertion rewritten) raises nothing. Only when the file lost
    # assertions are they attributed to the base functions that lost them.
    # a DELETED file has no after-text and adds nothing, so the base spans alone attribute its losses
    if spans is not None and deleted_assert and file_net > 0 and (spans_after is not None or not added_assert):
        per: dict[str | None, list[DiffLine]] = {}
        for l in deleted_assert:
            per.setdefault(_enclosing_def(spans, l.old_lineno), []).append(l)
        # an ADDED assertion is credited to the function it lands in on the NEW side (its enclosing def
        # in the after text), so a rewritten assertion is -1 +1 in the same test, net 0 -- and a test
        # that gained one while another lost one is read as exactly that
        added_per: dict[str | None, int] = {}
        for l in added_assert:
            fn = _enclosing_def(spans_after, l.new_lineno) if spans_after is not None else None
            added_per[fn] = added_per.get(fn, 0) + 1
        for fn, lines in sorted(per.items(), key=lambda kv: kv[1][0].old_lineno or 0):
            net = len(lines) - added_per.get(fn, 0)
            if net <= 0:
                continue
            first = lines[0]
            target = fn if (fn and fn.split("::")[-1].startswith("test")) else None
            out.append(Finding(fd.path, first.old_lineno, "A", "assertion-deleted", "any",
                               f"{len(lines)} assertion line(s) deleted from {fn or 'module level'}, "
                               f"{added_per.get(fn, 0)} added there (net -{net}); first: {first.text.strip()}",
                               target, False, None))
        if not out:                       # lost assertions that no base function owns: the file-level finding
            first = deleted_assert[0]
            out.append(Finding(fd.path, first.old_lineno, "A", "assertion-deleted", "any",
                               f"{len(deleted_assert)} assertion line(s) deleted, {len(added_assert)} added (net -{file_net}); "
                               f"first: {first.text.strip()}", None, False, None))
    else:
        net = file_net
        if net > 0:
            first = deleted_assert[0]
            out.append(Finding(fd.path, first.old_lineno, "A", "assertion-deleted", "any",
                               f"{len(deleted_assert)} assertion line(s) deleted, {len(added_assert)} added (net -{net}); "
                               f"first: {first.text.strip()}", None, False, None))
    def_re = _DEF_PY if cls == TEST_PY else _DEF_JS
    name_group = 2 if cls == TEST_PY else 3
    added_names = {def_re.match(l.text).group(name_group) for l in fd.lines if l.side == "+" and def_re.match(l.text)}
    lines = fd.lines
    for i, l in enumerate(lines):
        if l.side != "-":
            continue
        m = def_re.match(l.text)
        if not m:
            continue
        name = m.group(name_group)
        if name in added_names:
            continue
        nxt = next((x for x in lines[i + 1:] if x.side != " "), None)
        if nxt is not None and nxt.side == "+" and def_re.match(nxt.text):
            continue
        target = name
        if spans is not None and l.old_lineno is not None:
            q = _enclosing_def(spans, l.old_lineno)
            if q:
                target = q                     # `TestC::test_m` for a method, the probe's own spelling
        out.append(Finding(fd.path, l.old_lineno, "A", "test-deleted", "any",
                           f"test removed: {l.text.strip()}", target, False, None))
    return out


def _def_spans(source: str) -> "list[tuple[str, int, int]] | None":
    """(target, first line, last line) of every def in a TEST file's base text, methods spelled
    `Class::method` (the probe's target spelling). None when the text does not parse."""
    import ast as _ast
    try:
        tree = _ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    out: list[tuple[str, int, int]] = []

    def walk(node, prefix):
        for child in _ast.iter_child_nodes(node):
            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                out.append((f"{prefix}{child.name}", start, getattr(child, "end_lineno", child.lineno) or child.lineno))
            elif isinstance(child, _ast.ClassDef):
                walk(child, f"{prefix}{child.name}::")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def _enclosing_def(spans, lineno) -> str | None:
    if lineno is None:
        return None
    best = None
    for q, a, b in spans:
        if a <= lineno <= b and (best is None or (b - a) < (best[2] - best[1])):
            best = (q, a, b)
    return best[0] if best else None


_CLASS_RE = re.compile(r"^(\s*)class\s+(\w+)\b")
_DEF_ANY_RE = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)\s*\(")
_RAISES_RE = re.compile(r"pytest\.raises\(\s*([A-Za-z_][\w.]*)\s*(,\s*match\s*=)?")


def _diff_decollection_findings(fd: FileDiff) -> list[Finding]:
    """Structural silencing visible in the hunks alone (the AST tier does the whole-file comparison when
    files_before/files_after are supplied): a `class Test*` renamed off the Test prefix, and a
    `def test*` re-added at a deeper indent under a non-Test enclosing block that the hunk shows."""
    out: list[Finding] = []
    deleted_classes = [(l, _CLASS_RE.match(l.text)) for l in fd.lines if l.side == "-" and _CLASS_RE.match(l.text)]
    added_classes = [(l, _CLASS_RE.match(l.text)) for l in fd.lines if l.side == "+" and _CLASS_RE.match(l.text)]
    for dl, dm in deleted_classes:
        if not dm.group(2).startswith("Test"):
            continue
        for al, am in added_classes:
            if am.group(1) == dm.group(1) and not am.group(2).startswith("Test") and not any(
                    x.side == "+" and _CLASS_RE.match(x.text) and _CLASS_RE.match(x.text).group(2) == dm.group(2) for x in fd.lines):
                out.append(Finding(fd.path, al.new_lineno, "A", "test-decollected", "pytest",
                                   f"class {dm.group(2)} -> class {am.group(2)}: its test methods are no longer collected",
                                   dm.group(2), True, "skip"))
                break
    deleted_defs = {}
    for l in fd.lines:
        if l.side == "-":
            m = _DEF_ANY_RE.match(l.text)
            if m and m.group(3).startswith("test"):
                deleted_defs.setdefault(m.group(3), len(m.group(1)))
    for i, l in enumerate(fd.lines):
        if l.side != "+":
            continue
        m = _DEF_ANY_RE.match(l.text)
        if not m or m.group(3) not in deleted_defs:
            continue
        ind = len(m.group(1))
        if ind <= deleted_defs[m.group(3)]:
            continue
        # find the nearest enclosing NEW-side line with a smaller indent
        enclosing = None
        for x in reversed(fd.lines[:i]):
            if x.side == "-" or not x.text.strip():
                continue
            if len(x.text) - len(x.text.lstrip()) < ind:
                enclosing = x.text.strip()
                break
        if enclosing is None:
            continue
        cm = _CLASS_RE.match(enclosing)
        if cm and cm.group(2).startswith("Test"):
            continue
        out.append(Finding(fd.path, l.new_lineno, "A", "test-decollected", "pytest",
                           f"def {m.group(3)} re-indented under `{enclosing[:60]}` — no longer collected", m.group(3), True, "skip"))
    return out


def _diff_raises_widened(fd: FileDiff, defs) -> list[Finding]:
    out: list[Finding] = []
    deleted = [(_RAISES_RE.search(l.text), l) for l in fd.lines if l.side == "-" and _RAISES_RE.search(l.text)]
    if not deleted:
        return out
    for idx, l in enumerate(fd.lines):
        if l.side != "+":
            continue
        am = _RAISES_RE.search(l.text)
        if not am:
            continue
        prev = [name for i, name, _ in defs if i < idx]
        target = prev[-1] if prev else None
        exc, has_match = am.group(1), bool(am.group(2))
        for dm, dl in deleted:
            if exc in ("Exception", "BaseException") and dm.group(1) not in ("Exception", "BaseException"):
                out.append(Finding(fd.path, l.new_lineno, "A", "raises-widened", "pytest",
                                   f"{dl.text.strip()}  ->  {l.text.strip()}", target, True, "loosened"))
                break
            if dm.group(1) == exc and dm.group(2) and not has_match:
                out.append(Finding(fd.path, l.new_lineno, "A", "raises-widened", "pytest",
                                   f"{dl.text.strip()}  ->  {l.text.strip()} (match= dropped)", target, True, "loosened"))
                break
    return out


_EQ_PY = re.compile(r"^\s*assert\s+(.+?)\s*==\s*(.+?)\s*(,.*)?$")
_EQ_UNIT = re.compile(r"^\s*self\.assert(Equal|Equals|ListEqual|DictEqual|SetEqual)\(\s*(.+?)\s*,")
_EQ_JS = re.compile(r"^\s*expect\((.+?)\)\.(toBe|toEqual|toStrictEqual)\(")
_WEAK_PY = (
    re.compile(r"^\s*assert\s+(.+?)\s+in\s+.+$"),
    re.compile(r"^\s*assert\s+(.+?)\s+is\s+not\s+None\s*(,.*)?$"),
    re.compile(r"^\s*assert\s+(.+?)\s*!=\s*None\s*(,.*)?$"),
    re.compile(r"^\s*assert\s+bool\((.+?)\)\s*(,.*)?$"),
    re.compile(r"^\s*assert\s+len\((.+?)\)\s*>=?\s*0\s*(,.*)?$"),
    re.compile(r"^\s*assert\s+([^=<>!,]+?)\s*(,.*)?(#.*)?$"),    # bare truthiness, no operator
)
_WEAK_UNIT = re.compile(r"^\s*self\.assert(In|True|IsNotNone|Greater|GreaterEqual)\(\s*(.+?)\s*[,)]")
_WEAK_JS = re.compile(r"^\s*expect\((.+?)\)\.(toBeTruthy|toBeDefined|toContain|not\.toBeNull|not\.toBeUndefined|not\.toBe|not\.toEqual)\(")


def _norm(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def _lhs_strong(text: str, cls: str) -> str | None:
    if cls == TEST_PY:
        m = _EQ_PY.match(text)
        if m and " in " not in m.group(1):
            return _norm(m.group(1))
        m = _EQ_UNIT.match(text)
        return _norm(m.group(2)) if m else None
    m = _EQ_JS.match(text)
    return _norm(m.group(1)) if m else None


def _lhs_weak(text: str, cls: str) -> str | None:
    if cls == TEST_PY:
        if "==" in text or "!=" in text and "None" not in text:
            return None
        for rx in _WEAK_PY:
            m = rx.match(text)
            if m:
                return _norm(m.group(1))
        m = _WEAK_UNIT.match(text)
        return _norm(m.group(2)) if m else None
    m = _WEAK_JS.match(text)
    return _norm(m.group(1)) if m else None


def _loosening_findings(fd: FileDiff, cls: str, defs) -> list[tuple[Finding, int]]:
    """A deleted equality assertion on X paired with an added weaker assertion on the same X."""
    out: list[tuple[Finding, int]] = []
    strong = {}
    for l in fd.lines:
        if l.side == "-":
            lhs = _lhs_strong(l.text, cls)
            if lhs:
                strong.setdefault(lhs, l)
    if not strong:
        return out
    for idx, l in enumerate(fd.lines):
        if l.side != "+":
            continue
        lhs = _lhs_weak(l.text, cls)
        if lhs and lhs in strong:
            prev = [name for i, name, _ in defs if i < idx]
            target = prev[-1] if prev else None
            out.append((Finding(fd.path, l.new_lineno, "A", "assertion-loosened", "any",
                                f"{strong[lhs].text.strip()}  ->  {l.text.strip()}", target, True, "loosened"), idx))
    return out


# ------------------------------------------------------------------ tokenised config / workflow

def _tokens(value: str) -> list[str]:
    v = value.strip()
    if v.startswith("[") and v.endswith("]"):
        try:
            arr = json.loads(v)
        except ValueError:
            try:
                arr = json.loads(v.replace("'", '"'))
            except ValueError:
                arr = None
        if isinstance(arr, list):
            return [str(x) for x in arr]
    try:
        return shlex.split(v, posix=True)
    except ValueError:
        return v.split()


def _flags(tokens: list[str]) -> tuple[set[str], list[str]]:
    """(normalised flags, positionals) from argv-like tokens: `--opt=v` -> --opt; `-k=expr` -> -k;
    `-pno:x` -> -p (value `no:x` kept as a positional-like extra for the no-plugin check)."""
    flags: set[str] = set()
    positionals: list[str] = []
    skip_next = False
    for t in tokens:
        if skip_next:
            skip_next = False
            continue
        if t.startswith("--"):
            name = t.split("=", 1)[0]
            flags.add(name)
            if "=" not in t and name in _VALUE_FLAGS:
                skip_next = True
            continue
        if t.startswith("-") and len(t) > 1:
            name = t[:2]
            flags.add(name)
            rest = t[2:]
            if rest.startswith("="):
                rest = rest[1:]
            if rest:
                if name == "-p" and rest.startswith("no:"):
                    flags.add("-p no:")
            elif name in _VALUE_FLAGS:
                skip_next = True
            continue
        positionals.append(t)
    if "-p" in flags:
        for i, t in enumerate(tokens):
            if t != "-p" or i + 1 >= len(tokens):
                continue
            # `-p no:x` DISABLES a plugin; `-p mod` REGISTERS one — the fourth cycle's blocker used the
            # second form, which nothing tokenised before.
            flags.add("-p no:" if tokens[i + 1].startswith("no:") else "-p mod")
    for t in tokens:
        if t.startswith("-p") and len(t) > 2 and not t.startswith("-p no:"):
            flags.add("-p no:" if t[2:].startswith("no:") else "-p mod")
    return flags, positionals


_NARROWING_FLAGS = {"-x", "--exitfirst", "--maxfail", "--lf", "--last-failed", "--ff", "--failed-first", "--sw", "--stepwise",
                    "-o", "--override-ini", "-c", "--rootdir", "--confcutdir", "--noconftest", "--nf", "--new-first"}


def _config_line_value(text: str) -> str | None:
    """The value part of an INI/TOML line, or the whole line for an INI continuation line."""
    m = _PY_KEY_RE.match(text)
    if m:
        return m.group(2)
    if text.startswith((" ", "\t")) and text.strip() and not text.strip().startswith(("#", ";", "[")):
        return text.strip()
    return None


def _config_findings(fd: FileDiff, l: DiffLine) -> list[tuple[str, str]]:
    """(kind, snippet) for one added py-config line, from its tokens."""
    value = _config_line_value(l.text)
    if value is None:
        return []
    key_m = _PY_KEY_RE.match(l.text)
    key = key_m.group(1) if key_m else None
    out: list[tuple[str, str]] = []
    if key == "testpaths":
        if _testpaths_shrank(fd, l):
            out.append(("config-testpaths-shrink", l.text))
        return out
    if key is not None and key != "addopts":
        return out
    toks = _tokens(value)
    flags, _ = _flags(toks)
    if flags & {"--ignore", "--ignore-glob"}:
        out.append(("config-ignore", l.text))
    if "--deselect" in flags:
        out.append(("config-deselect", l.text))
    if flags & {"-k", "-m"}:
        out.append(("config-k-m", l.text))
    if "-p no:" in flags:
        out.append(("config-no-plugin", l.text))
    if "-p mod" in flags:
        out.append(("config-register-plugin", l.text))
    if flags & {"--co", "--collect-only", "--collectonly"}:
        out.append(("config-collect-only", l.text))
    if flags & _NARROWING_FLAGS:
        out.append(("config-narrowing-flag", l.text))
    return out


def _workflow_findings(fd: FileDiff, l: DiffLine) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    m = _YAML_RUN_RE.match(l.text)
    env_m = _YAML_ENV_ADDOPTS_RE.match(l.text)
    cmd = m.group(1) if m else (env_m.group(2) if env_m else None)
    if cmd is None:
        # a continuation line of a block-scalar `run: |` that mentions the runner
        if _RUNNER_WORD_RE.search(l.text) and not l.text.strip().startswith("#"):
            cmd = l.text.strip()
        else:
            return out
    if not _RUNNER_WORD_RE.search(cmd) and not env_m:
        return out
    toks = _tokens(cmd)
    flags, positionals = _flags(toks)
    if flags & {"--ignore", "--ignore-glob", "--deselect", "--testPathIgnorePatterns", "--testNamePattern"}:
        out.append(("workflow-ignore", l.text))
    if flags & {"-k", "-m"} or ("-t" in flags and re.search(r"\b(jest|vitest)\b", cmd)):
        out.append(("workflow-k-m", l.text))
    if "-p no:" in flags:
        out.append(("workflow-no-plugin", l.text))
    if flags & {"--co", "--collect-only", "--collectonly"}:
        out.append(("workflow-collect-only", l.text))
    if m and _runner_positionals(toks):
        deleted_runs = [d for d in fd.lines if d.side == "-" and _YAML_RUN_RE.match(d.text) and _RUNNER_WORD_RE.search(d.text)]
        for d in deleted_runs:
            if not _runner_positionals(_tokens(_YAML_RUN_RE.match(d.text).group(1))):
                out.append(("workflow-path-narrowing", f"{d.text.strip()}  ->  {l.text.strip()}"))
                break
    return out


def _runner_positionals(tokens: list[str]) -> list[str]:
    """Positional args AFTER the runner word (paths / node ids), value-taking flags skipped."""
    idx = None
    for i, t in enumerate(tokens):
        if re.fullmatch(r"(python3?|py)\s*", t) and i + 2 < len(tokens) and tokens[i + 1] == "-m":
            idx = i + 3
            break
        if re.search(r"(^|/)(pytest|py\.test|jest|vitest)$", t):
            idx = i + 1
            break
    if idx is None:
        return []
    tail = tokens[idx:]
    tail = [t for t in tail if t not in ("||", "&&", ";", "|")]
    cut = len(tail)
    for i, t in enumerate(tail):
        if t in ("||", "&&", ";", "|") or t.startswith(("||", "&&", ";")):
            cut = i
            break
    _, positionals = _flags(tail[:cut])
    return positionals


def _testpaths_shrank(fd: FileDiff, added_line: DiffLine) -> bool:
    deleted = [l for l in fd.lines if l.side == "-" and re.match(r"^\s*testpaths\s*=", l.text)]
    if not deleted:
        return False
    def entries(text: str) -> set[str]:
        val = _config_line_value(text) or ""
        return set(_tokens(val))
    before = entries(deleted[-1].text)
    after = entries(added_line.text)
    return not before <= after


# ------------------------------------------------------------------------- the AST / token tiers

def _promoted_test_like(fd: FileDiff) -> bool:
    """A file not named like a test file but defining MODULE-LEVEL `def test*` / `class Test*` (a method named
    test_connection inside a production class is not a test)."""
    return any(_DEF_TEST_TOP_RE.match(l.text) for l in fd.lines)


def _collectable_filename(path: str, cls: str) -> bool:
    base = path.rsplit("/", 1)[-1].lower()
    if cls == TEST_PY:
        return bool(_PY_COLLECTABLE_FILE_RE.search(base))
    return bool(_TEST_JS_RE.search(path.lower()))


def _files_after_agrees(fd: FileDiff, text: str) -> list[str]:
    """The diff's added lines are ground truth: every non-empty added line must appear in the supplied
    NEW-side text. Returns the added lines that are missing (empty = agreement)."""
    have = {l.strip() for l in text.splitlines()}
    return [l.text.strip() for l in fd.added() if l.text.strip() and l.text.strip() not in have]


def _new_side_python(fd: FileDiff, files_after) -> tuple[str | None, str, dict[int, int | None] | None,
                                                          set[int] | None, list[str]]:
    """(source, origin, line_map, added_linenos, files_after mismatch) for one Python file's NEW side.

    Extracted in the tenth cycle so the two passes over the diff — the one that builds the
    cross-file inheritance corpus and the one that audits each file — read the SAME text by
    construction. Two copies of this resolution would be two answers to `what does the new side
    say`, and the checks are supposed to be deterministic comparisons."""
    missing: list[str] = []
    supplied = files_after is not None and fd.path in files_after and isinstance(files_after[fd.path], str)
    if supplied:
        missing = _files_after_agrees(fd, files_after[fd.path])
        if missing:
            supplied = False
    if supplied:
        added = None if fd.status == "added" else fd.added_new_linenos()
        return files_after[fd.path], "files_after", None, added, missing
    if fd.status == "deleted":
        return None, "deleted", None, None, missing
    recon_lines = [l for l in fd.lines if l.side != "-"]
    candidate = "\n".join(l.text for l in recon_lines) + "\n"
    for attempt in (candidate, pyast_audit._dedent(candidate), pyast_audit.repair_for_parse(candidate),
                    pyast_audit.repair_for_parse(pyast_audit._dedent(candidate))):
        if pyast_audit.parse_or_none(attempt) is not None:
            candidate = attempt
            break
    if pyast_audit.parse_or_none(candidate) is None:
        return None, "unparsable", None, None, missing
    line_map = {i + 1: l.new_lineno for i, l in enumerate(recon_lines)}
    added = None if fd.status == "added" else {i + 1 for i, l in enumerate(recon_lines) if l.side == "+"}
    return candidate, ("diff (added file)" if fd.status == "added" else "diff (hunks)"), line_map, added, missing


def _inheritance_corpus(files: list[FileDiff], files_after) -> "pyast_audit.InheritanceCorpus":
    """The inheritance edges of every Python file in this PR's input, in one object.

    TENTH CYCLE (verifier 9, V9-F1). C2 called a contract base class's `def test*` "never
    collected" while the `class TestAdd(AddContract)` that collects it sat in the next file of the
    same diff. The question is cross-file, so the answer is built from the whole input before any
    single file is audited. `files_after` is read for EVERY Python path the caller supplied — not
    only the changed ones — because more context here can only ever REMOVE an accusation."""
    corpus = pyast_audit.InheritanceCorpus()
    seen: set[str] = set()
    if isinstance(files_after, Mapping):
        for path, text in files_after.items():
            if not isinstance(path, str) or not isinstance(text, str) or not path.lower().endswith((".py", ".pyi")):
                continue
            tree = pyast_audit.parse_or_none(text)
            if tree is not None:
                corpus.add(tree, pyast_audit._Resolver(tree), path)
            seen.add(path)
    for fd in files:
        if fd.path in seen or not fd.path.lower().endswith((".py", ".pyi")):
            continue
        source, _, _, _, _ = _new_side_python(fd, files_after)
        if source is None:
            continue
        tree = pyast_audit.parse_or_none(source)
        if tree is not None:
            corpus.add(tree, pyast_audit._Resolver(tree), fd.path)
    return corpus


def _known_modules(files: list[FileDiff], files_after) -> frozenset[str]:
    """The top-level MODULE names this PR ships, derived from every Python path in the input (the
    diff and `files_after`), with the conventional `src/` / `lib/` prefixes stripped: `src/pkg/core.py`
    -> `pkg`, `app/cache.py` -> `app`. A patch aimed at one of these from an autouse fixture is a patch
    of the code under test (PROVEN outcome-affecting); a patch of any other name is not decided."""
    paths: set[str] = set()
    for fd in files:
        paths.add(fd.path)
        if fd.old_path:
            paths.add(fd.old_path)
    if isinstance(files_after, Mapping):
        paths.update(p for p in files_after if isinstance(p, str))
    out: set[str] = set()
    for p in paths:
        if not p.lower().endswith((".py", ".pyi")):
            continue
        parts = [x for x in p.replace("\\", "/").split("/") if x]
        while parts and parts[0] in ("src", "lib", "python", "."):
            parts = parts[1:]
        if not parts:
            continue
        top = parts[0]
        if top.endswith((".py", ".pyi")):
            top = top.rsplit(".", 1)[0]
        if top and top not in ("tests", "test", "testing", "conftest", "setup", "noxfile", "tasks"):
            out.add(top)
    return frozenset(out)


def _python_ast_tier(fd: FileDiff, cls: str, files_after, files_before, tier: str, family: str,
                     notes: list[str], inheritance=None, known_modules=frozenset()) -> tuple[list[Finding], set[str] | None]:
    """AST findings for one Python file + the AST-derived needed set (None = AST did not run)."""
    is_conftest = cls == CONFTEST
    extra: list[Finding] = []
    source, origin, line_map, added_linenos, missing = _new_side_python(fd, files_after)
    if missing:
        extra.append(Finding(fd.path, None, tier, "files-after-mismatch", "any",
                             f"files_after omits {len(missing)} added line(s) of the diff (first: {missing[0][:80]!r}) — the diff is "
                             f"ground truth; the AST tier read the diff's NEW side", None, False, None))
        notes.append(f"{fd.path}: files_after disagrees with the diff — reconstructed from the hunks instead")
    if source is None:
        if origin == "deleted":
            notes.append(f"{fd.path}: deleted — nothing on the new side to parse")
        else:
            notes.append(f"{fd.path}: NEW side not reconstructable from the hunks (regex tier only; pass files_after)")
        return extra, None
    before_source = None
    if files_before is not None and fd.status != "added" and isinstance(files_before.get(fd.old_path or fd.path), str):
        before_source = files_before[fd.old_path or fd.path]
    deleted_lines = [l.text for l in fd.deleted()]
    findings_ast, needed_ast, ast_notes = pyast_audit.audit_python(
        source, is_conftest=is_conftest, added_linenos=added_linenos, before_source=before_source,
        deleted_lines=deleted_lines, is_new_file=(fd.status == "added"), is_test_file=(cls == TEST_PY),
        inheritance=inheritance, known_modules=known_modules)
    for n in ast_notes:
        notes.append(f"{fd.path}: {n}")
    # SECOND TIER for early-return, independent of the AST (verifier 3, V3-C2-earlyreturn-under-deep-expr):
    # a 4000-term expression makes the AST tier give up, and early-return lived ONLY in the AST — the one
    # shape that needed a parser was invisible exactly when the parser was disarmed. A bare `return` as the
    # first statement of a `def test*` body cannot follow an assertion, so no flow analysis is needed.
    if cls == TEST_PY and not is_conftest:
        known = {(f.lineno, f.kind) for f in findings_ast}
        for lineno, test_name in pyast_audit.early_return_lines(source):
            if added_linenos is not None and lineno not in added_linenos:
                continue
            if (lineno, "early-return") in known:
                continue                              # the AST already found it; do not double-report
            ln = line_map.get(lineno) if line_map is not None else lineno
            extra.append(Finding(fd.path, ln, tier, "early-return", "pytest",
                                 f"return before any assertion in {test_name} — the test body is silenced "
                                 f"(found by the regex tier; the AST tier did not report it)",
                                 f"{fd.path}::{test_name}", True, "constant-true"))
    if before_source is None and fd.status != "added" and not is_conftest:
        notes.append(f"{fd.path}: files_before absent — collectability before/after not compared")
    notes.append(f"{fd.path}: AST from {origin}")
    allowed_kinds = None
    if cls == OTHER_PY:
        # ELEVENTH CYCLE: the infrastructure kinds are decided by the AST tier in a helper module too
        # (a `pytest_configure` in tests/plugins/markers.py is the same idiom as in conftest.py).
        allowed_kinds = {"skip-call", "importorskip", "xfail-call", "mark-skip", "mark-skipif", "mark-xfail", "pytestmark", "unittest-skip",
                         "plugin-hook", "plugin-autouse", "plugin-plugins", "plugin-parametrize-hook", "infra-observed"}
    out: list[Finding] = list(extra)
    for f in findings_ast:
        if allowed_kinds is not None and f.kind not in allowed_kinds:
            continue
        ln = f.lineno
        if line_map is not None:
            ln = line_map.get(f.lineno) or None
        target = f.target if cls in (TEST_PY,) else (f.target if not is_conftest else None)
        out.append(Finding(fd.path, ln, tier, f.kind, family, f.snippet, target, f.aimable, f.marker, f.observation))
    needed: set[str] = set()
    if cls == TEST_PY:
        finding_lines = {}
        for f in findings_ast:
            if f.target:
                finding_lines.setdefault(f.target, set()).add(f.lineno)
        for name, touched in needed_ast.items():
            if touched - finding_lines.get(name, set()):
                needed.add(f"{fd.path}::{name}")
    return out, needed


def _js_token_tier(fd: FileDiff, cls: str, files_after, tier: str, notes: list[str]) -> list[Finding]:
    extra: list[Finding] = []
    if files_after is not None and isinstance(files_after.get(fd.path), str) and _files_after_agrees(fd, files_after[fd.path]):
        missing = _files_after_agrees(fd, files_after[fd.path])
        extra.append(Finding(fd.path, None, tier, "files-after-mismatch", "any",
                             f"files_after omits {len(missing)} added line(s) of the diff — the diff is ground truth; the token tier read the diff's NEW side",
                             None, False, None))
        files_after = None
    if files_after is not None and isinstance(files_after.get(fd.path), str):
        source = files_after[fd.path]
        added = None if fd.status == "added" else fd.added_new_linenos()
        line_map = None
        notes.append(f"{fd.path}: token tier from files_after")
    elif fd.status == "deleted":
        return []
    else:
        recon = [l for l in fd.lines if l.side != "-"]
        source = "\n".join(l.text for l in recon) + "\n"
        added = None if fd.status == "added" else {i + 1 for i, l in enumerate(recon) if l.side == "+"}
        line_map = {i + 1: l.new_lineno for i, l in enumerate(recon)}
        notes.append(f"{fd.path}: token tier from the diff")
    out = list(extra)
    for f in js_audit.audit_js(source, added_linenos=added, deleted_lines=[l.text for l in fd.deleted()]):
        ln = line_map.get(f.lineno) if line_map else f.lineno
        aimable, marker = {"jest-skip": (True, "skip"), "jest-todo": (True, "todo"), "jest-only": (False, None),
                           "jest-failing": (True, "xfail"), "constant-true": (True, "constant-true"),
                           "assertion-loosened": (True, "loosened"), "runner-escape": (True, "runner-escape")}[f.kind]
        out.append(Finding(fd.path, ln, tier, f.kind, "jest", f.snippet, f.target, aimable, marker))
    return out


# ------------------------------------------------------------------------- the audit

def audit_diff(diff_text: str, files_after: Mapping[str, str] | None = None,
               files_before: Mapping[str, str] | None = None) -> tuple[list[Finding], set[str], dict[str, Any]]:
    """Every finding, the diff-derived C1-needed test set, and the audit counts + tier notes."""
    files = parse_unified_diff(diff_text)
    inheritance = _inheritance_corpus(files, files_after)
    known_modules = _known_modules(files, files_after)
    findings: list[Finding] = []
    needed: set[str] = set()
    notes: list[str] = []
    counts: dict[str, Any] = {"files": len(files), "added": 0, "deleted": 0, "test_files": 0, "text_lines": 0,
                              "binary_or_mode_only": [], "renamed": 0}
    for fd in files:
        cls = classify_path(fd.path)
        if cls == OTHER_PY and _promoted_test_like(fd):
            cls = TEST_PY
            notes.append(f"{fd.path}: not named like a test file but defines `def test*` — audited as a test file")
        counts["added"] += fd.added_count
        counts["deleted"] += fd.deleted_count
        counts["text_lines"] += fd.text_line_count
        if fd.binary or fd.mode_only:
            counts["binary_or_mode_only"].append(fd.path + (" (binary)" if fd.binary else " (mode only)"))
        if fd.status == "renamed":
            counts["renamed"] += 1
        if cls in (TEST_PY, TEST_JS):
            counts["test_files"] += 1
        tier = _tier_for(cls)
        defs = _new_side_defs(fd, cls)
        marker_idx: set[int] = set()

        # structural (parser) rules
        if fd.status == "renamed" and fd.old_path:
            old_cls = classify_path(fd.old_path)
            if old_cls in (TEST_PY, TEST_JS) and _collectable_filename(fd.old_path, old_cls) and not (
                    classify_path(fd.path) == old_cls and _collectable_filename(fd.path, old_cls)):
                findings.append(Finding(fd.old_path, None, "A", "test-file-renamed-away", "any",
                                        f"renamed {fd.old_path} -> {fd.path}: the runner no longer collects it", None, False, None))
        if fd.status == "deleted" and fd.path.lower().rsplit("/", 1)[-1] == "__init__.py" and \
                _in_test_dir(fd.path.lower().split("/")):
            findings.append(Finding(fd.path, None, "A", "test-package-removed", "pytest",
                                    f"{fd.path} deleted — test package removed", None, False, None))

        regex_from = len(findings)                    # the regex tier's findings for this file start here
        for idx, l in enumerate(fd.lines):
            if l.side != "+":
                continue
            if cls == PY_CONFIG:
                for kind, snip in _config_findings(fd, l):
                    findings.append(Finding(fd.path, l.new_lineno, tier, kind, "pytest", snip, None, False, None))
            if cls == WORKFLOW:
                for kind, snip in _workflow_findings(fd, l):
                    findings.append(Finding(fd.path, l.new_lineno, tier, kind, "any", snip, None, False, None))
            for p in DETECTION_MATRIX:
                if p.side != "+" or cls not in p.files or p.regex is None or not p.regex.search(l.text):
                    continue
                if p.kind == "config-testpaths-shrink":
                    continue          # handled by the tokenised config walker
                if p.kind == "workflow-swallowed-exit" and l.text.strip().startswith("#"):
                    continue
                if p.kind == "config-option-changed" and any(f.path == fd.path and f.lineno == l.new_lineno and f.kind.startswith("config-") for f in findings):
                    continue          # a specific narrowing kind already names this line
                is_deco = bool(_DECORATOR.match(l.text))
                target = _target_for(fd, cls, idx, is_deco, defs) if p.aimable else None
                if p.kind == "jest-only":
                    target = None
                findings.append(Finding(fd.path, l.new_lineno, tier, p.kind, p.family, l.text, target,
                                        p.aimable, p.marker, p.observation))
                if p.aimable:
                    marker_idx.add(idx)
        if cls in (TEST_PY, TEST_JS):
            loosened = _loosening_findings(fd, cls, defs)
            for f, idx in loosened:
                findings.append(f)
                marker_idx.add(idx)
            _before = files_before.get(fd.old_path or fd.path) if isinstance(files_before, Mapping) else None
            _after = files_after.get(fd.path) if isinstance(files_after, Mapping) else None
            findings.extend(_deletion_findings(fd, cls, _before if isinstance(_before, str) else None,
                                               _after if isinstance(_after, str) else None))
        if cls == TEST_PY:
            findings.extend(_diff_decollection_findings(fd))
            findings.extend(_diff_raises_widened(fd, defs))
        needed_regex = _needed_from_diff(fd, cls, marker_idx)
        # the AST / token tiers
        if cls in (TEST_PY, CONFTEST, OTHER_PY) and fd.path.lower().endswith((".py", ".pyi")):
            ast_findings, needed_ast = _python_ast_tier(fd, cls, files_after, files_before, tier, "pytest", notes,
                                                        inheritance=inheritance, known_modules=known_modules)
            if needed_ast is not None:
                # ELEVENTH CYCLE (V10-F1). The AST tier READ this file and DECIDED its hooks, autouse
                # fixtures and pytest_plugins by their bodies. The regex tier's line matches for those
                # same kinds are shape-only (`def pytest_*` on a line) and would re-accuse what the AST
                # tier just decided; they yield. When the AST tier could NOT read the file the regex
                # findings stand — an unreadable conftest is refused, not waved through.
                findings[regex_from:] = [f for f in findings[regex_from:] if f.kind not in BODY_DECIDED_KINDS]
            findings.extend(ast_findings)
            needed |= needed_ast if needed_ast is not None else needed_regex
        elif cls in (TEST_JS, OTHER_JS):
            findings.extend(_js_token_tier(fd, cls, files_after, tier, notes))
            needed |= needed_regex
        else:
            needed |= needed_regex
    # de-duplicate: same path + line + kind from two layers is one finding (keep the first, prefer one with a target)
    dedup: dict[tuple, Finding] = {}
    for f in findings:
        k = (f.path, f.lineno, f.kind, f.observation)
        if k in dedup and (dedup[k].target or not f.target):
            continue
        dedup[k] = f
    counts["notes"] = notes
    return list(dedup.values()), needed, counts


def gaming_markers_from_diff(diff_text: str, files_after: Mapping[str, str] | None = None,
                             files_before: Mapping[str, str] | None = None, skip_allowlist=None,
                             silencing_probe: Mapping[str, Any] | None = None) -> dict[str, str]:
    """{path::test or path: marker} for every aimable marker the diff adds — the `gaming_markers`
    input C1 takes. File-level markers are keyed by path. `runner-escape` outranks the others on a
    given key (it voids a witness).

    FOURTEENTH CYCLE (verifier 13, E4/E6/H8 -- the allowlist remedy must clear the receipt). This read
    the diff ALONE and knew nothing of the loud-skip allowlist or the runtime probe, so a skip that C2
    had ALLOWED (on the base tree, with a reason) or DEFERRED (the probe ran every base test in the
    file and each still executes) was still handed to C1 as a gaming marker, and C1 said GAMED-SUSPECT
    over a documented, honest skip -- `receipt.py` maps that to `failure`. It now goes through the SAME
    judgement check_c2 applies (`_judge`): a finding C2 allowed or deferred is not a marker. Given no
    allowlist and no probe (an out-of-tree caller) it behaves exactly as before, which is the strict
    direction. A `runner-escape` is never allowlisted away here: it voids a witness by construction."""
    allow: list[tuple[str, str]] = []
    if skip_allowlist is not None:
        allow, _refused = _parse_allowlist(skip_allowlist)
    probe = runtime_silencing.analyse(silencing_probe if silencing_probe is not None else {"state": "not-supplied"},
                                      diff_text, files_after, files_before)
    judged = _judge(diff_text, files_after, files_before, allow, probe)
    out: dict[str, str] = {}
    for f, disposition, _hit in judged.rows:
        if not f.aimable or not f.marker:
            continue
        if disposition != "open" and f.marker != "runner-escape":
            continue
        key = f"{f.path}::{f.target}" if f.target else f.path
        if f.marker == "runner-escape" or key not in out:
            out[key] = f.marker
    return out


def contamination_from_diff(diff_text: str, files_after: Mapping[str, str] | None = None,
                            files_before: Mapping[str, str] | None = None) -> dict[str, str]:
    """{path: reason} for every changed file that can affect outcomes: a conftest or a kept test file
    defining an outcome-affecting hook or an autouse fixture, pytest_plugins, collect_ignore,
    add_marker, a `_pytest` import, a module attribute built from a string, plugin REGISTRATION
    (`-p <mod>`, a pytest11 entry point, PYTEST_PLUGINS) — and any changed file this scan could not
    read IN FULL. Parametrization-only hooks (observations) do not contaminate. The Action hands
    this to C1 as `contaminating_files` (owner ruling v2).

    SCOPE, and it is a contract surface (fifth cycle, verifier 4, finding 6). "Any changed file"
    means any changed file whose NEW TEXT the caller supplied in `files_after`. Given a diff alone,
    the tiers can only reconstruct a MODIFIED file from its own hunks, and a hook whose `def` line is
    unchanged context — or is not in the hunk at all — is not in the input to be found. Such a file
    is therefore NAMED as unscannable rather than silently cleared, because a scan that read part of
    a file cannot answer the question this function is asked. The Action passes `files_after` for
    every changed text file (`_changed_texts`), so end to end the claim holds literally; the
    fail-closed branch exists for every other caller."""
    findings, _, counts = audit_diff(diff_text, files_after, files_before)
    out: dict[str, list[str]] = {}
    for f in findings:
        # NOT tier-B-only since the fourth cycle: plugin REGISTRATION is a tier-C config finding
        # (`-p <mod>` in addopts, a pytest11 entry point, PYTEST_PLUGINS) and contaminates just as much
        # as a conftest hook — the registered module's hooks run on the reverted tree too.
        # ELEVENTH CYCLE: membership in CONTAMINATING_KINDS decides, not the observation flag — an
        # `infra-observed` hook is an observation for C2 and a contamination for C1 (the owner's
        # 2026-09-05 invariant, untouched). Parametrization-only hooks are not in the set and never were.
        if f.kind not in pyast_audit.CONTAMINATING_KINDS:
            continue
        what = f.snippet.split(" — ", 1)[0].strip()[:80]
        out.setdefault(f.path, []).append(f"{f.kind}: {what}")
    for note in counts.get("notes", []):
        path, _, rest = note.partition(": ")
        if "regex tier only" in rest or "not reconstructable" in rest:
            what = ("changed conftest.py the AST tier could not read — treated as outcome-affecting"
                    if path.endswith("conftest.py") else
                    "changed file the AST tier could not read at all — treated as outcome-affecting")
            out.setdefault(path, []).append(what)
        elif "AST from diff (hunks)" in rest or "token tier from the diff" in rest:
            # FIFTH CYCLE (verifier 4, finding 6). This scan's published scope is "IN ANY CHANGED
            # FILE", and that can only be true of a file whose NEW TEXT was read in full. Without
            # `files_after` the tiers reconstruct a MODIFIED file from its hunks alone, so a
            # `pytest_runtest_call` or an autouse fixture whose `def` line sits outside the hunk —
            # unchanged context, or not in the hunk at all — is simply not in the input. Returning
            # {} for that file is a green from a path that compared nothing, so the file is NAMED.
            # (The Action supplies files_after for every changed text file, which is why the real
            # Action refuses V4E13/V4E14; this fires only for a caller that hands over a diff alone.)
            out.setdefault(path, []).append(
                "changed file whose new text was not supplied — scanned from the diff's hunks only, so an "
                "outcome-affecting hook or fixture defined outside the hunk cannot be ruled out")
    return {p: "; ".join(dict.fromkeys(v)) for p, v in out.items()}


# ------------------------------------------------------------------------------ allowlist

# The kinds the AST tier decides by BODY (eleventh cycle); the regex tier's shape-only matches for
# them yield whenever the AST tier read the file.
BODY_DECIDED_KINDS: frozenset[str] = frozenset({
    "conftest-hook", "conftest-autouse", "conftest-plugins", "plugin-hook", "plugin-autouse", "plugin-plugins",
    "conftest-parametrize-hook", "plugin-parametrize-hook",     # the AST's descriptive observation line wins
})

_KNOWN_KINDS = {p.kind for p in DETECTION_MATRIX} | {"allowlist-entry-refused"}
_WILD = set("*?[]!")


def _names_a_path(pattern: str) -> bool:
    """At least one `/`-separated segment of the path part is a literal (no wildcard characters)."""
    path_part = re.split(r"::|:\d+$", pattern, maxsplit=1)[0]
    segs = [s for s in path_part.split("/") if s]
    return any(seg and not (set(seg) & _WILD) for seg in segs)


def _parse_allowlist(raw: Any) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(honoured entries, refused entries with the reason they were refused)."""
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"skip_allowlist must be a list of strings, got {type(raw).__name__}")
    out: list[tuple[str, str]] = []
    refused: list[tuple[str, str]] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise TypeError(f"skip_allowlist entries must be str, got {type(entry).__name__}: {entry!r}")
        s = entry.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(None, 1)
        pat = parts[0]
        reason = parts[1].strip() if len(parts) > 1 else ""
        if not reason:
            refused.append((pat, "no reason given — a loud skip needs a reason"))
            continue
        if pat in _KNOWN_KINDS:
            refused.append((pat, "a bare finding kind allows every finding of that kind everywhere"))
            continue
        if not _names_a_path(pat):
            refused.append((pat, "names no literal path segment — it would allow every skip in the repository"))
            continue
        out.append((pat, reason))
    return out, refused


def _allowed_by(f: Finding, allow: list[tuple[str, str]]) -> tuple[str, str, str] | None:
    for pat, reason in allow:
        for key in f.keys:
            if key == pat or fnmatch.fnmatchcase(key, pat):
                return pat, reason, key
    return None


# ------------------------------------------------------------------------------- the check

@dataclass
class _Judged:
    """Every finding with its DISPOSITION -- `open` (an accusation), `allowed` (the base-tree allowlist
    covers it), `observation` (never an accusation: the probe answered for it, or it was one to begin
    with) -- plus the proof lines that go beneath them. ONE judgement, consumed by check_c2 (which
    renders it) and by gaming_markers_from_diff (which hands C1 only the `open` aimable markers), so
    the two can never disagree about whether a skip was allowed or deferred."""
    rows: list                      # [(Finding, disposition, allow-hit | None)]
    rt_proof: dict                  # (path, target) -> the PROOF / DEFERRED / NOT DECIDED line
    deferred: int
    dedup_notes: list               # "NOTE ... is ONE finding, not two" lines
    needed_derived: set
    counts: dict
    consumed: set                   # (path, target) whose account a syntactic finding deferred to


def _judge(diff_text: str, files_after, files_before, allow: list, probe: runtime_silencing.Report) -> _Judged:
    findings, needed_derived, counts = audit_diff(diff_text, files_after, files_before)
    already = {(f.path, f.target) for f in findings if not f.observation and f.target}
    rt_proof: dict[tuple[str, str | None], str] = {}
    dedup_notes: list[str] = []
    allow_note = ("the loud-skip allowlist ({n} honoured entr{ies}) does not cover it; an intentional skip belongs "
                  "there, with a reason").format(n=len(allow), ies="y" if len(allow) == 1 else "ies")
    for row in probe.rows:
        if (row.path, row.target) in already:
            dedup_notes.append(f"  NOTE {row.test_id} {_DEDUP_TEXT}; it is ONE finding, not two")
            rt_proof[(row.path, row.target)] = row.proof(allow_note)     # the proof rides under that finding
            continue
        rt_proof[(row.path, row.target)] = row.proof(allow_note)
        findings.append(Finding(row.path, None, "A", "test-silenced-at-runtime", "any", row.snippet(),
                                row.target, False, None))
    for u in probe.undecided:
        findings.append(Finding(u.path, None, "A", "test-silenced-at-runtime", "any", u.snippet(),
                                u.target, False, None, observation=True))
        rt_proof[(u.path, u.target)] = u.proof()

    # ---- THIRTEENTH CYCLE. The syntactic tier DEFERS to the runtime fact where the runtime fact
    # exists (owner's ruling 2026-09-07, on verifier 12's H3/H8/H11 -- three accusations against
    # ORDINARY code from detectors that read the diff and never ran anything).
    #
    #   H3  a feature deleted TOGETHER WITH its test        -> `test-deleted` + `assertion-deleted`
    #   H11 a test RELOCATED to another file                -> the same two
    #   H8  a fixture that skips when a service is absent   -> `skip-call` (the charter's own example)
    #
    # All three are everyday work, and all three are DECIDABLE by running the test rather than reading
    # it. Where the probe can point at the positive fact that answers the syntactic tier's guess (the
    # closed list in `Report.defer_reason`), the syntactic finding becomes an OBSERVATION -- printed,
    # never an accusation. Where it cannot, the finding stands untouched: a deferral to a measurement
    # that did not happen would be a fake green.
    # FOURTEENTH CYCLE: the fact is now PER TEST (`Report.accounted`), the skip family is three kinds
    # wide, and an UNDECIDED account defers to a NOT DECIDED observation (the decidability rule).
    deferred = 0
    consumed: set[tuple[str, str | None]] = set()
    for i, f in enumerate(findings):
        if f.observation:
            continue
        why = probe.defer_reason(f.kind, f.path, f.target)
        if why is None:
            continue
        findings[i] = replace(f, observation=True, snippet=f"{f.snippet.strip()[:100]} -- NOT an accusation")
        rt_proof[(f.path, f.target)] = f"    DEFERRED {f.path}" + (f"::{f.target}" if f.target else "") + f": {why}"
        consumed.add((f.path, f.target))
        deferred += 1
    # An UNDECIDED account with no syntactic finding to defer is still SAID, as an observation: the
    # receipt names the test Corund could not decide about, instead of a silence a reader would take
    # for an all-clear.
    for a in probe.accounted:
        if a.undecided and (a.path, a.target) not in consumed:
            findings.append(Finding(a.path, None, "A", "test-silenced-at-runtime", "any", a.snippet(), a.target,
                                    False, None, observation=True))
            rt_proof[(a.path, a.target)] = f"    NOT DECIDED {a.path}::{a.target}: {a.why}"
            consumed.add((a.path, a.target))

    rows: list = []
    for f in findings:
        if f.observation:
            rows.append((f, "observation", None))
            continue
        hit = _allowed_by(f, allow) if f.kind != "allowlist-entry-refused" else None
        rows.append((f, "allowed" if hit else "open", hit))
    return _Judged(rows, rt_proof, deferred, dedup_notes, needed_derived, counts, consumed)


_DEDUP_TEXT = "is PROVEN silenced at runtime and is already reported above by a detector that read the diff"
_XFAIL_KINDS = frozenset({"mark-xfail", "xfail-call", "jest-failing"})


def check_c2(inputs: Mapping[str, Any]) -> Verdict:
    base, head = str(inputs["base_sha"]), str(inputs["head_sha"])
    diff_text = inputs["diff"]
    if not isinstance(diff_text, str):
        raise TypeError(f"diff must be str, got {type(diff_text).__name__}")
    if not diff_text.strip():
        raise MissingInput("diff (empty — nothing to audit is not a pass)")
    allow, refused = _parse_allowlist(inputs["skip_allowlist"])
    needed_in = inputs.get("c1_needed_tests")
    allow_path = inputs.get("skip_allowlist_path")
    files_after = inputs.get("files_after")
    files_before = inputs.get("files_before")
    for name, m in (("files_after", files_after), ("files_before", files_before)):
        if m is not None and not isinstance(m, Mapping):
            raise TypeError(f"{name} must be a mapping of path -> text")

    # TWELFTH CYCLE. REQUIRED, not optional (the tenth cycle's `tracked_files` ruling, applied to the
    # other check): without it C2's success sentence claimed "silences no test" from a comparison that
    # was never made, which is what verifier 11 refuted. An OMITTED key is NOT_RUN (runner.py); a
    # present one must speak the closed vocabulary or `analyse` raises -> CRASHED. It is never green.
    # FOURTEENTH CYCLE: the diff and both texts go in too, so the rule can read WHICH FUNCTIONS the
    # non-test diff touches (the subject linkage) at function granularity.
    probe = runtime_silencing.analyse(inputs["silencing_probe"], diff_text, files_after, files_before)
    judged = _judge(diff_text, files_after, files_before, allow, probe)
    counts = judged.counts
    if needed_in is not None:
        if not isinstance(needed_in, (list, tuple, set, frozenset)):
            raise TypeError("c1_needed_tests must be a list of test ids")
        needed = {str(x) for x in needed_in}
        needed_src = f"{len(needed)} test(s) named by the caller"
    else:
        needed = judged.needed_derived
        needed_src = f"{len(needed)} test(s) derived from the diff (added or modified)"

    ev = [f"C2 skip-audit: base {base[:12]} head {head[:12]} — audited {counts['files']} file(s) "
          f"({counts['test_files']} test file(s)), {counts['added']} added / {counts['deleted']} deleted line(s); "
          f"tiers A-D applied ({len(DETECTION_MATRIX)} patterns; regex + AST + token layers); C1-needed set: {needed_src}"]
    n_ast = sum(1 for n in counts["notes"] if "AST from" in n or "token tier from" in n)
    n_regex_only = sum(1 for n in counts["notes"] if "regex tier only" in n)
    ev.append(f"  AST/token tier: {n_ast} file(s) read ({'files_after supplied' if files_after is not None else 'from the diff — pass files_after for whole-file analysis'}), "
              f"{n_regex_only} regex-only" + ("" if files_before is not None else "; files_before absent — collectability before/after not compared"))
    for n in counts["notes"]:
        if "regex tier only" in n or "audited as a test file" in n or "not reconstructable" in n:
            ev.append(f"  NOTE {n}")
    rows = list(judged.rows)
    for pat, why in refused:
        rows.append((Finding(allow_path or "skip_allowlist", None, "A", "allowlist-entry-refused", "any",
                             f"allowlist entry `{pat}` REFUSED and not honoured: {why}", None, False, None), "open", None))

    ev.append(probe.scope_line())
    # THIRTEENTH CYCLE, the owner's third bar (2026-09-07): "the standing limits of red-on-revert
    # DISCLOSED as NAMED RESIDUALS ON EVERY RECEIPT, not chased". Not in a document a reader never
    # opens, and not only in the generated matrix -- on the receipt, beside the verdict they qualify,
    # every time, whether the verdict is PROVEN or FAILED. A green whose limits are named is worth
    # more than a green that pretends to have none; unstated is not covered.
    ev.extend(probe.residual_lines())
    # THIRTEENTH CYCLE. The rule now DECLINES to accuse in a shape it used to accuse in, and a decision
    # not to accuse is evidence too -- a reader who never sees it cannot tell "looked and found nothing"
    # from "did not look". FOURTEENTH CYCLE: each NOT SILENCED line names the test that answers for it
    # and the SUBJECT they share; an account a syntactic finding already deferred to is printed there.
    for a in probe.cleared:
        if (a.path, a.target) not in judged.consumed:
            ev.append(a.line())
    ev.extend(judged.dedup_notes)
    where_allow = f"the loud-skip allowlist at {allow_path}" if allow_path else "the loud-skip allowlist supplied"
    if judged.deferred:
        ev.append(f"  NOTE {judged.deferred} syntactic finding(s) of kind {', '.join(sorted(runtime_silencing.DEFER_TO_RUNTIME))} "
                  f"DEFER to the runtime probe, which RAN the tests those findings only read. They are printed below "
                  f"as OBSERVATIONS. A deletion, a relocation or a service-absent skip that costs no test its "
                  f"detection is ordinary work, and this check does not accuse ordinary work")

    if allow_path:
        for fd in parse_unified_diff(diff_text):
            if fd.path == allow_path or fd.old_path == allow_path:
                ev.append(f"  NOTE the diff edits the skip allowlist {allow_path} (+{fd.added_count}/-{fd.deleted_count} "
                          f"line(s)) — entries added by this PR do not count until merged; the BASE allowlist was applied")

    open_findings: list[Finding] = []
    gamed: list[Finding] = []
    n_allowed = 0
    n_observed = 0
    xfail_open = False
    needed_files = {n.split("::", 1)[0] for n in needed}
    rt_proof = judged.rt_proof
    for f, disposition, hit in rows:
        if disposition == "observation":
            n_observed += 1
            ev.append("  OBSERVATION" + f.line()[1:])
            if (f.path, f.target) in rt_proof:
                ev.append(rt_proof[(f.path, f.target)])
            continue
        if disposition == "allowed":
            pat, reason, key = hit
            n_allowed += 1
            ev.append(f"  ALLOW {key} [{f.kind}] matched `{pat}` — {reason}")
            continue
        open_findings.append(f)
        ev.append(f.line())
        # The runtime rule accuses, so it states its whole proof: both runs, and the allowlist's answer
        # for this exact test. An accusation a reader cannot check is the thing this product sells against.
        if (f.path, f.target) in rt_proof:
            ev.append(rt_proof[(f.path, f.target)])
        if f.kind in _XFAIL_KINDS:
            xfail_open = True
        if f.aimable:
            if f.target and f"{f.path}::{f.target}" in needed:
                gamed.append(f)
            elif f.target is None and (f.path in needed or f.path in needed_files):
                gamed.append(f)
            elif f.target and f.path in needed:      # caller named the whole file
                gamed.append(f)
    if xfail_open:
        # FOURTEENTH CYCLE (verifier 13, E5 / D08): the decision, stated where it applies. An xfail is
        # NOT deferred to the runtime probe the way a service-absent skip is, and the reason is said
        # here rather than left to a document.
        ev.append("  NOTE an xfail is not deferred to the runtime probe: a skipif or importorskip steps aside only where "
                  "the service or dependency is absent, and the probe can see that every existing test still executes; "
                  "an xfail keeps the test running and DISCARDS its verdict in every environment, so \"still executes\" "
                  "is not the fact that would clear it. A reason written inside the decorator is written by the same "
                  "hand that wrote the diff; the loud form is the entry on the BASE tree's allowlist "
                  f"({where_allow}), which is reviewed before it takes effect. Add the entry there, with the reason")

    findings = [f for f, d, _h in rows if d != "observation"]
    if not findings and counts["text_lines"] == 0:
        raise MissingInput("diff (no text line to audit: " + (", ".join(counts["binary_or_mode_only"]) or f"{counts['renamed']} rename(s) with no hunks") + " — nothing to audit is not a pass)")

    summary = f"{len(findings)} finding(s): {len(open_findings)} open, {n_allowed} allowed" + (f", {n_observed} observation(s)" if n_observed else "")
    if gamed:
        aimed = ", ".join(f"{g.path}::{g.target}" if g.target else f"{g.path} (file-level)" for g in gamed)
        ev.append(f"GAMED-SUSPECT: {summary}; marker(s) aimed at the test(s) C1 needed: {aimed}")
        return Verdict("C2", "GAMED_SUSPECT", base, head, None, tuple(ev))
    if open_findings:
        by_tier: dict[str, int] = {}
        for f in open_findings:
            by_tier[f.tier] = by_tier.get(f.tier, 0) + 1
        tiers = ", ".join(f"tier {t}: {n}" for t, n in sorted(by_tier.items()))
        n_rt = sum(1 for f in open_findings if f.kind == "test-silenced-at-runtime")
        rt = ""
        if n_rt:
            where = f"the loud-skip allowlist at {allow_path}" if allow_path else "the loud-skip allowlist"
            rt = (f"; {n_rt} test(s) PROVEN SILENCED AT RUNTIME — their base version fails by assertion against this "
                  f"PR's code and this PR's version of them does not, and {where} "
                  f"({len(allow)} honoured entr{'y' if len(allow) == 1 else 'ies'}) covers none of them. An "
                  f"intentional skip belongs there, with a reason")
        ev.append(f"FAILED: {summary} ({tiers}) — a silent skip is a failure unless it is on the loud-skip "
                  f"allowlist with a reason{rt}")
        return Verdict("C2", "FAILED", base, head, None, tuple(ev))
    # TWELFTH CYCLE. The sentence now says only as much as was actually compared. When the runtime
    # probe DID run it affirms the measured fact; when it did not, the claim is explicitly narrowed to
    # the diff's TEXT, because a silencing with no syntactic marker was not looked for at all.
    n_undecided = len(probe.undecided) + sum(1 for a in probe.accounted if a.undecided)
    if probe.measured and n_undecided:
        # FOURTEENTH CYCLE. The affirmation "deletes or silences no assertion or test" is NOT said over a
        # test this check could not decide about: the sentence covers exactly what was decided, and the
        # NOT DECIDED tests are named above, outside it. A green whose reach is stated is worth more than
        # a green that pretends to have none.
        ev.append(f"PROVEN: {summary} — the diff adds no silent skip, xfail, dead gate, constant-true assertion, "
                  f"runner escape or discovery narrowing, and silences no test this check could DECIDE about"
                  + ("" if probe.test_only else f": of the {probe.n_detecting} test(s) whose base version detects this "
                                                f"PR's change, every one still detects it when this PR's own version of it is run")
                  + f". {n_undecided} test(s) are NOT DECIDED and named above; this sentence does not cover them")
    elif probe.measured and probe.test_only:
        ev.append(f"PROVEN: {summary} — the diff adds no silent skip, xfail, dead gate, constant-true assertion, "
                  f"runner escape or discovery narrowing, and deletes or silences no assertion or test: this PR changes "
                  f"no non-test code, and every one of the {probe.n_functions} base test(s) in the files it touches "
                  f"still runs, is relocated, or is accounted for by name above")
    elif probe.measured:
        ev.append(f"PROVEN: {summary} — the diff adds no silent skip, xfail, dead gate, constant-true assertion, "
                  f"runner escape or discovery narrowing, and deletes or silences no assertion or test: of the "
                  f"{probe.n_detecting} test(s) whose base version detects this PR's change, every one still "
                  f"detects it when this PR's own version of it is run")
    else:
        # The affirming phrase "deletes or silences no assertion or test" does NOT appear here, and its
        # absence is pinned by a case row. Leaving it in and appending a caveat would keep the exact
        # sentence verifier 11 caught lying on the receipt, where a reader — and a grep — would still
        # find C2 affirming something it had not compared.
        ev.append(f"PROVEN: {summary} — the diff's TEXT adds no silent skip, xfail, dead gate, constant-true "
                  f"assertion, runner escape or discovery narrowing, and shows no deletion or loosening of an "
                  f"assertion. The RUNTIME silencing comparison DID NOT RUN ({probe.state}), so this is a verdict "
                  f"on the diff's TEXT ALONE: a test silenced with no syntactic marker was not looked for here")
    return Verdict("C2", "PROVEN", base, head, None, tuple(ev))


# ------------------------------------------------------------------- the matrix, rendered

# The frame column is the owner's ruling of the fifth cycle: a reader must see AT A GLANCE, per
# runner family, whether a PROVEN carries a VERIFIED witness frame. jest/vitest ships honest-but-
# weaker and clearly labelled, because a labelled frame-unverified C1 for JavaScript beats no
# JavaScript coverage; pytest is never waived and keeps the full-strength own-frame guarantee.
SUPPORT_MATRIX: tuple[tuple[str, str, str, str], ...] = (
    # runner family, launch support, witness frame verified, how per-test results are read
    ("pytest family (pytest, python -m pytest, tox/nox wrappers that pass args through)", "supported",
     "YES — always. The junit body carries frames, so the own-frame allowlist is applied in full to every witness: "
     "origin = the test's OWN MODULE and entry = the test's OWN FUNCTION. A witness whose frames are missing is "
     "UNPROVEN, never waived",
     "--junitxml report; `fail` = <failure> in the call phase whose message head is `assert` / `AssertionError` / `Failed:` "
     "(exact; a class name with any other head is an error); <error> = any other phase; a `--collect-only -q` pre-pass in the "
     "same tree gives the collected ids, and exit code / report counts / collected ids / witness ids are reconciled per run"),
    ("jest / vitest family (jest --json, vitest --reporter=json)", "supported",
     "NO — WAIVED, and labelled. The JSON report carries no frame for a failed assertion, so the runner declares "
     "`frame_attribution: unavailable: <why>` and C1 takes the red at the adapter's word. A jest/vitest PROVEN is "
     "FRAME-UNVERIFIED and WEAKER than a pytest PROVEN, and every receipt that uses the waiver says so on its own "
     "line. Accepted ONLY in that exact spelling with a real reason, and REFUSED for any witness whose run did "
     "supply an origin or an entry frame — a runner that declared frames unavailable and then supplied one has "
     "contradicted itself, and the allowlist is applied to it in full",
     "JSON reporter; assertionResults status + failureMessages; a suite-level testExecError is a collection error; "
     "exit code / report reconciled per run; collected-id reconciliation NOT AVAILABLE (no test-level listing without running)"),
    ("go test, cargo test, rspec, minitest, phpunit, dotnet test, gradle/maven, mocha (without a junit reporter)",
     "NOT SUPPORTED", "n/a — nothing is proven, so there is no witness to verify",
     "C1 and C2's runner half report NOT_RUN naming the runner; C2's diff audit still runs"),
)

ISOLATION_MATRIX: tuple[tuple[str, str], ...] = (
    ("execution tree", "a FRESH `git worktree` per phase (with-change, without-change, rerun) with its `.git` link REMOVED before "
                       "anything runs — no `git status` / `git log` / `rev-parse` oracle exists in any phase; the tree is deleted after"),
    ("the reverted tree", "head checkout + `git apply` of the reverse patch of the NON-TEST, NON-INFRASTRUCTURE files, then `.git` removed; "
                          "the rerun is a third fresh tree built the same way"),
    ("test infrastructure", "conftest.py, pytest.ini, pyproject.toml, setup.cfg, tox.ini, jest/vitest/vite config, package.json, "
                            ".github/workflows/**, `__init__.py` under test dirs are NEVER reverted (revert-protected surface); "
                            "their changes are C2 tier B-D / C3 findings"),
    ("temp dir", "TMPDIR/TEMP/TMP point at a fresh directory per run, deleted after; the report is written OUTSIDE the tree and "
                 "outside that TMPDIR at an unpredictable path"),
    ("environment", "PYTEST_ADDOPTS and PYTEST_PLUGINS are DROPPED from every phase, not merely reported: either can register a "
                    "plugin or inject flags into every run, and C1 never reverts the environment, so leaving them in place would "
                    "put code C1 cannot revert into the comparison. If one was set, the receipt names it"),
    ("test selection", "the changed test files C1 runs are narrowed by the REPOSITORY'S OWN pytest configuration: `testpaths`, "
                       "read at head from `pytest.ini` / `.pytest.ini` / `pyproject.toml` (`[tool.pytest.ini_options]` or "
                       "`[tool.pytest]`) / `tox.ini` `[pytest]` / `setup.cfg` `[tool:pytest]`, in pytest's own documented "
                       "precedence (the first file that MATCHES wins; candidates are never merged). A changed test file outside "
                       "the declaration is NOT selected and is NAMED on the receipt with the reason. Three bounds: a repo that "
                       "declares nothing selects exactly what it selected before; a caller who passed their own `test-globs` "
                       "overrules the declaration, and the receipt says so; the declaration is PYTEST'S, so it never narrows a "
                       "jest/vitest run. Selection narrows what RUNS — it never swallows a crash in what does"),
    ("reconciliation", "per run: the runner's exit code must agree with the report (pytest 0 = no failures and >= 1 test; 1 = >= 1 "
                       "failure/error; 5 = nothing collected; 2/3/4 = did not complete), the <testsuite> counts must equal the "
                       "counted <failure>/<error>/<skipped> ELEMENTS and, for `tests`, the counted <testcase>s, and the set of "
                       "reported ids must equal the set collected by the pre-pass; every witness must be collected in BOTH trees. "
                       "Any disagreement -> C1 CRASHED naming it; never PROVEN. ONE EXCEPTION, stated on the receipt whenever it "
                       "applies: pytest counts every OUTCOME in `<testsuite tests=N>` but writes one <testcase> per COLLECTED TEST, "
                       "so a stdlib `unittest.subTest()` run counts more than it itemises. A `tests` SURPLUS is an OBSERVATION "
                       "rather than a disagreement when — and only when — every <testcase> reconciles one-for-one with the ids the "
                       "collect-only pre-pass listed on that same tree. A DEFICIT, a surplus with no id comparison available, and a "
                       "surplus over an id comparison that DISAGREES all stay CRASHED; the other three attributes are compared in "
                       "both directions unchanged, so a <failure> deleted to fake a pass is still caught"),
    ("reverted diff composition", "named on the receipt (files, +/- lines, mode-only / binary / symlink); a reverse patch with NO text "
                                  "line is NOT_RUN 'nothing executable to revert'"),
    ("witness attribution (ALLOWLIST)", "two frames are read from every red's junit body (the Action appends --tb=auto on the command "
                                        "line, which outranks the repo's addopts, so both are present): the LAST frame `path:line` is the "
                                        "ORIGIN and the FIRST `def <name>(` line is the ENTRY. A witness is trusted ONLY when the origin's "
                                        "file is the test's OWN MODULE (exact after normalisation — no suffix matching) AND the entry is the "
                                        "test's OWN FUNCTION. Every other frame is UNPROVEN-contaminated: a plugin or a hook wherever it lives "
                                        "and whenever it was registered (including a pytest_runtest_call defined in the kept test file itself, "
                                        "whose origin IS the test's module), an autouse fixture, a helper module, the code under test, and a run "
                                        "that reported no frame at all. STATED COST: an HONEST test is refused as well when its assertion fires "
                                        "inside a shared helper MODULE, inside an `assert` in the code under test, or inside a TEST METHOD "
                                        "INHERITED FROM A BASE CLASS IN ANOTHER FILE — the mainstream `class TestAdd(AddContract)` contract-test "
                                        "layout, where the assertion's origin is the BASE CLASS's module and never the test's own. Assert in the "
                                        "test body to be provable. An adapter that carries no frames at all (jest/vitest JSON) must DECLARE it "
                                        "(frame_attribution 'unavailable: <why>'), that declaration is printed on the receipt, and the verdict is "
                                        "labelled FRAME-UNVERIFIED — see the frame column of the support matrix above"),
    ("contamination (owner ruling v2, widened in the fourth cycle)", "if the PR adds or modifies test infrastructure that can affect "
                                        "outcomes that code is never reverted and RUNS during the reverted phase: every witness is "
                                        "REFUSED — UNPROVEN-contaminated naming the file. A safety refusal, not a finding of intent. The scan "
                                        "covers a conftest hook other than the parametrization/reporting hooks, an autouse fixture, "
                                        "pytest_plugins, collect_ignore, add_marker and a `_pytest` import IN ANY CHANGED FILE (not only "
                                        "conftest.py — a pytest_* hook or an autouse fixture in a kept TEST file is registered the moment the "
                                        "file is loaded as a plugin), plus plugin REGISTRATION itself: `-p <mod>` in addopts (pytest.ini, "
                                        "pyproject.toml, setup.cfg, tox.ini), a `pytest_plugins` assignment anywhere, the PYTEST_PLUGINS "
                                        "environment variable, and a `pytest11` entry point declared in the packaging metadata"),
    ("guard disarmed", "if C2 crashed or its gaming markers could not be computed, C1 cannot certify a witness: CRASHED naming the "
                       "disarmed guard, never PROVEN"),
    ("not closed", "state persisted outside the tree and TMPDIR (HOME, XDG dirs, the original checkout path, the network) can still "
                   "carry between phases — NAMED on every receipt's isolation line; a token-free self-consistent forgery; an "
                   "outcome-rewriting hook in a conftest that predates the PR — stated in NOT COVERED; a sandbox is the next layer"),
)


def render_matrix() -> str:
    """The detection + support + isolation matrix as Markdown. checks/README.md must contain this block
    verbatim (checks/tests/test_readme_matrix.py), so the published coverage is derived from the code."""
    out = ["<!-- BEGIN GENERATED MATRIX (python3 -m corund_checks --matrix) -->",
           "",
           "### Runner support matrix (launch)", "",
           "| runner family | launch | witness frame VERIFIED? | how results are read |", "|---|---|---|---|"]
    for fam, sup, frames, how in SUPPORT_MATRIX:
        out.append(f"| {fam} | {sup} | {frames} | {how} |")
    out += ["", "### C1 execution isolation and reconciliation", "", "| layer | what the Action does |", "|---|---|"]
    for layer, what in ISOLATION_MATRIX:
        out.append(f"| {layer} | {what} |")
    out += ["", "### C2 detection tiers (every pattern, with its coverage and detection layer)", "",
            "| tier | kind | family | applies to | aimable | via | coverage | what it catches |", "|---|---|---|---|---|---|---|---|"]
    tier_names = {"A": "A in-test-file", "B": "B conftest relocation", "C": "C runner-config relocation",
                  "D": "D CI-workflow relocation"}
    for p in DETECTION_MATRIX:
        files = ", ".join(sorted(p.files))
        aim = "yes" if p.aimable else "no"
        out.append(f"| {tier_names[p.tier]} | `{p.kind}` | {p.family} | {files} | {aim} | {p.via} | {p.coverage} | {p.description} |")
    out += ["", "### Not covered (unstated = not covered; these are stated)", ""]
    for n in NOT_COVERED:
        out.append(f"- {n}")
    out += ["", "<!-- END GENERATED MATRIX -->"]
    return "\n".join(out) + "\n"
