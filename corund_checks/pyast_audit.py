"""C2's AST tier for Python — the honest answer to "is a regex-based skip audit ever complete?"
(it is not: aliases, parenthesised decorators, line continuations, `getattr`, string-built
attribute names and structural de-collection all pass a line regex). This module reads the NEW
side of a changed Python file with `ast`, resolves names to canonical dotted names (module scope
plus the enclosing function's locals; `import ... as`, `from ... import`, star imports from
sensitive modules, `getattr` with constant names, `sys.modules[...]`, `__dict__[...]`), folds
constants, and reports silencing shapes with an exact target (the enclosing test function).

Applies to test files, helper modules under test directories, and conftest.py (tier B kinds).
Findings are restricted to nodes that touch an ADDED line (so a pre-existing skip in a modified
file is not re-reported on every PR); the before/after collectability comparison is whole-file by
nature. Every walk that recurses (`ast.dump`, constant folding) is guarded: a pathological
expression degrades to "unknown", never to a crash — a crash here would disarm C1's markers.
What this tier cannot see is stated in `NOT_COVERED_AST` and rendered into the README's matrix;
unstated = not covered. stdlib only. NEW module.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Any

from . import infra_body
from .c1_red_on_revert import X7_RESIDUAL

# kind -> (aimable, marker C1 hears)
KIND_META: dict[str, tuple[bool, str | None]] = {
    "skip-call": (True, "skip"), "importorskip": (True, "skip"), "xfail-call": (True, "xfail"),
    "mark-skip": (True, "skip"), "mark-skipif": (True, "skip"), "mark-xfail": (True, "xfail"),
    "pytestmark": (True, "skip"), "unittest-skip": (True, "skip"), "empty-parametrize": (True, "skip"),
    "dead-gate": (True, "constant-true"), "constant-true": (True, "constant-true"), "early-return": (True, "constant-true"),
    "runner-escape": (True, "runner-escape"), "dynamic-code": (True, "runner-escape"),
    "assertion-swallowed": (True, "loosened"), "assertion-loosened": (True, "loosened"), "raises-widened": (True, "loosened"),
    "assertion-moved-under-condition": (True, "loosened"),
    "test-decollected": (True, "skip"), "test-not-collectable": (True, "skip"), "test-under-condition": (True, "skip"),
    "test-collection-unresolved": (False, None),
    "test-collection-resolved": (False, None),       # OBSERVATION: the resolution, stated (V10-F2)
    "conftest-hook": (False, None), "conftest-autouse": (False, None), "conftest-plugins": (False, None),
    "plugin-hook": (False, None), "plugin-autouse": (False, None), "plugin-plugins": (False, None),
    "plugin-parametrize-hook": (False, None),
    "conftest-dynamic-attr": (False, None), "conftest-add-marker": (False, None), "conftest-collect-ignore": (False, None),
    "conftest-mark-skip": (False, None), "conftest-internal-import": (False, None),
    "conftest-parametrize-hook": (False, None),      # OBSERVATION only: parametrization / reporting hooks
    # ELEVENTH CYCLE (V10-F1, the decidability rule): a hook / autouse fixture / pytest_plugins whose
    # body C2 DECIDED harmless or COULD NOT DECIDE — an OBSERVATION that says which, never a finding.
    # Still a C1 contamination (it is in CONTAMINATING_KINDS): the infrastructure runs on the reverted tree.
    "infra-observed": (False, None),
}

# Hooks that cannot change an outcome or what is collected: parametrization and reporting only.
# Everything else a conftest defines is outcome-affecting (pytest_configure can register plugins;
# every pytest_runtest_* / collection hook can change what runs or what is reported).
PARAMETRIZATION_ONLY_HOOKS: frozenset[str] = frozenset({
    "pytest_generate_tests", "pytest_addoption", "pytest_report_header", "pytest_terminal_summary",
    "pytest_make_parametrize_id", "pytest_addhooks", "pytest_plugin_registered", "pytest_report_collectionfinish",
    "pytest_html_results_table_header", "pytest_html_results_table_row", "pytest_html_report_title",
})
CONTAMINATING_KINDS: frozenset[str] = frozenset({
    "conftest-hook", "conftest-autouse", "conftest-plugins", "conftest-dynamic-attr", "conftest-add-marker",
    "conftest-collect-ignore", "conftest-mark-skip", "conftest-internal-import",
    # FOURTH CYCLE (owner's ruling 2026-09-05): outcome-affecting infrastructure is not conftest-shaped.
    # A hook, an autouse fixture or a plugin registration ANYWHERE the runner loads makes PR-authored code
    # run during the reverted phase, so every witness of that run is refused for safety.
    "plugin-hook", "plugin-autouse", "plugin-plugins",
    "config-register-plugin", "config-plugins-env", "config-entrypoint-plugin",
    # ELEVENTH CYCLE: an observed (benign or undecided) hook/fixture/pytest_plugins still runs during
    # the reverted phase. Observation for C2; contamination for C1 — the owner's invariant, untouched.
    "infra-observed",
})

# ------------------------------------------------------------------- constant-expression folding
#
# FIFTH CYCLE (verifier 4, the gate). Until now this folder ENUMERATED SHAPES: every adversarial
# cycle found a node type it walked straight past -- `(x := True)` and `f'{1}'` in cycle four,
# `'%d' % 1`, `f'{1:d}'`, `1 < 2 < 3` and `[*[1], 2]` in cycle five -- and every cycle answered with
# one more `isinstance` branch. Shape enumeration cannot converge, and README's published claim
# ("any constant-foldable truthy expression", one stated residual) was optimistic rather than TRUE.
#
# It is a GRAMMAR now: `ast.literal_eval`'s constant-expression language, widened to the operators,
# comparison chains, f-strings (specs and conversions included), subscripts and a short list of pure
# builtins, evaluated under an explicit size budget. Every expression node type the grammar does not
# evaluate is REFUSED BY NAME in `REFUSED_NODES`, and those names and reasons are rendered straight
# into `NOT_COVERED_AST` -- so the README's "Not covered" block is GENERATED from the refusals
# instead of written from memory, and the matrix row can state what is true instead of what is hoped.
#
# The partition is checked against `ast` ITSELF by checks/tests/test_const_fold_grammar.py, which
# enumerates the ast module's own `expr` subclasses -- scope derived from the interpreter, never a
# hand-typed list ("scope is derived, never searched"). A Python release that adds an
# expression node fails that test rather than silently widening this tier's blind spot.
#
# BUDGETED on purpose. `assert 2 ** 999999999 > 0` and `assert len('a' * 10 ** 9) > 0` are one short
# source line each; a fold with no budget would evaluate them and hang the check runner. Every
# operation whose RESULT can be far larger than its operands is bounded BEFORE the value is built.

FOLDED_NODES: frozenset[str] = frozenset({
    "Attribute", "BinOp", "BoolOp", "Call", "Compare", "Constant", "Dict", "FormattedValue",
    "IfExp", "JoinedStr", "Lambda", "List", "Name", "NamedExpr", "Set", "Slice", "Starred",
    "Subscript", "Tuple", "UnaryOp",
})
# SIXTH CYCLE (verifier 5, F3). FOLDED_NODES and REFUSED_NODES partition ast's expression universe
# BY NAME, and `test_const_fold_grammar.py` proved the partition — but never that a node listed as
# FOLDED actually folds anything. `Attribute` sat in FOLDED_NODES and returned unknown for every
# instance the parser can produce except one. Published as "REFUSES these node types BY NAME", that
# invites the complement inference — everything else folds — and the complement was false: a NAMING
# partition was being read as a BEHAVIOURAL one. These are the node types the folder handles only
# under a stated CONDITION, published beside the refusals so the complement inference is no longer
# available, and each one's narrowness is proven behaviourally rather than asserted here.
# ------------------------------------------------------------------- the census's own residual
#
# EIGHTH CYCLE (verifier 7, F-DISCLOSURE). `NARROW_NODES["Name"]` published FIVE reasons a name does
# not fold — a parameter, a loop variable, a name assigned twice, a global/nonlocal, and a name the
# file never binds. The seventh cycle added six more and the eighth added aliasing, and NONE of them
# reached the published text: `print(ok); assert ok` and `if True: ok = True; assert ok` are real
# tautologies that pass C2 unannounced, and the reader was told the tier was narrower than it is.
# That is the same drift `_REFUSAL_LINES()` was built to end for the node grammar, one layer down.
#
# So the categories are DATA, and the published sentence is GENERATED from them. Every `unsafe`
# marking in `_bound_names` goes through `mark()`, which REFUSES a category not declared here — a
# new refusal cannot reach a verdict before it reaches this table, which is the only ordering that
# keeps the two from drifting again. `checks/tests/test_const_fold_grammar.py` closes the loop from
# the other side: for every key below it runs a source that must reach that category, so a sentence
# published for a refusal the code no longer makes fails the suite too.
_UNSAFE_CATEGORIES: dict[str, str] = {
    "parameter": "a PARAMETER of the enclosing function — its value comes from the caller",
    "loop-target": "a LOOP or comprehension TARGET — rebound once per iteration",
    "destructured": "a name bound by DESTRUCTURING (`a, b = ...`, `*rest = ...`) — the fold does not "
                    "evaluate the right-hand side element-wise",
    "augmented": "a name changed by an AUGMENTED assignment (`n += 1`)",
    "except-as": "an `except ... as` name — bound by the runtime and deleted at the end of the handler",
    "deleted": "a name reached by `del`",
    "global-nonlocal": "a name declared `global` or `nonlocal`, here or by a nested scope — it is bound "
                       "somewhere this scope cannot see",
    "nested-scope-name": "the NAME of a nested `def`, `async def`, `class` or the target of one",
    "match-capture": "a `match` CAPTURE PATTERN (`case ok:`), which binds the name to the subject rather "
                     "than comparing against it",
    "conditional": "a name bound only under a CONDITION — inside `if`, `try`, a loop or a `match` case. "
                   "The condition is not evaluated, so `if True: ok = True` is refused with the rest",
    "method-call": "a name a METHOD IS CALLED ON (`errors.append(x)`, `d.update(...)`) — the method may "
                   "mutate the object without rebinding anything",
    "handed-to-call": "a name HANDED TO A CALL as an argument, positionally, by keyword, or unpacked "
                      "(`fill(d)`, `f(*d)`) — what the callee does to it is outside this file. This is a "
                      "refusal on SHAPE, so `print(ok)` and `fill('')` are refused too, even though "
                      "neither can change what the name is worth",
    "store-through": "a name a SUBSCRIPT or ATTRIBUTE store is rooted at (`d['k'] = v`, `xs[:] = ...`, "
                     "`o.a = v`), or an augmented assignment or `del` through one",
    "aliased-source": "a name that appears in the VALUE of a binding (`alias = errors`, "
                      "`box = {'e': errors}`, `sink, spare = errors, []`, or the thing a `for` iterates) "
                      "— the binding hands its object a SECOND HANDLE, and mutation through that handle "
                      "rebinds nothing here",
    "aliased-target": "a name BOUND FROM such a value — it may BE the object another name still reaches",
    "handed-out": "a name `return`ed or `yield`ed, here or by a nested scope — the object leaves for a "
                  "caller this file cannot see",
}
# Decided by `_Resolver.unfoldable_name` from the COUNTS rather than by a marking, so they are
# published beside the markings but are not reachable through `mark()`.
_COUNT_RESIDUALS: dict[str, str] = {
    "bound-twice": "a name this scope BINDS MORE THAN ONCE — the fold has no way to say which binding "
                   "reaches the assertion",
    "never-bound": "a name this file NEVER BINDS (an import, a fixture, a module-level constant defined "
                   "elsewhere) — the census has no opinion and the fold does not invent one",
}


# NINTH CYCLE (verifier 8, F-SCOPE). The eighth cycle published the census's refusals and the reader
# was still told less than the truth, because the LIST was the promise: everything not on it folds.
# Three cycles proved that list can never be complete. So the published sentence now leads with the
# RULE — the allowlist of positions a foldable name may appear in — and keeps the by-name refusals
# after it, because each of those is a shape a reader may recognise in their own file. What the
# inverted default REFUSES is therefore disclosed positively (here is the only thing that folds)
# rather than by enumeration (here are the things that do not).
_CONFINEMENT_RULE: str = (
    "every OTHER appearance of that name ANYWHERE IN THE FILE is a CLASSIFIED READ: the operand of an "
    "`assert`, `if` or `while` test, reached only through `not` / `and` / `or`, a comparison, an arithmetic "
    "operator, a conditional expression, a subscript read or a slice"
)
_CONFINEMENT_REFUSALS: str = (
    "ANY other appearance at all makes the name unknown, with no attempt to decide whether that use was "
    "harmless: handed to a call (`print(ok)`, `fill(buf)`, `f(*d)`), returned or yielded, stored into a "
    "container display, an attribute or a subscript, READ through an attribute (`errors.append`), used as a "
    "parameter DEFAULT, bound or read inside a nested `def`, `lambda`, comprehension, `class` body or "
    "`with`/`for`/`except` target in any way that is not itself a classified read, rebound, deleted, declared "
    "`global`/`nonlocal`, imported, bound by a `match` pattern, or read in any position this rule does not "
    "name. A read the analysis cannot CLASSIFY is refused for being unclassified"
)


def _NAME_RESIDUAL_LINES() -> tuple[str, ...]:
    """The census's refusals, rendered for `NARROW_NODES["Name"]` and the README's "Not covered" block.

    GENERATED from `_CONFINEMENT_RULE`, `_UNSAFE_CATEGORIES` and `_COUNT_RESIDUALS`, never written
    beside them."""
    reasons = "; ".join(v for _, v in sorted(_COUNT_RESIDUALS.items()) or ())
    marks = "; ".join(v for _, v in sorted(_UNSAFE_CATEGORIES.items()))
    return (
        f"a bare name folds ONLY as `TYPE_CHECKING`, or when the WHOLE FILE can be shown never to let its "
        f"object escape the assertion — the same scope binds it EXACTLY ONCE, UNCONDITIONALLY, to a constant "
        f"expression, and {_CONFINEMENT_RULE} (`ok = True` then `assert ok`). {_CONFINEMENT_REFUSALS}. So an "
        f"assertion that depends on any other name is never reported as constant-true, and the cost is missed "
        f"tautologies rather than accused authors. The refusals the census ALSO makes by name, each a shape a "
        f"reader may recognise: {marks}; {reasons}",
    )


NARROW_NODES: dict[str, str] = {
    "Attribute": "an attribute access — ONLY `typing.TYPE_CHECKING` folds (to False), and — since the eleventh cycle "
                 "— `self.X` inside a method of a class pytest collects BY NAME (a `Test*` class or a unittest.TestCase "
                 "subclass, not disabled) where `X` is bound EXACTLY ONCE at class-body level to a constant expression and "
                 "NOTHING can rebind it: no store to `.X` anywhere in the file or in any other file of the PR's input, no "
                 "`setattr` / `vars` / `__dict__` / `__setattr__` / `request.cls` / `request.instance` anywhere in the file, "
                 "no same-file subclass binding `X`, no binding under a condition. `(1j * 1j).real` and every other "
                 "attribute of every other value is unknown, because the fold reads no object's members",
    # EIGHTH CYCLE (verifier 7, F-DISCLOSURE): GENERATED from the census's own categories, never
    # written here. The hand-written sentence this replaces named five refusals; the code was making
    # thirteen, and the eight it did not mention included `print(ok); assert ok`.
    "Name": _NAME_RESIDUAL_LINES()[0],
    "Call": "a call — ONLY the pure builtins `bool`, `len`, `str`, `repr`, `any`, `all` and `isinstance(x, object)`, "
            "whose result is decided entirely by the constants handed to them; every other call, method calls "
            "included, is unknown",
    "Starred": "a `*x` unpacking — ONLY inside a list/tuple/set display whose starred value itself folds to a "
               "sequence; alone it is not a value",
    "Slice": "a slice — ONLY when every one of start/stop/step folds to an int or is absent",
}
REFUSED_NODES: dict[str, str] = {
    "Await": "an awaited value — the fold never runs a coroutine",
    "DictComp": "a dict comprehension — literal_eval-style folding runs no loops",
    "GeneratorExp": "a generator expression — literal_eval-style folding runs no loops",
    "ListComp": "a list comprehension — literal_eval-style folding runs no loops",
    "SetComp": "a set comprehension — literal_eval-style folding runs no loops",
    "Yield": "a yield expression — the fold never drives a generator",
    "YieldFrom": "a `yield from` expression — the fold never drives a generator",
}

_FOLD_MAX_LEN = 100_000        # longest str/bytes/sequence/mapping the fold will BUILD
_FOLD_MAX_BITS = 8_192         # widest integer the fold will BUILD
_FOLD_MAX_NODES = 20_000       # expression nodes visited in one fold


def _REFUSAL_LINES() -> tuple[str, ...]:
    """The constant folder's own refusals, rendered for the README's "Not covered" block.

    GENERATED from REFUSED_NODES and the fold's budget rather than written beside them, so the
    published residual list cannot drift from the code that produces it — the drift that made
    `constant-true`'s matrix row optimistic for four adversarial cycles. checks/README.md is checked
    against this text verbatim by checks/tests/test_readme_matrix.py."""
    names = "; ".join(f"`{k}` ({v})" for k, v in sorted(REFUSED_NODES.items()))
    narrow = "; ".join(f"`{k}` ({v})" for k, v in sorted(NARROW_NODES.items()))
    return (
        f"constant folding REFUSES these Python expression node types BY NAME, so an assertion whose value depends on "
        f"one is never reported as constant-true: {names}",
        f"constant folding HANDLES these node types only under the condition stated, so naming them as covered would "
        f"overclaim — an assertion whose value depends on one outside its condition is never reported as "
        f"constant-true: {narrow}",
        f"constant folding is BUDGETED, and the budget is applied as an UPPER BOUND computed BEFORE the value is "
        f"built, never to the finished value — a fold that had to build the value first to find out it was too big "
        f"would already have hung the check runner (`assert 2 ** 999999999 > 0` is one short source line). So the "
        f"refusal boundary is the BOUND, not the result: an expression is refused when its bound exceeds "
        f"{_FOLD_MAX_LEN} items or characters for a sequence, str, bytes or mapping, {_FOLD_MAX_BITS} bits for an "
        f"integer, or {_FOLD_MAX_LEN} for a format field, or when it visits more than {_FOLD_MAX_NODES} expression "
        f"nodes. For `a ** b` that bound is `bit_length(a) * b` and for `a << b` it is `bit_length(a) + b`, so "
        f"`2 ** 8000` is REFUSED — its bound is 16000 — even though the integer it would have built is 8001 bits "
        f"wide and therefore inside the {_FOLD_MAX_BITS}-bit budget. The bound is never below the true width, so "
        f"nothing over budget is ever built; some expressions under budget are refused as well",
    )


NOT_COVERED_AST: tuple[str, ...] = (
    "code assembled at runtime from non-constant strings (exec/eval of a computed string, attribute names built "
    "from function results): `exec`, a computed `eval`/`compile`, or a dynamic attribute of a sensitive module (os, sys, "
    "pytest, subprocess, signal, builtins, importlib, ctypes) is reported as `dynamic-code`; the assembled call is not resolved",
    "a skip or exit reached through a helper defined in a file OUTSIDE the diff (the helper's body is not in the input)",
    "a test silenced by a fixture or an outcome-rewriting hook in a conftest.py that PREDATES the PR (not in the diff): a hook "
    "that RAISES is caught by frame attribution (its origin is conftest.py); a hook that REWRITES the report text is not — the "
    "contamination rule covers PR-changed infrastructure only",
    "environment-variable-driven skips whose condition lives outside the diff",
    "a genuine assertion that fails on the reverted tree for a reason unrelated to the claimed fix (wrong-bug-red, disclosed)",
    # FOURTEENTH CYCLE, addendum -- the owner's ruling of 2026-09-08 (verifier 13's X7). The sentence is the one
    # constant C1 prints beside every PROVEN; this entry is the README's eighth residual line, GENERATED from it.
    "the theoretical floor of red-on-revert (the owner's ruling, 2026-09-08; verifier 13's X7), printed verbatim beside "
    "every C1 PROVEN and labelled `(escape side)`: " + X7_RESIDUAL + " X7 is the honest twin of the X5 escape -- "
    "byte-identical diff and runtime record, only `mul`'s return value differs -- and no deterministic check has an oracle "
    "for that value. It is the same fundamental limit as a PR that changes behaviour and updates its own test to match, "
    "already accepted and disclosed; DISCLOSED, not chased. The sentence is held once (`c1_red_on_revert.X7_RESIDUAL`), "
    "`checks/tests/test_headline_fix_guard.py` pins it on every C1 PROVEN the core case set produces, and `attack_core` / "
    "`attack_e2e` pin it on the receipt",
    "INFRASTRUCTURE IS DECIDED, OBSERVED OR PROVEN — never accused on shape alone (eleventh cycle, verifier 10). A "
    "`pytest_*` hook, an autouse fixture or a `pytest_plugins` assignment a PR adds is a C2 FINDING only when its BODY "
    "provably silences, drops, deselects or decides the outcome of a test (a skip raised or applied, a hook's item list "
    "emptied or filtered, a value returned from a hook whose return replaces pytest's default, a report's outcome assigned, "
    "an `addinivalue_line` that changes what is collected, a plugin registered through `config.pluginmanager`, a `raise` "
    "inside an outcome hook, the code under test patched for every test). A body in the closed harmless vocabulary "
    "(marker registration, printing/logging, sorting the items, seeding `random`, resetting an object, pass/yield) is "
    "DECIDED and is not a finding; a body outside both — a call into another file, file I/O, a computed attribute, a patch "
    "of a third-party name, `pytest_plugins` itself — is UNDECIDED and is an OBSERVATION that says so. In every case the "
    "file is still a C1 CONTAMINATION (PR-authored infrastructure runs during the reverted phase) and the receipt names it; "
    "the loud-skip allowlist remains available as a suppression, never as the remedy for an honest shape",
    "KNOWN OVER-FLAG: a `return` before the first assertion of a test that is a deliberate guard (`if not HAS_LIB: return`) is "
    "reported as early-return — it is a silent skip in effect; use pytest.skip with a reason",
    "an early `return` in a test whose `def` line spans several lines, when a pathological expression has ALSO disarmed the AST "
    "tier: the regex second tier reads the line after the `def` and a multi-line signature moves the body away from it",
    "a class attribute rebound by a SUBCLASS IN A FILE THE PR DID NOT TOUCH: `class TestAdd: expected = 3` with "
    "`def test_add(self): assert self.expected == 3` is reported as constant-true for `TestAdd::test_add` — which it IS, "
    "as collected — even if an unseen `class TestOther(TestAdd): expected = 4` elsewhere makes `TestOther::test_add` a "
    "real assertion. A subclass or a store to `.expected` in any file of the PR's input refuses the fold; one outside "
    "the input cannot be seen (eleventh cycle, verifier 10 V10-F7)",
    "a constant-true assertion whose value comes from a CALL the folder cannot evaluate (`assert (lambda: False)()`, "
    "`assert f(1) == f(1)` where f is impure): the fold deliberately stops at a call rather than guess a result. The "
    "exceptions, whose result is decided entirely by the constants handed to them, are `bool`, `len`, `str`, `repr`, "
    "`any`, `all` and `isinstance(x, object)`",
) + _REFUSAL_LINES()

_SKIP_CALLS = {"pytest.skip": "skip-call", "pytest.importorskip": "importorskip", "pytest.xfail": "xfail-call",
               "_pytest.outcomes.skip": "skip-call", "_pytest.outcomes.xfail": "xfail-call",
               "_pytest.outcomes.importorskip": "importorskip", "pytest.skip.Exception": "skip-call",
               "_pytest.outcomes.Skipped": "skip-call", "_pytest.outcomes.XFailed": "xfail-call",
               "unittest.SkipTest": "unittest-skip", "unittest.case.SkipTest": "unittest-skip"}
_SKIP_METHODS = {"skipTest"}
_MARK_DECORATORS = (("pytest.mark.skipif", "mark-skipif"), ("pytest.mark.skip", "mark-skip"), ("pytest.mark.xfail", "mark-xfail"),
                    ("_pytest.mark.MARK_GEN.skipif", "mark-skipif"), ("_pytest.mark.MARK_GEN.skip", "mark-skip"),
                    ("_pytest.mark.MARK_GEN.xfail", "mark-xfail"))
_UNITTEST_DECORATORS = ("unittest.skip", "unittest.skipIf", "unittest.skipUnless", "unittest.case.skip", "unittest.case.skipIf",
                        "unittest.case.skipUnless", "unittest.expectedFailure", "unittest.case.expectedFailure")
_RUNNER_ESCAPE_CALLS = {"os._exit", "sys.exit", "pytest.exit", "os.kill", "os.killpg", "os.abort", "signal.raise_signal",
                        "signal.alarm", "atexit.register", "pytest.main", "threading._shutdown", "_thread.interrupt_main",
                        "_pytest.outcomes.exit", "posix._exit", "nt._exit", "posix.kill", "posix.abort", "builtins.exit",
                        "builtins.quit", "exit", "quit"}
_RUNNER_ESCAPE_RAISES = {"SystemExit", "KeyboardInterrupt", "GeneratorExit", "pytest.exit.Exception", "_pytest.outcomes.Exit",
                         "builtins.SystemExit", "builtins.KeyboardInterrupt"}
_RUNNER_ESCAPE_TOKENS = ("junitxml", "xmlpath", "outputFile", "LogXML", "PYTEST_ADDOPTS", "--junit-xml", "resultlog", "--junitxml")
_RUNNER_ESCAPE_MODULES = ("ctypes",)
_SENSITIVE_MODULES = ("os", "sys", "pytest", "subprocess", "signal", "builtins", "importlib", "unittest", "ctypes", "_pytest",
                      "<module>", "atexit", "threading", "_thread", "posix", "nt", "faulthandler", "gc")
_STRING_BUILDERS = {"join", "format", "decode", "replace", "strip", "lower", "upper", "translate", "encode", "swapcase",
                    "casefold", "title", "capitalize", "lstrip", "rstrip", "partition", "zfill"}
_SWALLOW_TYPES = {"AssertionError", "Exception", "BaseException", "builtins.AssertionError", "builtins.Exception",
                  "builtins.BaseException"}
_RERAISE_CALLS = {"pytest.fail", "_pytest.outcomes.fail", "fail", "pytest.xfail", "unittest.TestCase.fail"}
_WEAK_UNITTEST = {"assertTrue", "assertIn", "assertIsNotNone", "assertGreater", "assertGreaterEqual", "assertLess",
                  "assertLessEqual", "assertIsInstance", "assertNotEqual", "assertIsNot", "assertNotIn", "assertFalse",
                  "assertRegex", "assertAlmostEqual"}
_STRONG_UNITTEST = {"assertEqual", "assertEquals", "assertIs", "assertListEqual", "assertDictEqual", "assertSetEqual",
                    "assertTupleEqual", "assertSequenceEqual", "assertMultiLineEqual", "assertCountEqual"}
_STRONG_MOCK = {"assert_called_once_with", "assert_called_with", "assert_has_calls", "assert_awaited_once_with",
                "assert_awaited_with"}
_WEAK_MOCK = {"assert_called", "assert_called_once", "assert_any_call", "assert_awaited", "assert_awaited_once"}
_BROAD_EXC = {"Exception", "BaseException", "builtins.Exception", "builtins.BaseException"}
_ASSERTION_CALL_CANONS = {"pytest.raises", "pytest.fail", "pytest.warns", "pytest.deprecated_call", "_pytest.outcomes.fail",
                          "_pytest.python_api.raises", "raises", "fail"}
_TEST_INFRA_HOOK_RE = re.compile(r"^pytest_\w+$")


@dataclass(frozen=True)
class AstFinding:
    kind: str
    lineno: int
    snippet: str
    target: str | None            # "test_x" | "TestC::test_m" | None (file-level)

    @property
    def aimable(self) -> bool:
        return KIND_META[self.kind][0]

    @property
    def marker(self) -> str | None:
        return KIND_META[self.kind][1]

    OBSERVATION_KINDS = frozenset({"conftest-parametrize-hook", "plugin-parametrize-hook",
                                   "test-collection-unresolved", "test-collection-resolved", "infra-observed"})

    @property
    def observation(self) -> bool:
        return self.kind in self.OBSERVATION_KINDS


# ----------------------------------------------------------------------------- guarded helpers

def safe_dump(node: ast.AST) -> str:
    """ast.dump without a RecursionError: a pathological expression folds to an opaque token."""
    try:
        return ast.dump(node)
    except RecursionError:
        return f"<deep:{type(node).__name__}:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}>"


def parse_or_none(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


# Statement types whose body may or may not run, or may run many times. A binding inside one of
# these is not an unconditional binding of the enclosing scope (verifier 6, V6N-44). `with` is NOT
# here: its body is always entered. `match_case` is, because only one case runs.
_CONDITIONAL: tuple[type, ...] = tuple(
    c for c in (getattr(ast, n, None) for n in
                ("If", "Try", "TryStar", "While", "For", "AsyncFor", "match_case", "IfExp", "ExceptHandler"))
    if c is not None)


def _scope_statements(scope: ast.AST) -> list[ast.stmt]:
    """The statements that belong to THIS scope, descending into blocks and stopping at a nested
    scope. A nested `def`'s statements are its own; counting them here is what let a nested local
    fold in the enclosing function (verifier 6, V6N-43)."""
    out: list[ast.stmt] = []

    def walk(body) -> None:
        for st in body:
            out.append(st)
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for field, value in ast.iter_fields(st):
                if field in ("body", "orelse", "finalbody") and isinstance(value, list):
                    walk(value)
                elif field in ("handlers", "cases") and isinstance(value, list):
                    for sub in value:
                        walk(sub.body)

    walk(getattr(scope, "body", []) or [])
    return out


def _module_statements(tree: ast.Module):
    """Module-level statements, descending into if/try/with blocks (still module scope)."""
    out = []

    def walk(body):
        for s in body:
            out.append(s)
            if isinstance(s, ast.If):
                walk(s.body); walk(s.orelse)
            elif isinstance(s, ast.Try):
                walk(s.body); walk(s.orelse); walk(s.finalbody)
                for h in s.handlers:
                    walk(h.body)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                walk(s.body)
    walk(tree.body)
    return out


# ----------------------------------------------------------------------------- resolution

def _bound_names(scope: ast.AST, *, module_level: bool,
                 record: dict[str, set[str]] | None = None) -> tuple[dict[str, int], set[str]]:
    """(how many times each name is BOUND in this scope, names that are unsafe to fold at all).

    SIXTH CYCLE (verifier 5, F2). Resolving a name to a constant is only sound when the name has
    exactly ONE binding in its scope and that binding is a plain `name = <constant expression>`. Every
    other way Python can bind a name — a parameter, a loop target, a `with ... as`, an `except ... as`,
    an augmented assignment, a walrus, a second assignment, an import, a def/class, a `del`, or a
    `global`/`nonlocal` declaration — either changes the value or puts it out of this function's
    reach, so it must make the name UNFOLDABLE rather than merely be ignored. Counting bindings is
    what separates `ok = True; assert ok` (foldable) from `ok = True; ok = f(); assert ok` (not).

    SEVENTH CYCLE (verifier 6, F2 — a REGRESSION introduced by the paragraph above). "Is this name
    rebound?" turned out to be the wrong question, and asking it alone made the check REFUSE HONEST
    CODE. `errors.append(x)` rebinds nothing, so `errors = []` counted one binding, folded to `[]`,
    and `assert not errors` — the mainstream collect-then-assert idiom — was reported as a
    constant-true tautology on a genuine bug-fixing PR. The receipt contradicted itself in place: C2
    printed `[A/constant-true] assert not errors` while C1 printed that same assertion going red on
    the reverted tree with `assert not [(5, 3), (9, 4)]`.

    Rebinding is only one of the ways a name stops being worth what its binding said. The question
    the census asks now is the wider one — CAN ANYTHING IN THIS SCOPE REACH THE OBJECT? — and a name
    is unsafe when any of these is true, none of which is a rebinding:

      * a METHOD CALL on the name (`errors.append(x)`, `d.update(...)`);
      * the name PASSED AS AN ARGUMENT to anything (`fill(d)`), including as `*d` / `**d`;
      * a SUBSCRIPT or ATTRIBUTE store rooted at the name (`d['k'] = v`, `xs[:] = ...`, `o.a = v`),
        or an augmented assignment or `del` through one;
      * the name MUTATED OR DECLARED by a nested scope — a def, a lambda or a class body that calls
        a method on it, hands it to a call, stores through it, declares it `global`/`nonlocal`, or
        returns it. A nested scope that only READS the name is not enough: reading cannot change
        the value, and refusing on a read would delete the tier's own catch (`f = lambda: ok`);
      * a binding that is not UNCONDITIONAL in this scope (inside `if`, `try`, a loop or a
        `match` case), because whether it happened at all is a runtime question;
      * a `match` CAPTURE PATTERN, which binds the name to the subject and is not a comparison.

    EIGHTH CYCLE (verifier 7, F2-ALIAS — the seventh cycle's own fix, one indirection later). The
    paragraph above asks the right question and answers it about NAMES. A plain `alias = errors`
    hands the OBJECT a second handle, and the census followed only the first: `alias.append(1)`
    marked `alias`, `errors` kept its one clean binding, folded to `[]`, and `assert not errors` was
    reported as a constant-true tautology on an honest bug-fixing PR — the same false refusal the
    seventh cycle fixed, reached through one more name. Verifier 7's decisive pair, at a588998: with
    the alias, counts {'errors': 1, 'alias': 1}, unsafe ['alias'], `errors` foldable=True; with the
    mutation written directly, unsafe ['errors'], foldable=False.

    So ALIASING is unsafe in BOTH DIRECTIONS. When a name appears in the VALUE of a binding —
    `alias = errors`, `box = {'e': errors}`, `pair = [errors]`, `sink, spare = errors, []`, or the
    thing a `for`/comprehension iterates — the SOURCE name is unsafe (something else can now reach
    its object) and so is the TARGET (it may BE that object). Marking both ends is what makes the
    rule transitive: `b = errors; c = b; c.append(1)` closes at any chain depth without the census
    having to model the chain. A name handed OUT of the scope by `return`/`yield` is unsafe for the
    same reason, in the one direction that applies.

    The refusals are on the SHAPE, never on a type inference. `buf = ''; fill(buf)` cannot really be
    mutated — str is immutable — and is refused anyway, because a census that has to be right about
    which types are immutable and which methods mutate is a census that will be wrong about one of
    them on somebody's honest PR. Refusing to fold costs a report nobody reads; folding wrongly costs
    a false refusal on a correct fix, which is the failure this whole product exists to not commit.
    What that costs is not left to be discovered: every category here is rendered into
    `NOT_COVERED_AST` by `_NAME_RESIDUAL_LINES()`, from this function's own `_UNSAFE_CATEGORIES`."""
    counts: dict[str, int] = {}
    unsafe: set[str] = set()

    def bump(name: str | None) -> None:
        if name:
            counts[name] = counts.get(name, 0) + 1

    def mark(name: str | None, why: str) -> None:
        """Refuse to fold `name`, under a category that must already be PUBLISHED.

        EIGHTH CYCLE (verifier 7, F-DISCLOSURE). The KeyError is the point, and it is deliberately
        not a soft check: a refusal category that has not been written into `_UNSAFE_CATEGORIES`
        cannot reach a verdict at all, so the code cannot acquire a reason to refuse that the
        README does not carry. The previous arrangement — mark freely, describe separately — is
        how six categories came to be enforced and none of them published."""
        if why not in _UNSAFE_CATEGORIES:
            raise KeyError(f"unsafe category {why!r} is not published in _UNSAFE_CATEGORIES")
        if name:
            unsafe.add(name)
            if record is not None:
                record.setdefault(why, set()).add(name)     # the test's read-out; never set in production

    def mark_all(names, why: str) -> None:
        for n in names:
            mark(n, why)

    def root_name(node: ast.AST | None) -> str | None:
        """The NAME a subscript/attribute chain is rooted at: `d['k'].x` -> `d`, `[1][0]` -> None.

        SEVENTH CYCLE (verifier 6, F2/V6N-63/V6N-64). A store through a subscript or an attribute
        mutates the object the root name is bound to, and rebinds nothing, so `target()` walked
        silently past it and the census went on folding a dict that had just been filled."""
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def target(node: ast.AST | None) -> None:
        """Every name a binding target can reach. A destructuring target is counted AND marked
        unsafe: the folder does not evaluate the right-hand side element-wise."""
        if node is None:
            return
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            # NOT a binding of the root name — a MUTATION of the object behind it. It must not be
            # counted (that would be a second binding, changing an unrelated verdict) and it must
            # make the name unsafe.
            mark(root_name(node), "store-through")
            return
        if isinstance(node, ast.Name):
            bump(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for e in node.elts:
                target(e)
                mark_all((n.id for n in ast.walk(e) if isinstance(n, ast.Name)), "destructured")
        elif isinstance(node, ast.Starred):
            target(node.value)
            mark_all((n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)), "destructured")

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = scope.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if arg is not None:
                bump(arg.arg)
                mark(arg.arg, "parameter")     # a parameter's value comes from the CALLER

    def mutated_by(node: ast.AST) -> None:
        """Mark every name `node` can MUTATE WITHOUT REBINDING (verifier 6, F2).

        Three shapes, and none of them is an assignment: a method call on the name, the name handed
        to a call as an argument, and a store through a subscript or an attribute of it."""
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute):
                    # errors.append(x) — a method may mutate
                    mark(root_name(n.func.value), "method-call")
                for a in [*n.args, *(k.value for k in n.keywords)]:
                    a = a.value if isinstance(a, ast.Starred) else a
                    if isinstance(a, ast.Name):
                        mark(a.id, "handed-to-call")     # fill(d) — the callee may mutate
            elif isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.ctx, (ast.Store, ast.Del)):
                mark(root_name(n), "store-through")

    def handles_in(value: ast.AST | None) -> set[str]:
        """Every name in a binding's VALUE that the binding could hand a SECOND HANDLE to.

        EIGHTH CYCLE (verifier 7, F2-ALIAS). A bare `ast.Name` in Load position inside a value is
        an object this binding can carry away — directly (`alias = errors`), inside a container
        display (`box = {'e': errors}`, `pair = [errors]`), through a conditional expression, or
        out of a destructuring (`sink, spare = errors, []`). The ONE Load name that is not carried
        away is a call's own callee: `n = len(xs)` aliases `xs`, never `len`, and excluding the
        func position is what keeps the pure-builtin folds (`bool`, `len`, `str`, `repr`, `any`,
        `all`) alive after this widening.

        This is deliberately a SHAPE rule and not an escape analysis. `b = list(errors)` copies and
        is refused with the rest, because a census that has to know which callees copy and which
        alias is a census that will be wrong about one of them on somebody's honest PR."""
        if value is None:
            return set()
        skip = {id(n.func) for n in ast.walk(value) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        return {n.id for n in ast.walk(value)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in skip}

    def target_names(node: ast.AST | None) -> set[str]:
        """Every name a binding target NAMES — a plain name, each name in a destructuring, and the
        ROOT of a subscript/attribute store (`box['e'] = errors` reaches the object behind `box`)."""
        if node is None:
            return set()
        r = root_name(node)
        if r is not None and isinstance(node, (ast.Subscript, ast.Attribute)):
            return {r}
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def alias_binding(value: ast.AST | None, *targets: ast.AST | None) -> None:
        """One binding, both ends. If the value can carry a handle, the SOURCE names lose their
        single-binding guarantee (something else can reach their objects now) and so do the TARGET
        names (one of them may BE that object). Marking both ends is what makes the rule transitive
        through `b = errors; c = b` without the census modelling the chain."""
        src = handles_in(value)
        if not src:
            return
        mark_all(src, "aliased-source")
        for t in targets:
            mark_all(target_names(t), "aliased-target")

    def handed_out(value: ast.AST | None) -> None:
        """`return errors` / `yield errors` hands the object to a caller this file cannot see.
        Source direction only — there is no target in this scope to mark."""
        mark_all(handles_in(value), "handed-out")

    # A nested function/class is its own scope: its body binds nothing here, but its NAME does.
    #
    # SEVENTH CYCLE (verifier 6, V6N-43). A FUNCTION scope used to descend into a nested `def` and
    # count its locals as its own, so `def _s(): ok = True` inside `test_a` made `ok` a constant of
    # `test_a` — where the name actually refers to the module's `ok = False`. Both scopes now stop at
    # a nested def, and what the nested scope can still do to an enclosing name — mutate it in place,
    # or declare it `global`/`nonlocal` — is collected instead of being walked past.
    def capture(scope_node: ast.AST) -> None:
        mutated_by(scope_node)
        for n in ast.walk(scope_node):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                mark_all(n.names, "global-nonlocal")
            elif isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom)):
                # EIGHTH CYCLE (verifier 7, F2-ALIAS): `def get(): return errors` then
                # `get().append(1)` — the mutation is rooted at a CALL, so `root_name` finds no
                # name to blame. The handle left through the nested scope's return.
                handed_out(n.value)

    def walk(node: ast.AST, *, top: bool, cond: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bump(child.name)
                mark(child.name, "nested-scope-name")
                capture(child)
                continue                       # a nested scope's bindings are ITS OWN, never this scope's
            if isinstance(child, ast.Lambda):
                capture(child)
                continue
            if isinstance(child, ast.MatchAs) and child.name:
                bump(child.name)               # `case ok:` BINDS ok to the subject; it is not a test
                mark(child.name, "match-capture")
            elif isinstance(child, ast.MatchStar) and child.name:
                bump(child.name)
                mark(child.name, "match-capture")
            elif isinstance(child, ast.MatchMapping) and child.rest:
                bump(child.rest)
                mark(child.rest, "match-capture")
            if isinstance(child, ast.Call):
                mutated_by(child)
            if isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom)):
                handed_out(child.value)        # the object leaves this scope (verifier 7, F2-ALIAS)
            if isinstance(child, ast.Assign):
                for t in child.targets:
                    target(t)
                alias_binding(child.value, *child.targets)
            elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
                target(child.target)
                alias_binding(child.value, child.target)
                if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                    mark(child.target.id, "augmented")
            elif isinstance(child, ast.NamedExpr):
                target(child.target)
                alias_binding(child.value, child.target)
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                target(child.target)
                alias_binding(child.iter, child.target)
                # rebound once per iteration
                mark_all((n.id for n in ast.walk(child.target) if isinstance(n, ast.Name)), "loop-target")
            elif isinstance(child, ast.withitem):
                target(child.optional_vars)
                alias_binding(child.context_expr, child.optional_vars)
            elif isinstance(child, ast.ExceptHandler):
                bump(child.name)
                mark(child.name, "except-as")
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for al in child.names:
                    bump((al.asname or al.name).split(".")[0])
            elif isinstance(child, ast.Delete):
                for t in child.targets:
                    target(t)
                    mark_all((n.id for n in ast.walk(t) if isinstance(n, ast.Name)), "deleted")
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                for nm in child.names:
                    bump(nm)
                    mark(nm, "global-nonlocal")   # bound somewhere this scope cannot see
            elif isinstance(child, ast.comprehension):
                target(child.target)
                alias_binding(child.iter, child.target)
                mark_all((n.id for n in ast.walk(child.target) if isinstance(n, ast.Name)), "loop-target")
            if cond:
                # SEVENTH CYCLE (verifier 6, V6N-44). ONE binding is not the same as an
                # UNCONDITIONAL one. `if sys.platform: ok = True` binds `ok` once, and whether it is
                # bound at all at the assert is a runtime question the folder never evaluates.
                mark_all((n.id for n in ast.walk(child)
                          if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)), "conditional")
            walk(child, top=False, cond=cond or isinstance(child, _CONDITIONAL))

    walk(scope, top=True, cond=False)
    # (the seventh cycle's `unsafe.discard("")` is gone: `mark()` is the only writer and it drops a
    # falsy name at the door, so the sentinel it removed can no longer be added.)
    return counts, unsafe


# ------------------------------------------------------- the inverted default (ninth cycle)
#
# NINTH CYCLE (verifier 8, F-SCOPE). Three cycles in a row fixed this census, and each was broken by
# the next spelling: rebinding, then in-place mutation, then aliasing within a scope, then aliasing
# across one. The pattern is not any of those shapes. It is that `_bound_names` tries to prove a name
# IS constant by ENUMERATING the ways it might not be — and that enumeration can never be complete,
# because Python keeps offering new ways to hand an object to a second holder (a parameter DEFAULT is
# not an assignment statement at all, and no enumeration of assignment shapes could ever have reached
# it).
#
# So the default is INVERTED. This function does not ask what could go wrong with a name; it asks
# whether the WHOLE FILE can be shown never to let the name's object escape the assertion, and
# answers NO unless every single appearance is on a short allowlist:
#
#   * ONE binding, as the sole target of a plain `name = ...` or `name: T = ...`;
#   * every other appearance a CLASSIFIED READ — the operand of an `assert`, `if` or `while` test,
#     reached only through `not` / `and` / `or`, a comparison, an arithmetic operator, a conditional
#     expression, a subscript read or a slice.
#
# ANYTHING ELSE — passed as an argument, returned, yielded, stored into a container, an attribute or
# a subscript, read through an attribute, bound or read inside a nested scope in any way that is not
# a classified read, used as a parameter default, rebound, deleted, declared `global`/`nonlocal`,
# imported, matched, or read in a position this list does not name — makes the name UNFOLDABLE, with
# NO attempt to decide whether that particular use was harmless. A read the analysis cannot classify
# is refused for being unclassified, which is the property that makes the rule sound by construction
# rather than sound until the next spelling.
#
# The trade is verifier 8's and it is stated, not discovered later: a missed tautology is a RESIDUAL
# (published, and measured by `measure_const_fold_flip_cost.py`), while an accusation of gaming aimed
# at ordinary code is a PRODUCT DEFECT. Literal tautologies — `assert True`, `assert 1 == 1` — carry
# no name at all, so no census decides them and they are untouched; they are the mainstream fake-green
# shape. The false-refusal direction now has a permanent harness of its own
# (`checks/verify/corpus_honest.py`), which is the measurement whose absence let this recur three times.
#
# `_bound_names` is KEPT and still runs. It can only ADD refusals to this one, never remove one, so
# every shape the previous three cycles closed stays closed and stays published by name.
_READ_TERMINALS: tuple[tuple[type, str], ...] = tuple(
    (c, f) for c, f in ((getattr(ast, n, None), fld) for n, fld in
                        (("Assert", "test"), ("If", "test"), ("While", "test"))) if c is not None)


def census_report(source: str) -> dict[str, Any]:
    """The census's STATED reasons, per scope and per name — the read-out the honest corpus asserts.

    ELEVENTH CYCLE (verifier 10, V10-F4; the owner's charter, item 4). The corpus proved the fold
    tier alive (a twin that must be flagged) but nothing proved WHICH refusal kept an honest member
    PROVEN: verifier 10 switched off `aliased-source`/`aliased-target`, `method-call`,
    `handed-to-call` and `handed-out` one at a time and the corpus reported OK 36 every time,
    because `_escaping_names` (the ninth cycle's inverted default) refuses the same names on its
    own. Redundancy is the design; an unobservable tier is not. So each member now DECLARES the
    category its structure exercises, and this function is how the corpus checks that the census
    still says so — the same `_bound_names` the fold consults, with its `record` read out, never a
    second implementation.

    Returns {"<module>": {name: {categories}}, "<function name>": {...}, ..., "__escaping__": {names
    the whole-file sound rule refuses}}. Never raises on unparsable input: returns {}."""
    tree = parse_or_none(source)
    if tree is None:
        return {}
    out: dict[str, Any] = {}
    rec: dict[str, set[str]] = {}
    try:
        _bound_names(tree, module_level=True, record=rec)
    except RecursionError:
        rec = {}
    out["<module>"] = _invert(rec)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rec = {}
            try:
                _bound_names(node, module_level=False, record=rec)
            except RecursionError:
                rec = {}
            out[node.name] = _invert(rec)
    try:
        out["__escaping__"] = set(_escaping_names(tree))
    except RecursionError:
        out["__escaping__"] = set()
    return out


def _invert(rec: dict[str, set[str]]) -> dict[str, set[str]]:
    by_name: dict[str, set[str]] = {}
    for why, names in rec.items():
        for n in names:
            by_name.setdefault(n, set()).add(why)
    return by_name


def _escaping_names(tree: ast.AST) -> set[str]:
    """Every name in this file with at least one appearance that is NOT on the allowlist above.

    The RETURN is the refusal set: a name in it is unfoldable, whatever else the census says. A name
    NOT in it has been shown, appearance by appearance, never to leave a classified read."""
    out: set[str] = set()

    def escape_all(node: ast.AST | None) -> None:
        """Every name anywhere under `node` escapes — including the names bound by forms that are
        not `ast.Name` at all (a parameter, an `except ... as`, an import alias, a `def`/`class`
        name, a `match` capture, a `global`/`nonlocal` declaration)."""
        if node is None:
            return
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                out.add(n.id)
            elif isinstance(n, ast.arg):
                out.add(n.arg)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                out.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                out.update(n.names)
            elif isinstance(n, ast.alias):
                out.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, ast.MatchAs) and n.name:
                out.add(n.name)
            elif isinstance(n, ast.MatchStar) and n.name:
                out.add(n.name)
            elif isinstance(n, ast.MatchMapping) and n.rest:
                out.add(n.rest)

    def read(node: ast.AST | None) -> None:
        """A position where a bare `Name` in Load context is a CLASSIFIED READ: it can neither change
        the object nor hand it to anyone. Anything else here falls through to `escape_all`."""
        if node is None:
            return
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                out.add(node.id)                       # a Store or Del in a test position is a binding
            return
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, ast.UnaryOp):
            read(node.operand)
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                read(v)
        elif isinstance(node, ast.BinOp):
            read(node.left), read(node.right)
        elif isinstance(node, ast.Compare):
            read(node.left)
            for c in node.comparators:
                read(c)
        elif isinstance(node, ast.IfExp):
            read(node.test), read(node.body), read(node.orelse)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            read(node.value), read(node.slice)
        elif isinstance(node, ast.Slice):
            read(node.lower), read(node.upper), read(node.step)
        else:
            escape_all(node)                           # a read this analysis cannot classify

    def body(stmts) -> None:
        for st in stmts or ():
            stmt(st)

    def stmt(node: ast.AST) -> None:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            escape_all(node.value)                     # a binding's VALUE hands out a handle; never a read
            return
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            escape_all(node.value), escape_all(node.annotation)
            return
        for cls, field in _READ_TERMINALS:
            if isinstance(node, cls):
                read(getattr(node, field, None))
                if isinstance(node, ast.Assert):
                    escape_all(node.msg)               # the msg is handed to AssertionError
                else:
                    body(getattr(node, "body", ())), body(getattr(node, "orelse", ()))
                return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            escape_all(node.target), escape_all(node.iter)
            body(node.body), body(node.orelse)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                escape_all(item.context_expr), escape_all(item.optional_vars)
            body(node.body)
            return
        if isinstance(node, getattr(ast, "TryStar", ast.Try)) or isinstance(node, ast.Try):
            body(node.body), body(node.orelse), body(node.finalbody)
            for h in node.handlers:
                escape_all(h.type)
                if h.name:
                    out.add(h.name)
                body(h.body)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)                         # the scope's own name is bound, not read
            for d in node.decorator_list:
                escape_all(d)
            if isinstance(node, ast.ClassDef):
                for b in node.bases:
                    escape_all(b)
                for k in node.keywords:
                    escape_all(k.value)
            else:
                escape_all(node.args)                  # parameters, defaults and annotations, all of them
                escape_all(node.returns)
            body(node.body)
            return
        if isinstance(node, getattr(ast, "Match", ())):
            escape_all(node.subject)
            for case in node.cases:
                escape_all(case.pattern), escape_all(case.guard)
                body(case.body)
            return
        escape_all(node)                               # return, yield, del, augassign, import, expr, ...

    body(getattr(tree, "body", ()) or ())
    if not hasattr(tree, "body"):
        escape_all(tree)                               # not a module: nothing here can be vouched for
    return out


class _Resolver:
    """Local name -> canonical dotted name. Module scope from imports and simple assignments at module
    level (two passes so an alias of an alias resolves); function scope from assignments inside the
    function, consulted first while `self.scope` is that function. Names bound to constant strings are
    tracked the same way (scope-aware, so `text = ""` in one test does not fold `if text:` in another)."""

    def __init__(self, tree: ast.AST):
        self.module_aliases: dict[str, str] = {}
        self.module_strings: dict[str, str] = {}
        self.local_aliases: dict[int, dict[str, str]] = {}
        self.local_strings: dict[int, dict[str, str]] = {}
        # SIXTH CYCLE (verifier 5, F2): the VALUE NODE of a name bound exactly once to a constant
        # expression, for any constant type. Nodes, not values, because folding a value needs a
        # resolver and the resolver is what is being built here; `_fold` resolves them lazily.
        self.module_consts: dict[str, ast.expr] = {}
        self.local_consts: dict[int, dict[str, ast.expr]] = {}
        self.star_modules: list[str] = []
        self.bound: set[str] = set()
        self.scope: ast.AST | None = None
        self._resolving: set[tuple[int, str]] = set()
        self._shadow_cache: dict[int, set[str]] = {}
        self._census_cache: dict[int, tuple[dict[str, int], set[str]]] = {}
        self._escaping_cache: set[str] | None = None
        # ELEVENTH CYCLE (V10-F7): attribute names any file in the PR's input STORES (`x.attr = ...`,
        # `setattr(x, "attr", ...)`), handed in by audit_python from the InheritanceCorpus. A class
        # attribute the fold would treat as constant is refused if ANY file in the input stores it.
        self.attrs_stored_elsewhere: frozenset[str] = frozenset()
        self._tree: ast.AST = tree
        self._module: ast.Module | None = tree if isinstance(tree, ast.Module) else None
        stmts = _module_statements(tree) if isinstance(tree, ast.Module) else list(ast.walk(tree))
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.bound.add(s.name)
        for _ in range(2):
            for s in stmts:
                self._bind(s, self.module_aliases, self.module_strings, module_level=True)
        if isinstance(tree, ast.Module):
            self.module_consts = self._const_bindings(tree, _scope_statements(tree), module_level=True)
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                la: dict[str, str] = {}
                ls: dict[str, str] = {}
                for _ in range(2):
                    for s in ast.walk(fn):
                        if isinstance(s, (ast.Assign, ast.Import, ast.ImportFrom)):
                            self._bind(s, la, ls, module_level=False)
                if la or ls:
                    self.local_aliases[id(fn)] = la
                    self.local_strings[id(fn)] = ls
                lc = self._const_bindings(fn, _scope_statements(fn), module_level=False)
                if lc:
                    self.local_consts[id(fn)] = lc

    @staticmethod
    def _const_bindings(scope: ast.AST, stmts: list, *, module_level: bool) -> dict[str, ast.expr]:
        """`name -> value node` for every name this scope binds EXACTLY ONCE, by a plain
        `name = <expr>`, and that nothing else in the scope can rebind or REACH AROUND.

        SEVENTH CYCLE. Two changes, both narrowing what is offered as a constant:

        * the statements are THIS SCOPE'S OWN (`_scope_statements`), not `ast.walk(scope)`. The walk
          reached into nested `def`s and offered their locals as this scope's constants, so
          `def _s(): ok = True` inside `test_a` folded `ok` at `test_a`'s assert — where the name is
          the module's (verifier 6, V6N-43);
        * `ast.AnnAssign` counts. `_bound_names` always COUNTED an annotated binding, and this only
          ever looked at `ast.Assign`, so `ok: bool = True` was counted as a binding and then never
          offered — `ok = True` was flagged and its annotated twin was not. One keyword apart, two
          tiers, and no reader could predict which they would get (verifier 6, V6N-62)."""
        counts, unsafe = _bound_names(scope, module_level=module_level)
        out: dict[str, ast.expr] = {}
        for st in stmts:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                name, value = st.targets[0].id, st.value
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                name, value = st.target.id, st.value
            else:
                continue
            if counts.get(name, 0) == 1 and name not in unsafe:
                out[name] = value
        return out

    def unfoldable_name(self, name: str) -> bool:
        """True when the CENSUS has an opinion about this name and the opinion is `do not fold it`.

        SEVENTH CYCLE (verifier 6, V6N-46). This is the gate the string path never had. A name the
        census marks unsafe — mutated in place, handed to a call, bound conditionally, bound twice —
        must be unknown to EVERY resolution path, not only to the one added in the sixth cycle.
        A name the census simply never saw bound is not unfoldable; it is unknown to the census, and
        the alias/string resolver is still allowed to answer for it.

        NINTH CYCLE (verifier 8, F-SCOPE). The census's opinion is now the SECOND question. The first
        is `_escaping_names`, which refuses unless every appearance of the name in the whole file is
        on the allowlist — the inverted default. It runs FIRST and its refusal is final, so the
        `this scope binds it cleanly` shortcut below can no longer wave a name past an appearance
        nobody classified. Keeping `_bound_names` behind it costs nothing and keeps every refusal the
        previous three cycles published reachable BY NAME."""
        if name in self._escaping():
            return True
        for scope, module_level in ((self.scope, False), (self._module, True)):
            if scope is None:
                continue
            counts, unsafe = self._census(scope, module_level)
            if name in unsafe or counts.get(name, 0) > 1:
                return True
            if name in counts:
                return False               # this scope binds it cleanly; an outer scope cannot veto
        return False

    def _escaping(self) -> set[str]:
        """`_escaping_names` over this file, computed once. Memoised on the resolver rather than
        per scope: the question it answers is about the FILE, which is the whole point of it."""
        if self._escaping_cache is None:
            self._escaping_cache = _escaping_names(self._tree)
        return self._escaping_cache

    def _census(self, scope: ast.AST, module_level: bool) -> tuple[dict[str, int], set[str]]:
        """`_bound_names` for one scope, memoised — the fold asks per operand."""
        hit = self._census_cache.get(id(scope))
        if hit is None:
            hit = _bound_names(scope, module_level=module_level)
            self._census_cache[id(scope)] = hit
        return hit

    def _const_node(self, name: str) -> tuple[ast.expr | None, int]:
        """(the value node bound to `name`, a scope key). Function scope shadows module scope, and a
        name the FUNCTION binds unsafely (a parameter, a loop target) must not fall through to a
        module-level constant of the same name."""
        if self.scope is not None:
            lc = self.local_consts.get(id(self.scope))
            if lc and name in lc:
                return lc[name], id(self.scope)
            # CACHED per scope. Without this, the census walks the whole function's AST on EVERY name
            # the fold looks up and does not find — O(function size) per lookup, in a fold that does a
            # lookup per operand. HONEST NOTE ON THE MEASUREMENT: this was written believing it
            # explained a slow test run, and a before/after benchmark (400 call-bound names x 400
            # assertions, the shape that takes this path) showed BOTH at 0.00s, so it did NOT. The
            # slow run was contention from concurrently running e2e harnesses. The cache is kept
            # because repeating an O(n) walk per lookup is wrong on its own terms and the pathological
            # input is one file away, not because it fixed anything measured.
            shadow = self._shadow_cache.get(id(self.scope))
            if shadow is None:
                counts, unsafe = _bound_names(self.scope, module_level=False)
                shadow = set(counts) | unsafe
                self._shadow_cache[id(self.scope)] = shadow
            if name in shadow:
                return None, 0
        return self.module_consts.get(name), 0

    def _bind(self, node: ast.AST, aliases: dict[str, str], strings: dict[str, str], *, module_level: bool) -> None:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                    self.bound.add(a.asname)
                else:
                    aliases[a.name.split(".")[0]] = a.name.split(".")[0]
                    self.bound.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            for a in node.names:
                if a.name == "*":
                    if mod and mod not in self.star_modules:
                        self.star_modules.append(mod)
                    continue
                aliases[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
                self.bound.add(a.asname or a.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            self.bound.add(name)
            s = self.const_str(node.value)
            if s is not None:
                strings[name] = s
                return
            c = self.canon(node.value)
            if c and c != name:
                aliases[name] = c

    # -- lookups honouring the current scope
    def _alias(self, name: str) -> str | None:
        if self.scope is not None:
            la = self.local_aliases.get(id(self.scope), {})
            if name in la:
                return la[name]
            if name in self.local_strings.get(id(self.scope), {}):
                return None
        return self.module_aliases.get(name)

    def _string(self, name: str) -> str | None:
        if self.scope is not None:
            ls = self.local_strings.get(id(self.scope), {})
            if name in ls:
                return ls[name]
            if name in self.local_aliases.get(id(self.scope), {}):
                return None
        return self.module_strings.get(name)

    def const_str(self, node: ast.AST | None) -> str | None:
        try:
            return self._const_str(node)
        except RecursionError:
            return None

    def _const_str(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
            try:
                return node.value.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if isinstance(node, ast.Name):
            return self._string(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = self._const_str(node.left), self._const_str(node.right)
            return a + b if a is not None and b is not None else None
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                s = self._const_str(v) if not isinstance(v, ast.FormattedValue) else self._const_str(v.value)
                if s is None:
                    return None
                parts.append(s)
            return "".join(parts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "decode":
            return self._const_str(node.func.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join" and len(node.args) == 1 \
                and isinstance(node.args[0], (ast.List, ast.Tuple)):
            sep = self._const_str(node.func.value)
            parts = [self._const_str(e) for e in node.args[0].elts]
            if sep is not None and all(p is not None for p in parts):
                return sep.join(parts)  # type: ignore[arg-type]
        return None

    def canon(self, node: ast.AST | None) -> str | None:
        try:
            return self._canon(node)
        except RecursionError:
            return None

    def _canon(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            a = self._alias(node.id)
            if a is not None:
                return a
            if node.id in self.bound:
                return node.id
            if self.star_modules and node.id not in ("True", "False", "None"):
                # an unbound name after `from X import *` is X.name — the first star module that is sensitive wins
                for m in self.star_modules:
                    if m in _SENSITIVE_MODULES:
                        return f"{m}.{node.id}"
                return f"{self.star_modules[0]}.{node.id}"
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._canon(node.value)
            return f"{base}.{node.attr}" if base else None
        if isinstance(node, ast.Call):
            f = node.func
            fc = self._canon(f)
            if fc in ("getattr", "builtins.getattr") and len(node.args) >= 2:
                base = self._canon(node.args[0])
                name = self._const_str(node.args[1])
                if base and name is not None:
                    return f"{base}.{name}"
                return f"{base}.<dynamic>" if base else "<dynamic>"
            if fc in ("__import__", "builtins.__import__", "importlib.import_module") and node.args:
                return self._const_str(node.args[0]) or "<dynamic-import>"
            if fc in ("vars", "builtins.vars") and node.args:
                base = self._canon(node.args[0])
                return f"{base}.__dict__" if base else None
            if fc in ("globals", "builtins.globals"):
                return "<module>"
            return fc
        if isinstance(node, ast.Subscript):
            base = self._canon(node.value)
            key = self._const_str(node.slice)
            if base == "sys.modules":
                if key == "__name__":
                    return "<module>"
                return key if key is not None else "<dynamic-module>"
            if base and base.endswith(".__dict__"):
                owner = base[: -len(".__dict__")]
                return f"{owner}.{key}" if key is not None else f"{owner}.<dynamic>"
            if base == "<module>":
                return f"<module>.{key}" if key is not None else "<module>.<dynamic>"
            if base and key is not None:
                return f"{base}.{key}"
            return None
        if isinstance(node, ast.Await):
            return self._canon(node.value)
        return None


def is_sensitive(canon_name: str | None) -> bool:
    if not canon_name:
        return False
    root = canon_name.split(".", 1)[0]
    return root in _SENSITIVE_MODULES or canon_name.startswith("<module>")


# ----------------------------------------------------------------------------- constants

_UNKNOWN = object()


def _pure(node: ast.AST) -> bool:
    """A side-effect-free expression whose two evaluations name the same thing: a name, an attribute
    chain, or (eleventh cycle, V10-F7) a subscript of one by a constant or another pure expression —
    `row["want"] == row["want"]` under a loop variable is `x == x` with an index."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _pure(node.value)
    if isinstance(node, ast.Subscript):
        idx = node.slice
        return _pure(node.value) and (isinstance(idx, ast.Constant) or _pure(idx))
    return False


_DEF_TEST_LINE_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+(test\w*)\s*\(")
_BARE_RETURN_RE = re.compile(r"^\s*return\s*(#.*)?$")
_DOCSTRING_QUOTES = ('"""', "'''")


def early_return_lines(source: str) -> list[tuple[int, str]]:
    """[(1-based line number of the `return`, test name)] for a BARE `return` that is the FIRST
    statement of a `def test*` body — a silent skip in effect.

    A SECOND TIER, deliberately independent of the AST. The AST tier is the only place early-return
    is found, and it gives up on a pathological expression (a 4000-term line raises RecursionError),
    which is exactly when an attacker would hide one — verifier 3's V3-C2-earlyreturn-under-deep-expr.
    Deliberately narrow: a `return` on the line right after the `def` (blank lines, comments and a
    docstring skipped) cannot come after an assertion, so it needs no flow analysis and cannot
    over-flag the way a general `return` scan would."""
    out: list[tuple[int, str]] = []
    lines = source.splitlines()
    for i, line in enumerate(lines):
        m = _DEF_TEST_LINE_RE.match(line)
        if m is None:
            continue
        stripped = line.rstrip()
        if stripped.endswith(("\\", ",", "(")) or not stripped.endswith(":"):
            continue                                  # a multi-line signature: the body is not next
        indent = len(m.group(1))
        j = i + 1
        open_doc = None
        while j < len(lines):
            s = lines[j].strip()
            if open_doc is not None:
                if s.endswith(open_doc):
                    open_doc = None
                j += 1
                continue
            if not s or s.startswith("#"):
                j += 1
                continue
            quote = next((q for q in _DOCSTRING_QUOTES if s.startswith(q)), None)
            if quote is None:
                break
            if not (len(s) > len(quote) and s.endswith(quote)):
                open_doc = quote
            j += 1
        if j >= len(lines):
            continue
        body = lines[j]
        if len(body) - len(body.lstrip()) <= indent:
            continue                                  # dedented: the body is empty / not this def's
        if _BARE_RETURN_RE.match(body):
            out.append((j + 1, m.group(2)))
    return out




class _FoldRefused(Exception):
    """The fold hit its budget, or a value it will not build. Caught in const_value -> _UNKNOWN."""


class _Budget:
    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes = _FOLD_MAX_NODES

    def spend(self) -> None:
        self.nodes -= 1
        if self.nodes <= 0:
            raise _FoldRefused("node budget exhausted")


def _sized(v: Any) -> int | None:
    return len(v) if isinstance(v, (str, bytes, bytearray, list, tuple, set, frozenset, dict)) else None


def _guard(v: Any) -> Any:
    """Refuse a value the fold has BUILT that is over budget. A constant already written in the
    source is never refused — nothing was built to make it."""
    n = _sized(v)
    if n is not None and n > _FOLD_MAX_LEN:
        raise _FoldRefused("built value too large")
    if isinstance(v, int) and not isinstance(v, bool) and v.bit_length() > _FOLD_MAX_BITS:
        raise _FoldRefused("built integer too wide")
    return v


_WIDTH_RE = re.compile(r"\d{4,}")


def _guard_width(spec: str) -> None:
    """A format spec or %-format whose WIDTH would build a huge string is refused BEFORE it is built:
    `f'{1:>1000000000}'` is nineteen characters of source and a gigabyte of result."""
    for run in _WIDTH_RE.findall(spec):
        if int(run) > _FOLD_MAX_LEN:
            raise _FoldRefused("format width too large")


_BINOPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift, ast.BitOr: operator.or_,
    ast.BitXor: operator.xor, ast.BitAnd: operator.and_,
    # MatMult is absent on purpose: no constant type implements `@`.
}
_CMPOPS: dict[type, Any] = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    # `is` / `is not` between CONSTANTS: identity depends on interning and differs between
    # implementations, so the fold answers the equality question it can actually decide.
    ast.Is: lambda a, b: a is b or a == b,
    ast.IsNot: lambda a, b: not (a is b or a == b),
}


def const_value(node: ast.AST, res: _Resolver) -> Any:
    """Constant-fold: the Python value when the whole expression is one this grammar evaluates, else
    _UNKNOWN. Never raises."""
    try:
        return _fold(node, res, _Budget())
    except _FoldRefused:
        return _UNKNOWN
    except (RecursionError, MemoryError, ArithmeticError, LookupError, AttributeError,
            TypeError, ValueError, UnicodeError):
        return _UNKNOWN


def _fold(node: ast.AST, res: _Resolver, bud: _Budget) -> Any:
    bud.spend()
    if type(node).__name__ in REFUSED_NODES:
        return _UNKNOWN

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = _elements(node.elts, res, bud)
        if items is _UNKNOWN:
            return _UNKNOWN
        if isinstance(node, ast.Tuple):
            return _guard(tuple(items))
        if isinstance(node, ast.Set):
            try:
                return _guard(set(items))
            except TypeError:
                return _UNKNOWN                 # an unhashable element: not a constant set
        return _guard(items)

    if isinstance(node, ast.Dict):
        out: dict = {}
        for k, v in zip(node.keys, node.values):
            fv = _fold(v, res, bud)
            if fv is _UNKNOWN:
                return _UNKNOWN
            if k is None:                       # `{**other}`
                if not isinstance(fv, dict):
                    return _UNKNOWN
                out.update(fv)
            else:
                fk = _fold(k, res, bud)
                if fk is _UNKNOWN:
                    return _UNKNOWN
                try:
                    out[fk] = fv
                except TypeError:
                    return _UNKNOWN             # an unhashable key: not a constant dict
            if len(out) > _FOLD_MAX_LEN:
                raise _FoldRefused("dict too large")
        return out

    if isinstance(node, ast.Starred):
        # only meaningful inside a display, where _elements expands it; alone it is not a value
        return _UNKNOWN

    if isinstance(node, ast.Name):
        c = res.canon(node)
        if c == "typing.TYPE_CHECKING" or node.id == "TYPE_CHECKING":
            return False
        # SIXTH CYCLE (verifier 5, F2). The resolver tracked names bound to STRINGS and no other
        # constant type, so `ok = True` / `assert ok` was PROVEN while the identical rewrite to
        # `assert True` was FAILED — an inconsistency inside one tier that no reader could predict,
        # and the one verifier 5 rode end to end (V5E17, V5E19). Every constant type the folder
        # handles is now resolved through a name, under the same one-binding census.
        #
        # SEVENTH CYCLE (verifier 6, V6N-46). That last sentence was not yet true. The STRING lookup
        # ran FIRST and consulted no census at all — it is the fifth-cycle alias/marker resolver,
        # which answers a different question — so `buf = ''` went on folding after `fill(buf)` while
        # `xs = []` after `fill(xs)` correctly did not. The census is now the gate for every constant
        # type including strings: a name this scope cannot vouch for is UNKNOWN, whatever it is bound
        # to, and the string path is consulted only for a name the census has no binding for at all
        # (an import alias, a module-level constant the census never saw bound).
        if res.unfoldable_name(node.id):
            return _UNKNOWN
        vnode, key = res._const_node(node.id)
        if vnode is None:
            s = res._string(node.id)
            if s is not None:
                return s
        if vnode is None:
            return _UNKNOWN
        guard = (key, node.id)
        if guard in res._resolving:            # `a = b` / `b = a`: a cycle is not a constant
            return _UNKNOWN
        res._resolving.add(guard)
        try:
            return _fold(vnode, res, bud)
        finally:
            res._resolving.discard(guard)

    if isinstance(node, ast.Attribute):
        if res.canon(node) == "typing.TYPE_CHECKING":
            return False
        return _class_attr_constant(node, res, bud)

    if isinstance(node, ast.NamedExpr):
        # `assert (ok := True)` — the assert sees the assignment expression's VALUE
        return _fold(node.value, res, bud)

    if isinstance(node, ast.Lambda):
        # `assert (lambda: False)` — the OBJECT is always truthy; the body is never called. (A CALLED
        # lambda, `(lambda: False)()`, is an ast.Call and stays UNKNOWN: the result is not folded.)
        return True

    if isinstance(node, ast.IfExp):
        cv = _fold(node.test, res, bud)
        if cv is _UNKNOWN:
            return _UNKNOWN
        return _fold(node.body if cv else node.orelse, res, bud)

    if isinstance(node, ast.UnaryOp):
        v = _fold(node.operand, res, bud)
        if v is _UNKNOWN:
            return _UNKNOWN
        try:
            if isinstance(node.op, ast.Not):
                return not v
            if isinstance(node.op, ast.USub):
                return _guard(-v)
            if isinstance(node.op, ast.UAdd):
                return _guard(+v)
            if isinstance(node.op, ast.Invert):
                return _guard(~v)
        except _FoldRefused:
            raise
        except Exception:                       # noqa: BLE001 — the operand does not support it
            return _UNKNOWN
        return _UNKNOWN

    if isinstance(node, ast.BoolOp):
        vals = [_fold(v, res, bud) for v in node.values]
        if isinstance(node.op, ast.Or):
            if any(v is not _UNKNOWN and bool(v) for v in vals):
                return True
            if all(v is not _UNKNOWN and not v for v in vals):
                return False
            return _UNKNOWN
        if any(v is not _UNKNOWN and not v for v in vals):
            return False
        if all(v is not _UNKNOWN and bool(v) for v in vals):
            return True
        return _UNKNOWN

    if isinstance(node, ast.BinOp):
        lv = _fold(node.left, res, bud)
        if lv is _UNKNOWN:
            return _UNKNOWN
        rv = _fold(node.right, res, bud)
        if rv is _UNKNOWN:
            return _UNKNOWN
        return _binop(node.op, lv, rv)

    if isinstance(node, ast.Compare):
        # `a < b < c` is `a < b and b < c`. Folded link by link; a FALSE link ends the chain, because
        # Python short-circuits there and no later link — known or not — can change the answer.
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            r = _compare(left, op, right, res, bud)
            if r is _UNKNOWN:
                return _UNKNOWN
            if not r:
                return False
            left = right
        return True

    if isinstance(node, ast.Subscript):
        target = _fold(node.value, res, bud)
        if target is _UNKNOWN:
            return _UNKNOWN
        key = _fold(node.slice, res, bud)
        if key is _UNKNOWN:
            return _UNKNOWN
        try:
            return _guard(target[key])
        except _FoldRefused:
            raise
        except Exception:                       # noqa: BLE001 — not subscriptable / out of range
            return _UNKNOWN

    if isinstance(node, ast.Slice):
        parts: list[Any] = []
        for p in (node.lower, node.upper, node.step):
            if p is None:
                parts.append(None)
                continue
            v = _fold(p, res, bud)
            if v is _UNKNOWN or not (v is None or isinstance(v, int)):
                return _UNKNOWN
            parts.append(v)
        return slice(*parts)

    if isinstance(node, ast.JoinedStr):
        parts_s: list[str] = []
        total = 0
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                piece: Any = v.value
            elif isinstance(v, ast.FormattedValue):
                piece = _formatted(v, res, bud)
            else:
                return _UNKNOWN
            if piece is _UNKNOWN:
                return _UNKNOWN
            parts_s.append(piece)
            total += len(piece)
            if total > _FOLD_MAX_LEN:
                raise _FoldRefused("f-string too large")
        return "".join(parts_s)

    if isinstance(node, ast.FormattedValue):
        return _formatted(node, res, bud)

    if isinstance(node, ast.Call):
        return _call(node, res, bud)

    return _UNKNOWN


def _elements(elts: list[ast.expr], res: _Resolver, bud: _Budget) -> Any:
    """The values of a list/tuple/set display, expanding `*starred` sub-displays."""
    out: list[Any] = []
    for e in elts:
        if isinstance(e, ast.Starred):
            inner = _fold(e.value, res, bud)
            if inner is _UNKNOWN or not isinstance(inner, (list, tuple, set, frozenset, str, bytes, range)):
                return _UNKNOWN
            out.extend(inner)
        else:
            v = _fold(e, res, bud)
            if v is _UNKNOWN:
                return _UNKNOWN
            out.append(v)
        if len(out) > _FOLD_MAX_LEN:
            raise _FoldRefused("display too large")
    return out


def _binop(op: ast.operator, lv: Any, rv: Any) -> Any:
    """An operator applied to two folded constants. Every operation whose RESULT can be far bigger
    than its operands is bounded BEFORE the value is built."""
    fn = _BINOPS.get(type(op))
    if fn is None:
        return _UNKNOWN
    ln, rn = _sized(lv), _sized(rv)
    l_int = isinstance(lv, int) and not isinstance(lv, bool)
    r_int = isinstance(rv, int) and not isinstance(rv, bool)
    if isinstance(op, ast.Mult):
        if ln is not None and r_int and ln * max(rv, 0) > _FOLD_MAX_LEN:
            raise _FoldRefused("repetition too large")
        if rn is not None and l_int and rn * max(lv, 0) > _FOLD_MAX_LEN:
            raise _FoldRefused("repetition too large")
        if l_int and r_int and lv.bit_length() + rv.bit_length() > _FOLD_MAX_BITS:
            raise _FoldRefused("product too wide")
    elif isinstance(op, ast.Add) and ln is not None and rn is not None and ln + rn > _FOLD_MAX_LEN:
        raise _FoldRefused("concatenation too large")
    elif isinstance(op, ast.Pow) and l_int and r_int and rv > 0 and max(lv.bit_length(), 1) * rv > _FOLD_MAX_BITS:
        raise _FoldRefused("power too wide")
    elif isinstance(op, ast.LShift) and l_int and r_int and rv > 0 and lv.bit_length() + rv > _FOLD_MAX_BITS:
        raise _FoldRefused("shift too wide")
    elif isinstance(op, ast.Mod) and isinstance(lv, (str, bytes)):
        _guard_width(lv if isinstance(lv, str) else lv.decode("latin-1", "replace"))
    try:
        return _guard(fn(lv, rv))
    except _FoldRefused:
        raise
    except Exception:                           # noqa: BLE001 — a TypeError here means "not constant"
        return _UNKNOWN


def _compare(left: ast.AST, op: ast.cmpop, right: ast.AST, res: _Resolver, bud: _Budget) -> Any:
    # `x == x` on a pure name or attribute chain: identical source, always true, though neither side
    # folds to a value.
    if isinstance(op, (ast.Eq, ast.Is, ast.LtE, ast.GtE)) and _pure(left) and safe_dump(left) == safe_dump(right):
        return True
    if isinstance(op, (ast.NotEq, ast.IsNot, ast.Lt, ast.Gt)) and _pure(left) and safe_dump(left) == safe_dump(right):
        return False
    lv = _fold(left, res, bud)
    if lv is _UNKNOWN:
        return _UNKNOWN
    if isinstance(op, (ast.In, ast.NotIn)):
        return _membership(lv, op, right, res, bud)
    rv = _fold(right, res, bud)
    if rv is _UNKNOWN:
        return _UNKNOWN
    fn = _CMPOPS.get(type(op))
    if fn is None:
        return _UNKNOWN
    try:
        return bool(fn(lv, rv))
    except Exception:                           # noqa: BLE001 — uncomparable types
        return _UNKNOWN


def _membership(lv: Any, op: ast.cmpop, right: ast.AST, res: _Resolver, bud: _Budget) -> Any:
    """`x in [...]`. A container that folds WHOLE answers exactly. A container that does not fold can
    still answer `in` from one matching KNOWN element (it is definitely in) — but never `not in`,
    which needs every element. (Answering `not in` from partial knowledge, as this did before the
    fifth cycle, calls an honest assertion constant-true.)"""
    rv = _fold(right, res, bud)
    if rv is not _UNKNOWN:
        try:
            return (lv in rv) if isinstance(op, ast.In) else (lv not in rv)
        except Exception:                       # noqa: BLE001 — not a container
            return _UNKNOWN
    if not isinstance(right, (ast.List, ast.Tuple, ast.Set)):
        return _UNKNOWN
    for e in right.elts:
        if isinstance(e, ast.Starred):
            continue
        v = _fold(e, res, bud)
        try:
            if v is not _UNKNOWN and v == lv:
                return isinstance(op, ast.In)
        except Exception:                       # noqa: BLE001
            return _UNKNOWN
    return _UNKNOWN


def _formatted(node: ast.FormattedValue, res: _Resolver, bud: _Budget) -> Any:
    """One `{...}` of an f-string: its value, its `!conversion` and its `:format_spec`, all folded.
    The spec is itself a JoinedStr and may nest (`f'{1:{w}d}'`)."""
    v = _fold(node.value, res, bud)
    if v is _UNKNOWN:
        return _UNKNOWN
    if node.conversion == ord("r"):
        v = repr(v)
    elif node.conversion == ord("a"):
        v = ascii(v)
    elif node.conversion == ord("s"):
        v = str(v)
    elif node.conversion != -1:
        return _UNKNOWN
    spec = ""
    if node.format_spec is not None:
        spec = _fold(node.format_spec, res, bud)
        if spec is _UNKNOWN or not isinstance(spec, str):
            return _UNKNOWN
    _guard_width(spec)
    try:
        return _guard(format(v, spec))
    except _FoldRefused:
        raise
    except Exception:                           # noqa: BLE001 — a spec the value does not accept
        return _UNKNOWN


def _call(node: ast.Call, res: _Resolver, bud: _Budget) -> Any:
    """The fold STOPS at a call, except for the pure builtins below whose result is decided entirely
    by the constants handed to them. Every other call is the stated residual: not guessed."""
    if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
        return _UNKNOWN
    fc = res.canon(node.func)
    if fc == "isinstance" and len(node.args) == 2 and res.canon(node.args[1]) in ("object", "builtins.object"):
        return True
    if fc in ("any", "all") and len(node.args) == 1:
        if isinstance(node.args[0], (ast.List, ast.Tuple, ast.Set)) and not node.args[0].elts:
            return fc == "all"
        v = _fold(node.args[0], res, bud)
        if v is not _UNKNOWN and _sized(v) is not None:
            try:
                return any(v) if fc == "any" else all(v)
            except Exception:                   # noqa: BLE001
                return _UNKNOWN
        return _UNKNOWN
    if fc == "len" and len(node.args) == 1:
        arg = node.args[0]
        # the LENGTH of a display is known even when its elements are not: `len([f(), g()]) == 2`
        if isinstance(arg, (ast.List, ast.Tuple, ast.Set)) and not any(isinstance(e, ast.Starred) for e in arg.elts):
            return len(arg.elts)
        if isinstance(arg, ast.Dict) and all(k is not None for k in arg.keys):
            return len(arg.keys)
    if len(node.args) != 1:
        return _UNKNOWN
    v = _fold(node.args[0], res, bud)
    if v is _UNKNOWN:
        return _UNKNOWN
    if fc == "bool":
        return bool(v)
    if fc == "len":
        n = _sized(v)
        return n if n is not None else _UNKNOWN
    if fc in ("str", "repr"):
        try:
            return _guard(str(v) if fc == "str" else repr(v))
        except _FoldRefused:
            raise
        except Exception:                       # noqa: BLE001
            return _UNKNOWN
    return _UNKNOWN


_ATTR_STORE_BANS: frozenset[str] = frozenset({"setattr", "builtins.setattr", "delattr", "vars", "builtins.vars"})


def _class_attr_constant(node: ast.Attribute, res: _Resolver, bud: _Budget) -> Any:
    """`self.X` inside a method of a class pytest collects BY NAME, where `X` is bound EXACTLY ONCE, at
    class-body level, to a constant expression, and nothing anywhere in the file (or in any other
    file of the PR's input) can rebind it. Folds to that constant; everything else is _UNKNOWN.

    ELEVENTH CYCLE (verifier 10, V10-F7). `class TestAdd: expected = 3; def test_add(self): assert
    self.expected == 3` read PROVEN while its module-level twin read GAMED-SUSPECT, and no by-design
    row or NOT-COVERED line mentioned class attributes. The refusals, each a way the value could
    differ from the class-body binding:
      * the receiver is not the method's FIRST parameter (`self`), or the method is not a direct
        member of a class;
      * the class is not collectable by name (a `Test*` class or a unittest.TestCase subclass, not
        disabled) — a contract base's method runs through subclasses that may override the attribute;
      * the attribute is bound other than exactly once at class-body level (under an `if`, twice,
        by destructuring, by an annotation without a value);
      * the class-body value does not fold to a constant;
      * ANY store to `.X` anywhere in the file (`self.X = ...`, `cls.X = ...`, `TestAdd.X = ...`,
        `request.cls.X = ...`), an augmented assignment or `del` through it, or a `setattr` / `vars`
        / `__dict__` / `__setattr__` / `request.cls` / `request.instance` anywhere in the file;
      * a class in the SAME file that inherits this class and binds `X` (an override);
      * a store to `.X` in ANY OTHER file of the PR's input (`attrs_stored_elsewhere`).
    A subclass in a file the PR did not touch that overrides `X` is the stated limit: the base
    class's OWN collected test (`TestAdd::test_add`) is still constant-true, and that is what the
    receipt reports about that test, by its own id."""
    scope = res.scope
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) or not isinstance(node.value, ast.Name):
        return _UNKNOWN
    params = scope.args.posonlyargs + scope.args.args
    if not params or params[0].arg != node.value.id:
        return _UNKNOWN
    tree = res._tree
    cls = next((c for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and scope in c.body), None)
    if cls is None or _class_disabled(cls, res) or not (cls.name.startswith("Test") or _is_unittest_case(cls, res)):
        return _UNKNOWN
    attr = node.attr
    if attr in res.attrs_stored_elsewhere:
        return _UNKNOWN
    bindings = []
    for st in cls.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and st.targets[0].id == attr:
            bindings.append(st.value)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == attr:
            bindings.append(st.value)                 # None when there is no value: refused below
        elif isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.If, ast.For, ast.While, ast.Try, ast.With)):
            if any(isinstance(n, ast.Name) and n.id == attr and isinstance(n.ctx, ast.Store) for n in ast.walk(st)
                   if n is not st) and not (isinstance(st, (ast.Assign, ast.AnnAssign))):
                return _UNKNOWN                       # bound under a condition / loop / by destructuring
    if len(bindings) != 1 or bindings[0] is None:
        return _UNKNOWN
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == attr and not isinstance(n.ctx, ast.Load):
            return _UNKNOWN
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute) and n.target.attr == attr:
            return _UNKNOWN
        if isinstance(n, ast.Attribute) and n.attr in ("__dict__", "__setattr__", "cls", "instance") \
                and isinstance(n.value, ast.Name) and (n.attr in ("__dict__", "__setattr__") or n.value.id == "request"):
            return _UNKNOWN
        if isinstance(n, ast.Call):
            c = res.canon(n.func) or ""
            if c in _ATTR_STORE_BANS or c.endswith(".__setattr__"):
                return _UNKNOWN
        if isinstance(n, ast.ClassDef) and n is not cls and cls.name in _base_names(n, res):
            if any(isinstance(m, ast.Name) and m.id == attr and isinstance(m.ctx, ast.Store) for m in ast.walk(n)):
                return _UNKNOWN                       # a same-file subclass overrides it
    keep = res.scope
    res.scope = None                                  # the class-body expression resolves at module level
    try:
        return _fold(bindings[0], res, bud)
    finally:
        res.scope = keep


def known_true(node: ast.AST, res: _Resolver) -> bool:
    v = const_value(node, res)
    return v is not _UNKNOWN and bool(v)


def known_false(node: ast.AST, res: _Resolver) -> bool:
    v = const_value(node, res)
    return v is not _UNKNOWN and not bool(v)


# ----------------------------------------------------------------------------- collectability

def _is_test_func(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")


def _is_unittest_case(node: ast.ClassDef, res: _Resolver) -> bool:
    for b in node.bases:
        c = res.canon(b) or ""
        if c in ("unittest.TestCase", "unittest.case.TestCase", "TestCase") or c.endswith(".TestCase") or c.endswith("TestCase"):
            return True
    return False


def _class_disabled(node: ast.ClassDef, res: _Resolver) -> bool:
    for s in node.body:
        if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__test__" for t in s.targets) and known_false(s.value, res):
            return True
        if isinstance(s, ast.FunctionDef) and s.name == "__init__":
            return True
    return False


@dataclass(frozen=True)
class ClassRec:
    """One class declared at MODULE scope (or nested in another such class), as the collection
    question needs it: what it inherits, and whether pytest collects it in its own right."""
    name: str
    bases: tuple[str, ...]          # base names reduced to their LAST dotted segment
    collectable_here: bool          # Test*-named or a unittest.TestCase subclass, and not disabled
    disabled: bool                  # `__test__ = False`, or it defines __init__


def _base_names(node: ast.ClassDef, res: _Resolver) -> tuple[str, ...]:
    """A class's bases as BARE names. `from contract import AddContract` resolves through the
    resolver to `contract.AddContract`; the last segment is what the defining module called the
    class, so it is the segment both files agree on. Matching on a bare name can over-match two
    same-named classes in different files, and that direction is the safe one: an over-match
    means C2 declines to accuse, never that it accuses something it should not."""
    out = []
    for b in node.bases:
        name = (res.canon(b) or "")
        if not name:
            name = b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "")
        if name:
            out.append(name.rsplit(".", 1)[-1])
    return tuple(out)


def collection_reach(tree: ast.Module, res: _Resolver) -> tuple[list[ClassRec], dict[str, list[str]]]:
    """(every module-scope class, {test-named def -> the non-collectable classes that HOLD it}).

    TENTH CYCLE (verifier 9, V9-F1). `collectable()` answers "does THIS module collect this def",
    which is the wrong question for a `def test*` in a contract base class: pytest collects it
    through every `Test*` subclass that inherits it, and the subclass is usually in another file.
    This function supplies the two facts that turn that into a mechanical, deterministic decision —
    the inheritance edges, and which defs are waiting on one.

    A def that appears in NEITHER `collectable()`'s collected set NOR this holder map is
    structurally unreachable: nested in a function scope, behind a constant-false gate, or in a
    module or class switched off with `__test__ = False`. No subclass anywhere can rescue those,
    so they stay findings."""
    res.scope = None
    classes: list[ClassRec] = []
    held: dict[str, list[str]] = {}
    module_off = any(isinstance(s, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "__test__" for x in s.targets)
                     and known_false(s.value, res) for s in tree.body)
    disabled_attr: set[str] = set()
    for s in ast.walk(tree):
        if isinstance(s, ast.Assign) and known_false(s.value, res):
            for x in s.targets:
                if isinstance(x, ast.Attribute) and x.attr == "__test__" and isinstance(x.value, ast.Name):
                    disabled_attr.add(x.value.id)

    def walk(body) -> None:
        for s in body:
            if isinstance(s, ast.ClassDef):
                dis = _class_disabled(s, res) or s.name in disabled_attr
                coll = (s.name.startswith("Test") or _is_unittest_case(s, res)) and not dis
                classes.append(ClassRec(s.name, _base_names(s, res), coll, dis))
                if not dis:
                    if not coll:
                        for inner in s.body:
                            if _is_test_func(inner):
                                held.setdefault(inner.name, []).append(s.name)
                    walk(s.body)
            elif isinstance(s, ast.If):
                if known_false(s.test, res):
                    walk(s.orelse)
                elif known_true(s.test, res):
                    walk(s.body)
                else:
                    walk(s.body)
                    walk(s.orelse)
            elif isinstance(s, ast.Try):
                walk(s.body)
                for h in s.handlers:
                    walk(h.body)
                walk(s.orelse)
                walk(s.finalbody)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                walk(s.body)

    if not module_off:
        walk(tree.body)
    return classes, held


class InheritanceCorpus:
    """Which classes, across EVERY file in one PR's input, a collectable class inherits.

    Built once per `audit_diff` from the whole file set the caller supplied, because the question
    "is this base class collected somewhere" is not answerable from one file, and the tenth cycle's
    finding was C2 answering it anyway — with `never`."""

    def __init__(self) -> None:
        self._by_name: dict[str, list[ClassRec]] = {}
        self._collectable: list[ClassRec] = []
        self._where: dict[int, str] = {}          # id(ClassRec) -> the path it was declared in
        self._reach: set[str] | None = None
        self.modules = 0
        self.stored_attrs: set[str] = set()        # V10-F7: `.attr` stored anywhere in the input

    def add(self, tree: ast.Module, res: _Resolver, path: str = "") -> None:
        classes, _ = collection_reach(tree, res)
        self.modules += 1
        self._reach = None
        for n in ast.walk(tree):                      # V10-F7: attribute names this file STORES
            if isinstance(n, ast.Attribute) and not isinstance(n.ctx, ast.Load):
                self.stored_attrs.add(n.attr)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute):
                self.stored_attrs.add(n.target.attr)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr" and len(n.args) >= 2 \
                    and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
                self.stored_attrs.add(n.args[1].value)
        for rec in classes:
            self._by_name.setdefault(rec.name, []).append(rec)
            self._where[id(rec)] = path
            if rec.collectable_here:
                self._collectable.append(rec)

    def _closure(self) -> set[str]:
        """Every class name a collectable class reaches through its base chain, transitively."""
        if self._reach is not None:
            return self._reach
        seen: set[str] = set()
        stack = [b for rec in self._collectable for b in rec.bases]
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            for rec in self._by_name.get(name, ()):
                if rec.disabled:
                    continue          # a base switched off collects nothing through this edge
                stack.extend(rec.bases)
        self._reach = seen
        return seen

    def collected_through(self, holders: list[str]) -> str | None:
        """The name of a holder class that some collectable class in the input inherits, or None."""
        reach = self._closure()
        for h in holders:
            if h in reach:
                return h
        return None

    def collector_of(self, holder: str) -> tuple[str, str] | None:
        """(collectable class name, its path) for a class in the input that reaches `holder` through
        its base chain — the FACT the receipt states when it resolves a base class's test method.

        ELEVENTH CYCLE (verifier 10, V10-F2). With `collected_through` stubbed to None, every
        contract-base row still read PROVEN: the finding merely degraded to an OBSERVATION carrying a
        FALSE sentence ("C2 found no class in this PR's files that inherits it") while the subclass
        sat in the same input. A fix whose success leaves no trace on the receipt cannot be pinned,
        so the resolution is now STATED — which class, in which file — and the pin rows and the
        honest corpus assert that sentence rather than the verdict alone."""
        for rec in self._collectable:
            seen: set[str] = set()
            stack = list(rec.bases)
            while stack:
                name = stack.pop()
                if name in seen:
                    continue
                seen.add(name)
                if name == holder:
                    return rec.name, self._where.get(id(rec), "")
                for base in self._by_name.get(name, ()):
                    if not base.disabled:
                        stack.extend(base.bases)
        return None


def collectable(tree: ast.Module, res: _Resolver) -> tuple[set[str], set[str], set[str]]:
    """(collected ids, ids defined under a non-constant module-level condition, every test-named def anywhere).
    pytest defaults: python_functions=test*, python_classes=Test* (plus unittest.TestCase subclasses),
    `__test__ = False` disables a module/class/function; a def nested in a function is never collected."""
    res.scope = None
    collected: set[str] = set()
    conditional: set[str] = set()
    everywhere: set[str] = set()
    for n in ast.walk(tree):
        if _is_test_func(n):
            everywhere.add(n.name)
    module_off = any(isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__test__" for t in s.targets)
                     and known_false(s.value, res) for s in tree.body)
    disabled_attr: set[str] = set()
    for s in ast.walk(tree):
        if isinstance(s, ast.Assign) and known_false(s.value, res):
            for t in s.targets:
                if isinstance(t, ast.Attribute) and t.attr == "__test__" and isinstance(t.value, ast.Name):
                    disabled_attr.add(t.value.id)

    def walk(body, prefix: str, cond: bool):
        for s in body:
            if _is_test_func(s):
                (conditional if cond else collected).add(prefix + s.name)
            elif isinstance(s, ast.ClassDef):
                if (s.name.startswith("Test") or _is_unittest_case(s, res)) and not _class_disabled(s, res) and s.name not in disabled_attr:
                    walk(s.body, prefix + s.name + "::", cond)
            elif isinstance(s, ast.If):
                if known_false(s.test, res):
                    walk(s.orelse, prefix, cond)
                elif known_true(s.test, res):
                    walk(s.body, prefix, cond)
                else:
                    walk(s.body, prefix, True)
                    walk(s.orelse, prefix, True)
            elif isinstance(s, ast.Try):
                walk(s.body, prefix, cond)
                for h in s.handlers:
                    walk(h.body, prefix, True)
                walk(s.orelse, prefix, cond)
                walk(s.finalbody, prefix, cond)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                walk(s.body, prefix, cond)

    if not module_off:
        walk(tree.body, "", False)
    collected = {c for c in collected if c.split("::")[-1] not in disabled_attr}
    return collected, conditional, everywhere


# ----------------------------------------------------------------------------- the audit

def _lines_of(node: ast.AST) -> range:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", start)
    if start is None:
        return range(0)
    return range(start, (end or start) + 1)


def _is_assertion_like(n: ast.AST, res: _Resolver) -> bool:
    if isinstance(n, ast.Assert):
        return True
    if isinstance(n, ast.Call):
        if isinstance(n.func, ast.Attribute) and (n.func.attr.startswith("assert") or n.func.attr == "fail"):
            return True
        if (res.canon(n.func) or "") in _ASSERTION_CALL_CANONS:
            return True
    return False


class _Audit:
    def __init__(self, source: str, tree: ast.Module, *, is_conftest: bool, is_test_file: bool, added: set[int] | None,
                 deleted_lines: list[str], first_party: frozenset[str] = frozenset()):
        self.src = source
        self.src_lines = source.splitlines()
        self.tree = tree
        self.res = _Resolver(tree)
        self.is_conftest = is_conftest
        self.is_test_file = is_test_file
        self.added = added
        self.deleted_lines = deleted_lines
        # ELEVENTH CYCLE (V10-F1): what the infrastructure-body rule needs — the top-level module names
        # this PR ships (a patch aimed at one is a patch of the code under test) and this file's own
        # module-level defs (a same-file helper a hook calls is followed rather than left undecided).
        self.first_party = frozenset(first_party)
        self.helpers = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.findings: list[AstFinding] = []
        self.needed: dict[str, set[int]] = {}
        self._parents: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self._parents[id(child)] = parent
        self.deleted_strong = self._deleted_strong()

    # -- helpers
    def touched(self, node: ast.AST) -> bool:
        if self.added is None:
            return True
        return any(l in self.added for l in _lines_of(node))

    def snippet(self, node: ast.AST) -> str:
        ln = getattr(node, "lineno", None)
        if ln and 1 <= ln <= len(self.src_lines):
            return self.src_lines[ln - 1].strip()[:120]
        return type(node).__name__

    def add(self, kind: str, node: ast.AST, target: str | None, snippet: str | None = None) -> None:
        self.findings.append(AstFinding(kind, getattr(node, "lineno", 0) or 0, snippet or self.snippet(node), target))

    def decide_infra(self, body_owner: ast.AST, *, hook_name: str | None, is_fixture: bool) -> "infra_body.Decision":
        """The decidability rule (eleventh cycle, verifier 10 V10-F1): SILENCING / BENIGN / UNDECIDED."""
        return infra_body.decide(body_owner, hook_name=hook_name, is_fixture=is_fixture, canon=self.res.canon,
                                 const=lambda n: const_value(n, self.res), first_party=self.first_party,
                                 helpers=self.helpers)

    def add_infra(self, finding_kind: str, node: ast.AST, decision: "infra_body.Decision", what: str) -> None:
        """One receipt line per hook / fixture / plugins assignment, worded by the decision:
        SILENCING -> the finding kind with the proof; BENIGN / UNDECIDED -> `infra-observed`."""
        # The decision word comes FIRST on the line (the receipt renders the first 140 characters of a
        # snippet), then what was seen, then the C1 consequence — the same on every line.
        where = "conftest.py" if self.is_conftest else "a non-conftest file"
        c1 = ("C1 still refuses witnesses while this PR-authored infrastructure runs during the reverted phase "
              "(a safety refusal, not an accusation)")
        if decision.verdict == "silencing":
            self.add(finding_kind, node, None,
                     f"PROVEN outcome-affecting — {what} in {where}: {decision.why}. Also a C1 contamination")
        elif decision.verdict == "benign":
            self.add("infra-observed", node, None,
                     f"DECIDED harmless — {what} in {where}: {decision.why}; not a finding. {c1}")
        else:
            self.add("infra-observed", node, None,
                     f"NOT DECIDED from this file — {what} in {where}: {decision.why}. C2 draws no conclusion and makes "
                     f"no finding. {c1}")

    def enclosing_function(self, node: ast.AST) -> ast.AST | None:
        cur = self._parents.get(id(node))
        while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            cur = self._parents.get(id(cur))
        return cur if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)) else None

    def enclosing_test(self, node: ast.AST) -> str | None:
        func = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else self.enclosing_function(node)
        if func is None or not func.name.startswith("test"):
            return None
        cls = self._parents.get(id(func))
        return f"{cls.name}::{func.name}" if isinstance(cls, ast.ClassDef) else func.name

    def _set_scope(self, node: ast.AST) -> None:
        self.res.scope = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else self.enclosing_function(node)

    def _deleted_strong(self) -> dict[str, tuple[str, str]]:
        out: dict[str, tuple[str, str]] = {}
        lines = list(self.deleted_lines)
        i = 0
        while i < len(lines):
            joined = None
            for span in range(1, 6):
                chunk = "\n".join(lines[i:i + span])
                joined = parse_or_none(_dedent(chunk)) or parse_or_none(repair_for_parse(_dedent(chunk)))
                if joined is not None:
                    break
            if joined is None:
                i += 1
                continue
            for node in ast.walk(joined):
                try:
                    keys = self._strong_keys(node)
                except RecursionError:
                    keys = []
                for key, kind in keys:
                    out.setdefault(key, (kind, lines[i].strip()))
            i += 1
        return out

    def _strong_keys(self, node: ast.AST) -> list[tuple[str, str]]:
        res = self.res
        out = []
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 \
                and isinstance(node.test.ops[0], (ast.Eq, ast.Is)):
            for side in (node.test.left, node.test.comparators[0]):
                if not isinstance(side, ast.Constant):
                    out.append((safe_dump(side), "eq"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in _STRONG_UNITTEST and node.args:
                for a in node.args[:2]:
                    if not isinstance(a, ast.Constant):
                        out.append((safe_dump(a), "eq"))
            if attr in _STRONG_MOCK:
                out.append((safe_dump(node.func.value), "mock"))
            fc = res.canon(node.func)
            if fc in ("pytest.raises", "raises") and node.args:
                exc = res.canon(node.args[0]) or safe_dump(node.args[0])
                has_match = any(k.arg == "match" for k in node.keywords)
                out.append((f"raises:{exc}", "raises-match" if has_match else "raises"))
        return out

    def _subdumps(self, node: ast.AST) -> set[str]:
        return {safe_dump(n) for n in ast.walk(node)}

    # -- passes
    def run(self) -> None:
        for node in ast.walk(self.tree):
            self._set_scope(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit_function(node)
            elif isinstance(node, ast.ClassDef):
                self.visit_class(node)
            elif isinstance(node, ast.Call):
                self.visit_call(node)
            elif isinstance(node, ast.Raise):
                self.visit_raise(node)
            elif isinstance(node, ast.Assign):
                self.visit_assign(node)
            elif isinstance(node, ast.Assert):
                self.visit_assert(node)
            elif isinstance(node, (ast.If, ast.While)):
                self.visit_gate(node)
            elif isinstance(node, ast.Try):
                self.visit_try(node)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                self.visit_with(node)
            elif isinstance(node, ast.Return):
                self.visit_return(node)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.visit_string(node)
            elif isinstance(node, ast.Attribute):
                self.visit_attribute(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)) and self.is_conftest:
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(m.startswith("_pytest") for m in mods) and self.touched(node):
                    self.add("conftest-internal-import", node, None)

    def visit_function(self, node) -> None:
        target = self.enclosing_test(node) if node.name.startswith("test") else None
        if node.name.startswith("test") and self.touched(node):
            t = self.enclosing_test(node)
            if t:
                span = set(_lines_of(node))
                self.needed[t] = span if self.added is None else (span & self.added)
        for d in node.decorator_list:
            self.visit_decorator(d, target or (node.name if node.name.startswith("test") else None), node)
        if _TEST_INFRA_HOOK_RE.match(node.name) and self.touched(node):
            # FOURTH CYCLE: not conftest-only any more. pytest calls a `pytest_*` hook from ANY module it
            # has loaded — a test module registered with `-p <mod>`, listed in `pytest_plugins`, named in
            # PYTEST_PLUGINS or declared as a pytest11 entry point. Verifier 3's EN3 forged a C1 witness
            # with exactly that: a pytest_runtest_call in a kept test file, whose origin IS the test's own
            # module. The file is never reverted, so the hook runs during the reverted phase.
            if node.name in PARAMETRIZATION_ONLY_HOOKS:
                self.add("conftest-parametrize-hook" if self.is_conftest else "plugin-parametrize-hook", node, None,
                         f"def {node.name}(...) — parametrization/reporting only; noted, not a finding, not a C1 contamination")
            else:
                # ELEVENTH CYCLE (verifier 10, V10-F1): the hook's BODY decides. A `pytest_configure` that only
                # registers markers is not a finding; a body this rule cannot read is an observation that
                # says so; a body that provably silences, drops or decides an outcome is the finding it was.
                self.add_infra("conftest-hook" if self.is_conftest else "plugin-hook", node,
                               self.decide_infra(node, hook_name=node.name, is_fixture=False),
                               f"def {node.name}(...)" + ("" if self.is_conftest else
                                                          " (run for every test once this module is loaded as a plugin; never reverted)"))

    def visit_class(self, node: ast.ClassDef) -> None:
        for d in node.decorator_list:
            self.visit_decorator(d, None, node)

    def visit_decorator(self, d: ast.AST, target: str | None, owner: ast.AST) -> None:
        if not self.touched(d):
            return
        c = self.res.canon(d) or ""
        for prefix, kind in _MARK_DECORATORS:
            if c == prefix or c.startswith(prefix + "(") or c.startswith(prefix + "."):
                self.add(kind if not self.is_conftest else "conftest-mark-skip", d, target)
                return
        if c in _UNITTEST_DECORATORS:
            self.add("unittest-skip", d, target)
            return
        if c.startswith("pytest.mark.parametrize") and isinstance(d, ast.Call):
            args = list(d.args) + [k.value for k in d.keywords if k.arg == "argvalues"]
            if len(args) >= 2 and isinstance(args[1], (ast.List, ast.Tuple)) and not args[1].elts:
                self.add("empty-parametrize", d, target)
                return
        if (c.startswith("pytest.fixture") or c.startswith("pytest.yield_fixture")) and isinstance(d, ast.Call):
            for k in d.keywords:
                if k.arg == "autouse" and known_true(k.value, self.res):
                    # ELEVENTH CYCLE (verifier 10, V10-F1): the fixture's BODY decides. Seeding `random`,
                    # clearing a cache, pytest-django's `def _db(db): pass` are DECIDED harmless; a patch of
                    # the code under test is PROVEN; a call this file cannot see is UNDECIDED and observed.
                    self.add_infra("conftest-autouse" if self.is_conftest else "plugin-autouse", d,
                                   self.decide_infra(owner, hook_name=None, is_fixture=True),
                                   f"autouse fixture `{getattr(owner, 'name', '?')}` (runs for every test in scope, unasked)")
        if c.endswith("<dynamic>") and is_sensitive(c):
            self.add("dynamic-code", d, target)

    def _eval_arg_is_dynamic(self, arg: ast.AST | None) -> bool:
        """`eval`/`compile` on a string the audit cannot read: a constant carrying an escape token, or a
        string built at runtime (concatenation, f-string, str-method chains, chr/bytes). `eval(repr(x))`,
        `eval(name)` and `eval(str(x))` are honest idioms and are not flagged."""
        if arg is None:
            return False
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return any(tok in arg.value for tok in _RUNNER_ESCAPE_TOKENS) or any(w in arg.value for w in ("_exit", "exit(", "import "))
        if isinstance(arg, (ast.BinOp, ast.JoinedStr)):
            return True
        if isinstance(arg, ast.Call):
            fc = self.res.canon(arg.func) or ""
            if isinstance(arg.func, ast.Attribute) and arg.func.attr in _STRING_BUILDERS:
                return True
            if fc in ("chr", "builtins.chr", "bytes", "builtins.bytes", "bytearray", "str.join", "codecs.decode", "base64.b64decode",
                      "binascii.unhexlify", "bytes.fromhex"):
                return True
            return False
        if isinstance(arg, ast.Subscript):
            return True
        return False

    def visit_call(self, node: ast.Call) -> None:
        c = self.res.canon(node.func) or ""
        target = self.enclosing_test(node)
        if not self.touched(node):
            return
        if c in _SKIP_CALLS:
            kind = _SKIP_CALLS[c]
            self.add("conftest-mark-skip" if self.is_conftest else kind, node, target)
            return
        if c.startswith("pytest.mark.") and any(c.startswith(p) for p, _ in _MARK_DECORATORS) and not isinstance(
                self._parents.get(id(node)), (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parent = self._parents.get(id(node))
            if isinstance(parent, ast.Assign):
                names = [t.id for t in parent.targets if isinstance(t, ast.Name)]
                if names and any(self._is_test_name(n) for n in names):
                    self.add("mark-skip" if "skip" in c else "mark-xfail", node, names[0])
                    return
            if self.is_conftest:
                self.add("conftest-mark-skip", node, None)
                return
        if isinstance(node.func, ast.Attribute) and node.func.attr in _SKIP_METHODS:
            self.add("unittest-skip", node, target)
            return
        if c.endswith(".skipTest"):
            self.add("unittest-skip", node, target)
            return
        if c in _RUNNER_ESCAPE_CALLS or any(c == m or c.startswith(m + ".") for m in _RUNNER_ESCAPE_MODULES):
            self.add("runner-escape", node, target)
            return
        if c in ("exec", "builtins.exec"):
            self.add("dynamic-code", node, target, f"{self.snippet(node)} — exec() in test code")
            return
        if c in ("eval", "compile", "builtins.eval", "builtins.compile") and node.args and self._eval_arg_is_dynamic(node.args[0]):
            self.add("dynamic-code", node, target, f"{self.snippet(node)} — {c.split('.')[-1]}() of a string the audit cannot read")
            return
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_marker" and self.is_conftest:
            arg = node.args[0] if node.args else None
            ac = self.res.canon(arg) or ""
            s = self.res.const_str(arg)
            if any(ac.startswith(p) for p, _ in _MARK_DECORATORS) or (s or "") in ("skip", "skipif", "xfail"):
                self.add("conftest-add-marker", node, None)
                return
        if c == "setattr" and node.args and self.res.canon(node.args[0]) == "<module>" and self.is_conftest:
            self.add("conftest-dynamic-attr", node, None)
            return
        if (c.endswith(".<dynamic>") or c.startswith("<dynamic")) and is_sensitive(c):
            self.add("dynamic-code", node, target, f"{self.snippet(node)} — a dynamic attribute of {c.split('.')[0]}")
            return
        if c == "pytest.raises" or c == "raises":
            self._check_raises(node, target)
        if isinstance(node.func, ast.Attribute) and node.func.attr in _WEAK_MOCK:
            key = safe_dump(node.func.value)
            if key in self.deleted_strong and self.deleted_strong[key][0] == "mock":
                self.add("assertion-loosened", node, target, f"{self.deleted_strong[key][1]}  ->  {self.snippet(node)}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in _WEAK_UNITTEST | _STRONG_UNITTEST:
            self._check_unittest_assert(node, target)

    def _check_raises(self, node: ast.Call, target: str | None) -> None:
        if not node.args:
            return
        exc = self.res.canon(node.args[0]) or safe_dump(node.args[0])
        has_match = any(k.arg == "match" for k in node.keywords)
        if exc in _BROAD_EXC and any(k.startswith("raises:") and k != f"raises:{exc}" for k in self.deleted_strong):
            self.add("raises-widened", node, target, f"pytest.raises({exc}) replaces a narrower pytest.raises")
        elif not has_match and self.deleted_strong.get(f"raises:{exc}", ("",))[0] == "raises-match":
            self.add("raises-widened", node, target, f"pytest.raises({exc}) lost its match=")

    def _check_unittest_assert(self, node: ast.Call, target: str | None) -> None:
        attr = node.func.attr  # type: ignore[union-attr]
        if attr in _WEAK_UNITTEST:
            for a in node.args:
                if safe_dump(a) in self.deleted_strong and self.deleted_strong[safe_dump(a)][0] == "eq":
                    self.add("assertion-loosened", node, target, f"{self.deleted_strong[safe_dump(a)][1]}  ->  {self.snippet(node)}")
                    break
            if attr in ("assertTrue", "assertFalse") and node.args and (known_true if attr == "assertTrue" else known_false)(node.args[0], self.res):
                self.add("constant-true", node, target)
            elif attr == "assertIsNotNone" and node.args and const_value(node.args[0], self.res) not in (_UNKNOWN, None):
                self.add("constant-true", node, target)
        elif attr in ("assertEqual", "assertEquals", "assertIs") and len(node.args) >= 2:
            a, b = node.args[0], node.args[1]
            va, vb = const_value(a, self.res), const_value(b, self.res)
            if (va is not _UNKNOWN and vb is not _UNKNOWN and va == vb) or (_pure(a) and safe_dump(a) == safe_dump(b)):
                self.add("constant-true", node, target)

    def visit_raise(self, node: ast.Raise) -> None:
        if not self.touched(node) or node.exc is None:
            return
        c = self.res.canon(node.exc) or ""
        target = self.enclosing_test(node)
        if c in _SKIP_CALLS or c in ("pytest.skip.Exception", "unittest.SkipTest"):
            self.add("conftest-mark-skip" if self.is_conftest else _SKIP_CALLS.get(c, "skip-call"), node, target)
        elif c in _RUNNER_ESCAPE_RAISES:
            self.add("runner-escape", node, target)

    def _is_test_name(self, name: str) -> bool:
        return name.startswith("test")

    def visit_assign(self, node: ast.Assign) -> None:
        if not self.touched(node):
            return
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "pytestmark" in names:
            marks = self._mark_canons(node.value)
            if marks:
                self.add("conftest-mark-skip" if self.is_conftest else "pytestmark", node, None,
                         f"pytestmark = {', '.join(marks)} — silences every test in the file")
            return
        hooks = [n for n in names if _TEST_INFRA_HOOK_RE.match(n) and n != "pytest_plugins"]
        if hooks:
            if all(h in PARAMETRIZATION_ONLY_HOOKS for h in hooks):
                self.add("conftest-parametrize-hook" if self.is_conftest else "plugin-parametrize-hook", node, None,
                         f"{hooks[0]} = ... — parametrization/reporting only; noted")
            else:
                # ELEVENTH CYCLE: a hook bound by assignment is decided by the BODY it is bound to — a
                # lambda, or a same-file def named on the right-hand side; anything else is undecided.
                owner: ast.AST = node.value
                if isinstance(owner, ast.Name) and owner.id in self.helpers:
                    owner = self.helpers[owner.id]
                self.add_infra("conftest-hook" if self.is_conftest else "plugin-hook", node,
                               self.decide_infra(owner, hook_name=hooks[0], is_fixture=False),
                               f"{hooks[0]} = ... (a pytest hook bound by assignment)")
            return
        if "pytest_plugins" in names:
            # ELEVENTH CYCLE: the modules it names are outside this file, so their hooks cannot be read
            # here — UNDECIDED, observed, never accused on shape. Still a C1 contamination.
            mods = const_value(node.value, self.res)
            shown = f" naming {mods!r}" if isinstance(mods, (str, list, tuple)) else ""
            self.add_infra("conftest-plugins" if self.is_conftest else "plugin-plugins", node,
                           infra_body.Decision("undecided", [f"`pytest_plugins`{shown} loads modules whose bodies are not in "
                                                             f"this file; whatever they hook runs on every tree"]),
                           "pytest_plugins = ...")
            return
        if self.is_conftest:
            if any(n in ("collect_ignore", "collect_ignore_glob") for n in names):
                self.add("conftest-collect-ignore", node, None)
                return
        for t in node.targets:
            if isinstance(t, ast.Attribute) and t.attr == "__test__" and known_false(node.value, self.res):
                nm = t.value.id if isinstance(t.value, ast.Name) else self.snippet(node)
                self.add("test-decollected", node, nm if self._is_test_name(nm) else None, f"{nm}.__test__ = False")
            if isinstance(t, ast.Name) and t.id == "__test__" and known_false(node.value, self.res):
                self.add("test-decollected", node, None, "__test__ = False — nothing in this module is collected")
            if isinstance(t, ast.Subscript) and self.res.canon(t.value) == "<module>" and self.is_conftest:
                self.add("conftest-dynamic-attr", node, None)

    def _mark_canons(self, value: ast.AST) -> list[str]:
        out = []
        for n in ast.walk(value):
            c = self.res.canon(n) or ""
            if any(c == p or c.startswith(p + ".") for p, _ in _MARK_DECORATORS):
                out.append(c)
        return sorted(set(out))

    def visit_assert(self, node: ast.Assert) -> None:
        if not self.touched(node):
            return
        target = self.enclosing_test(node)
        if known_true(node.test, self.res):
            self.add("constant-true", node, target)
        subs = self._subdumps(node.test)
        strong_on_x = set()
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], (ast.Eq, ast.Is)):
            strong_on_x = {safe_dump(t.left), safe_dump(t.comparators[0])}
        if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And):
            for v in t.values:
                if isinstance(v, ast.Compare) and len(v.ops) == 1 and isinstance(v.ops[0], (ast.Eq, ast.Is)):
                    strong_on_x |= {safe_dump(v.left), safe_dump(v.comparators[0])}
        for key, (kind, text) in self.deleted_strong.items():
            if kind == "eq" and key in subs and key not in strong_on_x:
                self.add("assertion-loosened", node, target, f"{text}  ->  {self.snippet(node)}")
                break
        cur = self._parents.get(id(node))
        while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            if isinstance(cur, ast.If) and const_value(cur.test, self.res) is _UNKNOWN and self.touched(cur):
                mine = safe_dump(node.test)
                for line in self.deleted_lines:
                    d = parse_or_none(_dedent(line))
                    if d is None:
                        continue
                    if any(isinstance(n, ast.Assert) and safe_dump(n.test) == mine for n in ast.walk(d)):
                        self.add("assertion-moved-under-condition", node, target,
                                 f"{line.strip()}  ->  under `if {self.snippet(cur)[3:60]}`")
                        cur = None
                        break
                if cur is None:
                    break
            cur = self._parents.get(id(cur))

    def visit_gate(self, node) -> None:
        if not self.touched(node):
            return
        if known_false(node.test, self.res):
            has_test_content = any(isinstance(n, ast.Assert) or _is_test_func(n) for n in ast.walk(node))
            if has_test_content:
                self.add("dead-gate", node, self.enclosing_test(node))

    def _handler_reraises(self, h: ast.ExceptHandler) -> bool:
        for s in h.body:
            for n in ast.walk(s):
                if isinstance(n, ast.Raise):
                    return True
                if isinstance(n, ast.Call):
                    c = self.res.canon(n.func) or ""
                    if c in _RERAISE_CALLS or (isinstance(n.func, ast.Attribute) and n.func.attr == "fail"):
                        return True
                if isinstance(n, ast.Assert) and known_false(n.test, self.res):
                    return True
        return False

    def visit_try(self, node: ast.Try) -> None:
        if not any(isinstance(n, ast.Assert) for s in node.body for n in ast.walk(s)):
            return
        for h in node.handlers:
            hc = self.res.canon(h.type) if h.type is not None else None
            catches = h.type is None or hc in _SWALLOW_TYPES or (isinstance(h.type, ast.Tuple) and any(
                self.res.canon(e) in _SWALLOW_TYPES for e in h.type.elts))
            if catches and not self._handler_reraises(h) and (self.touched(node) or self.touched(h)):
                first_assert = next(n for s in node.body for n in ast.walk(s) if isinstance(n, ast.Assert))
                self.add("assertion-swallowed", first_assert, self.enclosing_test(node),
                         f"{self.snippet(first_assert)}  inside try/except {hc or 'bare'} that does not re-raise")
                return

    def visit_with(self, node) -> None:
        if not self.touched(node):
            return
        for item in node.items:
            ce = item.context_expr
            c = self.res.canon(ce.func if isinstance(ce, ast.Call) else ce) or ""
            if c in ("contextlib.suppress", "suppress") and isinstance(ce, ast.Call):
                if any((self.res.canon(a) or "") in _SWALLOW_TYPES for a in ce.args) and any(
                        isinstance(n, ast.Assert) for s in node.body for n in ast.walk(s)):
                    first_assert = next(n for s in node.body for n in ast.walk(s) if isinstance(n, ast.Assert))
                    self.add("assertion-swallowed", first_assert, self.enclosing_test(node),
                             f"{self.snippet(first_assert)}  inside contextlib.suppress(...) that swallows AssertionError")

    def visit_return(self, node: ast.Return) -> None:
        if not self.touched(node):
            return
        func = self.enclosing_function(node)
        if func is None or not func.name.startswith("test"):
            return
        if node.value is not None and _is_assertion_like(node.value, self.res):
            return                                   # `return self.assertEqual(...)` is the assertion
        first_assert_line = None
        for n in ast.walk(func):
            if _is_assertion_like(n, self.res) and (first_assert_line is None or n.lineno < first_assert_line):
                first_assert_line = n.lineno
        if first_assert_line is None or node.lineno < first_assert_line:
            self.add("early-return", node, self.enclosing_test(node),
                     f"return before any assertion in {func.name} — the test body is silenced")

    def visit_string(self, node: ast.Constant) -> None:
        if not self.touched(node):
            return
        v = node.value
        if any(tok in v for tok in _RUNNER_ESCAPE_TOKENS):
            parent = self._parents.get(id(node))
            if isinstance(parent, ast.Expr):
                return
            self.add("runner-escape", node, self.enclosing_test(node), f"string token {v[:60]!r} — a report path / runner option in test code")

    def visit_attribute(self, node: ast.Attribute) -> None:
        if not self.touched(node):
            return
        c = self.res.canon(node) or ""
        if node.attr in ("xmlpath", "_xml"):
            self.add("runner-escape", node, self.enclosing_test(node), f"{self.snippet(node)} — reaches for the report writer")
            return
        if c == "sys.argv" and self.is_test_file:
            parent = self._parents.get(id(node))
            if isinstance(parent, (ast.Assign, ast.AugAssign)) and getattr(parent, "targets", [parent]) and node in getattr(parent, "targets", [getattr(parent, "target", None)]):
                return
            self.add("runner-escape", node, self.enclosing_test(node),
                     f"{self.snippet(node)} — reads sys.argv in test code (the report path is a runner argument)")


def repair_for_parse(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        code = line.split("#", 1)[0].rstrip()
        if not code.endswith(":"):
            continue
        ind = len(line) - len(line.lstrip())
        nxt = next((l for l in lines[i + 1:] if l.strip()), None)
        if nxt is None or (len(nxt) - len(nxt.lstrip())) <= ind:
            out.append(" " * (ind + 4) + "pass")
    return "\n".join(out) + "\n"


def _dedent(text: str) -> str:
    lines = text.splitlines()
    ind = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[ind:] for l in lines)


def audit_python(source: str, *, is_conftest: bool, added_linenos: set[int] | None, before_source: str | None,
                 deleted_lines: list[str], is_new_file: bool, is_test_file: bool = True,
                 inheritance: "InheritanceCorpus | None" = None,
                 known_modules: "frozenset[str] | None" = None) -> tuple[list[AstFinding], dict[str, set[int]], list[str]]:
    """(findings, {test the diff touched: its added line numbers}, notes). Never raises: unparsable
    input or a pathological expression returns ([], {}, [note]) and the caller falls back to the regex tier."""
    tree = parse_or_none(source)
    notes: list[str] = []
    if tree is None:
        return [], {}, ["AST: NEW side does not parse — regex tier only"]
    try:
        a = _Audit(source, tree, is_conftest=is_conftest, is_test_file=is_test_file, added=added_linenos, deleted_lines=deleted_lines,
                   first_party=frozenset(known_modules or ()))
        stored = getattr(inheritance, "stored_attrs", None)
        if stored:
            a.res.attrs_stored_elsewhere = frozenset(stored)
        a.run()
    except RecursionError:
        return [], {}, ["AST: recursion limit reached on a pathological expression — regex tier only"]
    findings = list(a.findings)
    if not is_conftest:
        try:
            after_ids, after_cond, after_defs = collectable(tree, a.res)
        except RecursionError:
            after_ids, after_cond, after_defs = set(), set(), set()
            notes.append("AST: collectability not computed (recursion limit)")
        after_names = {c.split("::")[-1] for c in after_ids | after_cond}
        if before_source is not None:
            btree = parse_or_none(before_source)
            if btree is None:
                notes.append("AST: OLD side does not parse — collectability not compared")
            else:
                try:
                    before_ids, _, _ = collectable(btree, _Resolver(btree))
                except RecursionError:
                    before_ids = set()
                for lost in sorted(before_ids - after_ids):
                    name = lost.split("::")[-1]
                    if name in after_defs and name not in after_names:
                        findings.append(AstFinding("test-decollected", _def_line(tree, name), f"{lost} is defined but no longer collected", lost))
        elif is_new_file:
            # TENTH CYCLE (verifier 9, V9-F1). A `def test*` this module does not collect is not
            # therefore never collected: a contract base class is collected through its Test*
            # SUBCLASSES, which usually live in another file. Three answers, and the check now
            # gives whichever one it can actually stand behind.
            try:
                held: dict[str, list[str]] | None
                _, held = collection_reach(tree, a.res)
            except RecursionError:
                held = None
                notes.append("AST: inheritance not computed (recursion limit) — no collection finding is made "
                             "on this file, and this note is why")
            for name in sorted(after_defs):
                if name in after_names:
                    continue
                if held is None:
                    continue
                holders = held.get(name)
                if not holders:
                    # structurally unreachable: nested in a function scope, behind a constant-false
                    # gate, or in a module/class switched off. No subclass anywhere rescues these.
                    findings.append(AstFinding("test-not-collectable", _def_line(tree, name),
                                               f"def {name} is nested inside a function, or in a module or class "
                                               f"switched off with `__test__ = False` — no class anywhere can "
                                               f"collect it", name))
                    continue
                via = inheritance.collected_through(holders) if inheritance is not None else None
                if via is not None:
                    # A collectable class in this PR inherits it: pytest collects it THERE. Stated on
                    # the receipt as an OBSERVATION naming the class and its file (V10-F2), so the
                    # resolution is a fact a reader can check and a pin row can assert — a `continue`
                    # here left no trace, and a fix that leaves no trace cannot be proven alive.
                    who = inheritance.collector_of(via)
                    collector = f"`{who[0]}` in {who[1]}" if who and who[1] else (f"`{who[0]}`" if who else "a subclass")
                    findings.append(AstFinding(
                        "test-collection-resolved", _def_line(tree, name),
                        f"def {name} is in `{via}`, which pytest does not collect by name — collected through "
                        f"{collector}, a class in this PR's files that inherits it; pytest runs it there", name))
                    continue
                findings.append(AstFinding(
                    "test-collection-unresolved", _def_line(tree, name),
                    f"def {name} is in `{holders[0]}`, which pytest does not collect by name — C2 found no "
                    f"class in this PR's files that inherits it, and it cannot see files this PR did not "
                    f"change, so it does not conclude whether the test runs. If `{holders[0]}` is a contract "
                    f"base collected through a subclass elsewhere, nothing is wrong here", name))
        for cond in sorted(after_cond):
            if added_linenos is None or _def_line(tree, cond.split("::")[-1]) in added_linenos:
                findings.append(AstFinding("test-under-condition", _def_line(tree, cond.split("::")[-1]),
                                           f"{cond} is defined under a non-constant module-level condition", cond))
    seen = set()
    out = []
    for f in findings:
        k = (f.kind, f.lineno, f.target)
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    return out, a.needed, notes


def _def_line(tree: ast.Module, name: str) -> int:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno
    return 0
