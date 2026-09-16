"""C1 — red-on-revert, as a pure comparison of two test-result snapshots.

The caller ran the PR's new/changed tests twice: WITH the PR's non-test diff applied, and with it
REVERTED onto the base. This module decides (design rulings of 2026-09-05):

  PROVEN         at least one test that PASSES with the change EXECUTED and FAILED BY ASSERTION, in
                 the CALL phase, on the reverted tree (and, if a rerun was supplied, reproduced there).
  UNPROVEN       reason "compile"          the reverted tree raised instead of asserting: ImportError /
                                           ModuleNotFoundError / SyntaxError at import, or any
                                           non-assertion exception raised in the test body
                        "collection"       the test did not execute on the reverted tree: collection
                                           failure, fixture/setup/teardown error, or a skip
                        "flaky"            red on the first reverted run, not on the rerun
                        "green-on-revert"  the test passes on both trees — the canonical fake green
  (GAMED_SUSPECT  RETIRED for C1 on 2026-09-15, the owner's decision. It was C1's only accusatory
                 verdict and could be reached only through C2's gaming markers -- the surface the
                 field sweep of 1,079 merged PRs measured at 6.58% false-accusation rate and 12.3%
                 accusatory precision, against gates of <3% and >=85%. `verdict.ALLOWED_STATES["C1"]`
                 now REFUSES it, so no caller can mint one. A marker aimed at a test C1 needed is
                 recorded as an OBSERVATION and the verdict is the honest UNPROVEN reason that test
                 gave. Historically it meant: a skip / xfail / constant-true marker the PR's own diff
                 added (C2's finding, passed in as `gaming_markers`) was aimed at a test C1 needed AND
                 that test produced no assertion red; or a `runner-escape` marker (os._exit, sys.exit,
                 a report-path token) was aimed at a WITNESS — a test that can end the runner or write
                 its report cannot be trusted to have gone red. Both now read UNPROVEN-contaminated.)
  UNPROVEN       reason "contaminated"  (owner ruling v2) the PR adds or modifies test infrastructure
                 that can affect outcomes (a conftest hook, an autouse fixture, a plugin) — that code
                 is never reverted, so it RUNS during the reverted phase; every witness is refused for
                 safety and the receipt names the file. A safety refusal, not a finding of intent.
  CRASHED        (guard disarmed) C2's gaming markers could not be computed (`gaming_markers_error`)
                 and a witness is present: C1 cannot certify a witness without its marker source.

DESIGN INVARIANT (ledgered in checks/README.md): A C1 witness is trusted only when its assertion
failure originates in the test's own call on a tree whose non-test, non-infrastructure code is the
only thing reverted; PR-authored infrastructure running during the reverted phase contaminates the
witness. Frame attribution enforces the first half: a dict result may carry `origin` (the raising
frame, `path:line`); an origin in a conftest.py, a plugin package or pytest's own internals is not
the test's own assertion.
  NOT_RUN        nothing to compare (no results; or no test id present in both runs).
  CRASHED        (via the runner) any exception here: an unknown status word, a phase that is not
                 in the vocabulary, a non-string test id, two ids that normalise to one.

Per-test result vocabulary (checks/README.md): a plain string "pass" | "fail" | "skip" | "error"
(EXACT lowercase words — anything else was not produced by an adapter and is CRASHED), where
"fail" MEANS executed-and-failed-by-assertion and "error" MEANS any non-assertion exception in any
phase; or a dict {"status", "phase"?, "type"?, "message"?}. The dict form must carry POSITIVE
evidence to count as an assertion red: `type` exactly one of the assertion types, or (with no type)
a message whose first token is an assertion shape. A dict `fail` in phase setup/teardown/
collection/import, a `fail` with a non-assertion type, a `[XPASS(strict)]` failure, or a bare dict
`fail` with neither type nor message is reclassified and the evidence says so.

Two optional keys adopted from the bench lane (owner's ruling): `test_results_without_change_rerun`
and `test_failure_kinds_without_change` (see README). `execution_notes` lets the runner state on the
receipt HOW the runs were isolated and reconciled. Tests red WITH the change are named in the
evidence and excluded from the witnesses. NEW module.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .verdict import MissingInput, Verdict

STATUSES = ("pass", "fail", "skip", "error")
PHASES = ("import", "collection", "setup", "call", "teardown")
WITNESS_PHASES = (None, "call")
# EXACT type names that mean "failed by assertion". No prefix / suffix matching: `AssertionErrorButNot`
# is not an assertion, `pkg.core.NotFound` is not, `StopIteration` is not.
ASSERTION_TYPES = frozenset({
    "AssertionError", "Failed", "JestAssertionError", "AssertionError [ERR_ASSERTION]", "ERR_ASSERTION",
})
# With no `type`, the message's FIRST token must be one of these shapes (pytest's short repr for a
# bare assert starts with `assert `; unittest/explicit raise with `AssertionError`; pytest.fail with
# `Failed:`; node/jest with `AssertionError [ERR_ASSERTION]` or `Error: expect(`).
_ASSERTION_MSG_RE = re.compile(r"^(assert(\s|$)|AssertionError(\s|:|$)|Failed:|AssertionError \[ERR_ASSERTION\]|Error: expect\()")
_XPASS_RE = re.compile(r"^\[XPASS\(strict\)\]")
IMPORT_TYPES = frozenset({"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError"})
MARKER_KINDS = ("skip", "xfail", "constant-true", "skipif", "todo", "only", "runner-escape", "loosened")
VOIDING_MARKERS = frozenset({"runner-escape"})
# FOURTEENTH CYCLE, addendum -- the owner's ruling of 2026-09-08, verbatim. Verifier 13's X7: a PR whose
# code and test are wrong IN AGREEMENT is byte-identical, in diff and in runtime record, to its honest
# twin; only the return value differs, and no deterministic check has an oracle for it. That is the
# theoretical floor of red-on-revert -- the same fundamental limit as behaviour-change-updates-test,
# already accepted and disclosed -- and it is DISCLOSED as a named residual beside every PROVEN, in
# these words, never chased. Held here as ONE constant so every surface quotes the same sentence.
X7_RESIDUAL = ("Corund proves a test ran and fails on revert; it does not verify the test asserts the CORRECT "
               "value, so a PR whose code and test are wrong in agreement is NOT DECIDED.")
# ---------------------------------------------------------------- frame attribution (ALLOWLIST)
# Owner's ruling of 2026-09-05 (fourth adversarial cycle): frame attribution is an ALLOWLIST, not a
# denylist. The five-path denylist it replaces (conftest.py, site-packages, dist-packages, _pytest,
# pluggy) trusted every origin it did not recognise — a plugin module at `myplugin/hooks.py`, a repo
# `plugins/` directory, and, worst, a `pytest_runtest_call` hook defined in the KEPT TEST FILE
# ITSELF, whose origin IS the test's own module. A witness is now trusted only when it AFFIRMS both:
#   origin  — the raising frame's file is the test id's OWN MODULE (exact, after normalisation), and
#   entry   — the FIRST frame's `def <name>(` is the test's OWN FUNCTION.
# Anything else — a helper module, the code under test, a plugin, a hook, a fixture, an absent frame,
# a path that merely ends with the module's path — is a non-witness: UNPROVEN, reason `contaminated`,
# worded as a safety refusal and never as an accusation.
_ORIGIN_LINECOL_RE = re.compile(r"(?::\d+)+$")
FRAME_WAIVED_PREFIX = "unavailable"
# A trailing `:line` or `:line:col`, as an adapter conventionally appends it to a path.
# ONE `:<digits>` group, stripped REPEATEDLY so every suffix boundary becomes a reading of its own
# (verifier 6, F1). The old regex took line and column together and so skipped the reading BETWEEN
# them — the file that actually raised — which made the published ambiguity refusal unreachable.
_ONE_GROUP_SUFFIX_RE = re.compile(r":\d+$")
# A MODULE whose own name ends in `:<digits>` or contains a backslash cannot be compared against an
# origin string without guessing which reading the adapter meant. See `own_frame_affirmed`.
_AMBIGUOUS_MODULE_RE = re.compile(r"(?::\d+$)|\\")


def normalize_path_only(path: str) -> str:
    """PATH-IDENTITY normalisation, and nothing else.

    SIXTH CYCLE (verifier 5, F1). The only rewrites permitted here are ones that cannot make two
    DIFFERENT tracked files compare equal: a leading `./` and a run of `/` both name the same file on
    every filesystem git supports. Stripping a `:<line>` suffix and rewriting `\\` to `/` do NOT have
    that property — `tests/test_a.py:9` and `tests/test_a.py` are two files git can track at once, and
    so are `tests\\test_x.py` and `tests/test_x.py` — so neither belongs in a function whose result is
    fed to an `==`. They now live in `origin_readings`, which returns BOTH readings instead of
    silently choosing one."""
    s = str(path).strip()
    while s.startswith("./"):
        s = s[2:]
    return re.sub(r"/{2,}", "/", s)


def origin_readings(origin: str) -> list[str]:
    """Every distinct repo path an adapter's `origin` string could be naming.

    `origin` is conventionally `path`, `path:line` or `path:line:col`. But `:` and `\\` are both legal
    characters in a POSIX filename, so `tests/test_a.py:9` denotes EITHER `tests/test_a.py` at line 9
    OR a file whose name literally ends `:9`, and `tests\\test_x.py` denotes EITHER that literal name
    OR `tests/test_x.py` written with Windows separators. The old `normalize_origin_path` picked one
    reading and returned it as a fact; this returns all of them, most literal first, and leaves the
    choice to a caller that can refuse when more than one could be true.

    SEVENTH CYCLE (verifier 6, F1). The suffix came off in ONE BITE — a line-and-column regex — so
    `tests/test_a.py:9:3` produced `{tests/test_a.py:9:3, tests/test_a.py}` and never the
    INTERMEDIATE `tests/test_a.py:9`, which is the file that actually raised when a file by that name
    is tracked. Only one reading was ever a tracked file, `len(real) > 1` was never true, and the
    ambiguity refusal checks/README.md promises could not fire at all — the guarantee was published
    and unreachable. Verifier 6's own V6N-04 is the proof of the shape: with THREE numeric groups
    (`mod:9:3:4`) the one-bite strip happens to leave an intermediate behind, and the refusal fires
    correctly there and only there.

    Readings are now enumerated PROGRESSIVELY: every suffix boundary is offered, one group at a time,
    most literal first. `a.py:9:3` yields `a.py:9:3`, `a.py:9`, `a.py`. The comparison stays a STRING
    comparison — `:0009` and `:9` are different filenames and must not be conflated (V6N-14)."""
    base = normalize_path_only(origin)
    out = [base]
    cur = base
    while True:
        nxt = _ONE_GROUP_SUFFIX_RE.sub("", cur)
        if not nxt or nxt == cur:
            break
        out.append(nxt)
        cur = nxt
    for r in list(out):                        # the Windows-separator reading of each
        if "\\" in r:
            alt = normalize_path_only(r.replace("\\", "/"))
            if alt and alt not in out:
                out.append(alt)
    return out


def normalize_origin_path(origin: str) -> str:
    """DEPRECATED, kept only so an out-of-tree caller does not break: the CONVENTIONAL reading of an
    origin. It is no longer used for the own-frame comparison, because a single reading cannot express
    the ambiguity that comparison has to refuse (verifier 5, F1)."""
    return origin_readings(origin)[-1]


def split_test_id(test_id: str) -> list[str]:
    """A pytest id split on the `::` separators that are OUTSIDE any `[parametrisation]` brackets.

    FIFTH CYCLE (verifier 4, V4-11). `rsplit("::", 1)` on `tests/test_x.py::test_a[a::b]` returns
    `b]` — a parametrisation VALUE, chosen by whoever wrote the parametrize list. That put the
    attacker on BOTH sides of the own-frame comparison: pick the param `a::pytest_runtest_call`, name
    the entry frame to match, and the allowlist affirms a hook. The same mis-split also FALSELY
    REFUSED the honest run of any parametrised test whose id happens to contain `::` (a file path, a
    module name, a URL, a `Class::method` string) — verifier 4's two honest controls.

    SIXTH CYCLE (verifier 5, F5). Bracket DEPTH was the wrong model. `depth = max(0, depth - 1)`
    clamped at zero, so a parametrisation value that closes a bracket it never opened dropped the
    scanner back to depth 0 and let a `::` INSIDE the value split the id: `test_a[]::_oracle[]` — the
    rendering of the single param value `]::_oracle[` — yielded the function name `_oracle`, putting
    the attacker back on both sides of the entry comparison (core repro V5-26).

    The grammar is simpler than the depth model assumed, and simpler is what closes it: a pytest id is
    `<file>::<class>::...::<function>` with an OPTIONAL `[parametrisation]` that is always LAST and
    always runs to the end of the string. So the first `[` ends the separator-bearing part of the id,
    and everything from it onward is one opaque value that is never split, whatever it contains.

    FAILS CLOSED on the one shape this cannot parse: a test whose FILE path itself contains a `[`.
    Such an id yields a single part, `function_name_of` returns None, and `own_frame_affirmed` refuses
    to affirm rather than guessing — UNPROVEN, never a false PROVEN (disclosed residual)."""
    cut = test_id.find("[")
    head = test_id if cut < 0 else test_id[:cut]
    tail = "" if cut < 0 else test_id[cut:]
    parts = head.split("::")
    parts[-1] = parts[-1] + tail
    return parts


def function_name_of(test_id: str) -> str | None:
    """The test's own function name: the id's last depth-0 `::` segment minus its `[parametrisation]`.

    None means the id NAMES NO FUNCTION — and since the fifth cycle that is a refusal at the call
    site, never a skipped comparison (see own_frame_affirmed)."""
    parts = split_test_id(test_id)
    if len(parts) < 2:
        return None
    last = parts[-1].strip()
    name = last.split("[", 1)[0].strip()
    return name or None


def own_frame_affirmed(test_id: str, oc: "Outcome", waiver: str | None,
                       tracked: frozenset[str] | None = None) -> tuple[bool, str]:
    """(affirmed, why_not). The ALLOWLIST: only the test's own module + own function affirm a witness.

    `waiver` is the caller's `frame_attribution` input — an explicit declaration that the adapter
    cannot attribute frames at all (the jest/vitest JSON report carries none). Owner's ruling of the
    fifth cycle: the waiver is GRANTED for jest/vitest with the hardening below intact, is never
    available to pytest in practice (its junit body always carries frames), and is PRINTED on every
    receipt that uses it. It is honoured only when BOTH hold:

      * it is spelled in the documented form, `unavailable: <reason>` with a real reason
        (parse_frame_attribution refuses every other spelling — a parse error, never a silent
        waiver); and
      * the reverted run supplied NEITHER frame for this witness. A runner that declared it cannot
        attribute frames and then SUPPLIED one has contradicted itself, so the frames are taken and
        the allowlist applies in full (verifier 4's V4-22: the waiver was overriding an origin that
        was present AND was a plugin).

    The second condition is deliberately `neither` rather than `not both`: honouring a waiver when
    only one frame arrived would leave exactly the V4-22 escape open with one field dropped, and the
    cost of the stricter reading is UNPROVEN on a self-contradicting adapter — the safe direction."""
    module = _file_of(test_id)
    fname = function_name_of(test_id)
    if waiver and not oc.origin and not oc.entry:
        return True, ""
    if not oc.origin:
        return False, (f"the reverted run reported no raising frame for this test, so C1 cannot affirm that the "
                       f"assertion came from {module} itself")
    want = normalize_path_only(module)
    # SIXTH CYCLE (verifier 5, F1). The module is used VERBATIM — never line-stripped, never
    # separator-rewritten. Rewriting the module was what let `tests/test_a.py:9` (a real, tracked file)
    # and `tests/test_a.py` (a DIFFERENT real, tracked file) compare equal, so a helper module's
    # assertion affirmed as the test's own (end-to-end repro V5E16, exit 0, green receipt).
    if _AMBIGUOUS_MODULE_RE.search(want):
        return False, (f"the test's own module is named `{want}`, whose name ends in `:<line>` or contains a "
                       f"backslash — C1 cannot tell that name apart from an adapter's `path:line` frame string, so "
                       f"it cannot affirm which file raised. Two different tracked files may be in play here, and a "
                       f"comparison that cannot distinguish them does not affirm")
    readings = origin_readings(oc.origin)
    if tracked is not None:
        real = [r for r in readings if r in tracked]
        if len(real) > 1:
            return False, (f"the raising frame `{oc.origin}` names more than one tracked file — {', '.join(sorted(real))} "
                           f"— so C1 cannot tell which one raised; an ambiguous frame does not affirm")
        if real:
            readings = real
    if want not in readings:
        return False, (f"the assertion was raised in {readings[-1]}, not in the test's own module {want}")
    if not oc.entry:
        return False, (f"the reverted run named no entry frame, so C1 cannot affirm that {want} raised inside "
                       f"the test's own function")
    if fname is None:
        # FAIL CLOSED. A half of a two-half allowlist that cannot be EVALUATED is not an affirmation:
        # `if fname and ...` used to skip the entry check entirely for an id with no `::`, so a red
        # whose entry frame was `pytest_runtest_call` affirmed itself (verifier 4, V4-07 / V4-08).
        return False, (f"the test id `{test_id}` names no `::<function>`, so C1 cannot compare the raising frame "
                       f"`{oc.entry}` against the test's own function — the entry half of the own-frame allowlist "
                       f"cannot be evaluated, and an allowlist half that cannot be evaluated does not affirm")
    if oc.entry != fname:
        return False, (f"the raising frame in {want} is `{oc.entry}`, not the test's own function `{fname}` — "
                       f"a hook, a fixture or a helper defined beside the test raised it")
    return True, ""


# SIXTH CYCLE (verifier 5, F8). Runner families whose report ALWAYS carries frames for a failed
# assertion, so a declaration that frames are unavailable is a CONTRADICTION rather than a waiver.
# This lives in the CORE, beside the allowlist it switches off, and not only in the Action's
# FRAME_ATTRIBUTION map — the rule: "a guard consulted by convention is a guard defeated by a
# semicolon. Put the assertion inside the thing it protects, not beside it." Before this, `check_c1`
# honoured a waiver from ANY caller and had no runner-family input at all, so the support matrix's
# "pytest is never waived" was enforced only by the caller's own good behaviour (core repro V5-20).
FRAME_VERIFIED_FAMILIES = frozenset({"pytest"})

# SEVENTH CYCLE (verifier 6, F3). The line above was a DENYLIST OF ONE STRING, and everything not on
# it was GRANTED the waiver: `pytst` (one transposed letter), `python -m pytest` (a spelling the
# support matrix itself names for the pytest family), `pytest-xdist`, `''` and `'   '` all switched
# off the own-frame allowlist and returned PROVEN. A guard whose bypass is a typo is not a guard, and
# the sixth cycle's own F8 fix existed precisely to stop this decision depending on the caller —
# moving it into the core and then leaving it open by default moved the hole rather than closing it.
# A blank value failing OPEN is also the same defect F6 closed for `gaming_markers_error` that cycle,
# one function away in this file.
#
# The rule for this surface is an ALLOWLIST: only the families that genuinely CANNOT attribute
# frames may waive, everything else refuses, and an unknown or empty family refuses loudest. The
# waiver exists for exactly one reason — the jest/vitest JSON report carries no frame for a failed
# assertion — so the set of families that may use it is the set of families with that property, named
# here, and nothing else is admitted by omission.
FRAME_UNVERIFIABLE_FAMILIES = frozenset({"jest", "vitest"})
# A reason must contain something a reader can act on. Punctuation is not a reason: `unavailable: .`
# and `unavailable:::` both passed the old non-empty-after-strip test (core repros V5-17 / V5-16).
_REASON_SUBSTANCE_RE = re.compile(r"[^\W\d_]")


def parse_frame_attribution(raw: Any, runner_family: str | None = None) -> str | None:
    """None (strict, the default) or the reason string of an explicit `unavailable: <why>` waiver.

    FIFTH CYCLE (verifier 4, V4-21). This used to accept ANY string whose lowercase form merely
    STARTED with `unavailable`, while the contract has always documented `unavailable: <why>`. So
    `frame_attribution="unavailable"` — no colon, no reason — silently disabled the entire own-frame
    allowlist, and `"unavailablish: nope"` did too. The allowlist's own off-switch was easier to
    reach than the allowlist. The documented form is now REQUIRED exactly: the prefix, a colon, and a
    non-empty reason. Every other spelling is a parse error that CRASHES the check, which is a state
    of its own and is never folded into a verdict (an absolute rule of this codebase)."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError(f"frame_attribution must be a str, got {type(raw).__name__}")
    s = raw.strip()
    if s == "available":
        return None
    head, sep, reason = s.partition(":")
    if head.strip().lower() == FRAME_WAIVED_PREFIX:
        if not sep or not reason.strip():
            raise ValueError(
                f"frame_attribution={raw!r} waives the own-frame allowlist but states no reason. The documented "
                f"form is `unavailable: <why>` — a waiver without a reason is not a waiver, because the reason is "
                f"what gets printed on the receipt.")
        if len(_REASON_SUBSTANCE_RE.findall(reason)) < 3:
            raise ValueError(
                f"frame_attribution={raw!r} states a reason with no words in it. The reason is PRINTED on the "
                f"receipt as the sole explanation of why this verdict is frame-unverified, so punctuation is not a "
                f"reason — `unavailable: .` and `unavailable:::` passed the old non-empty test and told a reader "
                f"nothing. Name the runner and what its report omits.")
        if runner_family is None:
            raise ValueError(
                f"frame_attribution={raw!r} waives the own-frame allowlist but no `runner_family` was supplied. The "
                f"waiver is granted per RUNNER — it exists because the jest/vitest JSON report carries no frame — so "
                f"C1 cannot honour it without knowing which runner produced these results. Pass runner_family.")
        if runner_family in FRAME_VERIFIED_FAMILIES:
            raise ValueError(
                f"frame_attribution={raw!r} waives the own-frame allowlist for runner_family={runner_family!r}, whose "
                f"report ALWAYS carries a frame for a failed assertion. The waiver is not available to it, and this "
                f"refusal is made by C1 itself rather than by the caller's own map: a check that lets any caller "
                f"switch off its central allowlist is not making the guarantee its receipt claims.")
        if runner_family not in FRAME_UNVERIFIABLE_FAMILIES:
            known = ", ".join(sorted(FRAME_UNVERIFIABLE_FAMILIES))
            raise ValueError(
                f"frame_attribution={raw!r} waives the own-frame allowlist for runner_family={runner_family!r}, which "
                f"is not a family C1 knows to be unable to attribute frames. The waiver is an ALLOWLIST, not a "
                f"denylist: it exists because the jest/vitest JSON report carries no frame for a failed assertion, so "
                f"the only families that may use it are the ones with that property ({known}). Every other value "
                f"refuses — a misspelling (`pytst`), an alternative spelling of a frame-verified runner "
                f"(`python -m pytest`, `pytest-xdist`), a family nobody has assessed, and the empty string most of "
                f"all, because a blank family is a caller that did not say which runner produced these results. "
                f"Naming a new family here is a decision someone makes and writes down; it must never be the default "
                f"a typo falls into. The ONE normalisation applied before this comparison is ASCII case and "
                f"surrounding whitespace ({FRAME_WAIVED_PREFIX!r} and {FRAME_WAIVED_PREFIX.capitalize()!r} are the "
                f"same word; `Jest` and `jest` are the same runner). That admits no family nobody wrote down — a "
                f"runner's identity is its name, and ASCII case is not part of it — while `pytst`, `python -m "
                f"pytest` and `pytest-xdist` are different STRINGS and all still refuse.")
        return s
    raise ValueError(f"frame_attribution={raw!r} is not one of 'available' or 'unavailable: <why>'")
FAILURE_KINDS = ("assertion", "import", "collection", "error", "timeout")
SESSION_KEY = "<session>"

# TENTH CYCLE (the owner's queue, 2026-09-07: "guards are enforced by the core, never left to the
# caller"). The `<session>` entry was the last piece of FREE TEXT in this check's contract. Its
# reason is printed on the receipt verbatim and — when there are no per-test results — it is the
# whole content of a NOT_RUN, so an adapter could put anything at all in the one line a reader uses
# to decide whether a run happened. Per-TEST kinds have been a closed vocabulary since the fourth
# cycle; the session-level one was not, and "the adapter writes something sensible" is good
# behaviour rather than enforcement.
#
# The form is now `<kind>: <reason>`, exactly the shape `parse_frame_attribution` already requires
# of a waiver, and every other spelling is a parse error that CRASHES — a state of its own, never
# folded into a verdict.
#
# The vocabulary is DERIVED rather than invented: it is FAILURE_KINDS minus `assertion` (a session
# does not assert; only a test does) plus the two things only a session can report — that it
# produced NO report at all, and that it said something about itself that is not a failure.
SESSION_NOTE_KINDS: tuple[str, ...] = ("no-report", "import", "collection", "timeout", "error", "warning")


def parse_session_note(raw: str) -> tuple[str, str]:
    """(kind, reason) from a `<session>` entry spelled `<kind>: <reason>`. Raises otherwise.

    The reason must contain words for the same reason a frame waiver's must: it is PRINTED as the
    sole explanation of what the run said about itself, and punctuation explains nothing."""
    s = raw.strip()
    head, sep, reason = s.partition(":")
    kind = head.strip().lower()
    if not sep or kind not in SESSION_NOTE_KINDS:
        raise ValueError(
            f"test_failure_kinds_without_change[{SESSION_KEY!r}]={raw!r} is not a session note this check can read. "
            f"The form is `<kind>: <reason>` and <kind> is one of {', '.join(SESSION_NOTE_KINDS)} — a CLOSED "
            f"vocabulary the core enforces, not free text the adapter is trusted to keep sensible. This line is "
            f"printed on the receipt, and when the reverted run produced no per-test results it IS the verdict's "
            f"whole explanation, so what it may say is a decision written down here rather than whatever an "
            f"adapter happened to format. Only ASCII case and surrounding whitespace are normalised.")
    if len(_REASON_SUBSTANCE_RE.findall(reason)) < 3:
        raise ValueError(
            f"test_failure_kinds_without_change[{SESSION_KEY!r}]={raw!r} names the kind `{kind}` but states no "
            f"reason with words in it. The reason is the only thing that tells a reader WHAT the run said about "
            f"itself; punctuation is not a reason.")
    return kind, reason.strip()



def failure_kind_from_text(text: str) -> str:
    """Normalise a `test_failure_kinds_without_change` value to the owner's vocabulary. Exact kind
    words pass through; free text (an exception line, a pytest reason) is classified by prefix."""
    t = text.strip()
    tl = t.lower()
    if tl in FAILURE_KINDS:
        return tl
    if tl.startswith("[xpass(strict)]"):
        return "error"
    if tl.startswith(("assert", "assertionerror", "failed:")) or "assertionerror" in tl[:40]:
        return "assertion"
    if "importerror" in tl or "modulenotfounderror" in tl or tl.startswith("import") or "syntaxerror" in tl:
        return "import"
    if "timeout" in tl or "timed out" in tl:
        return "timeout"
    if any(w in tl for w in ("collection", "fixture", "setup", "teardown", "not found")):
        return "collection"
    return "error"


def normalize_test_id(raw: Any, key: str) -> str:
    """Test ids are strings; `./` prefixes, doubled slashes and surrounding whitespace are not
    identity. Anything that is not a str is refused (CRASHED via the runner)."""
    if not isinstance(raw, str):
        raise TypeError(f"{key}: test ids must be str, got {type(raw).__name__}: {raw!r}")
    s = raw.strip()
    while s.startswith("./"):
        s = s[2:]
    s = re.sub(r"/{2,}", "/", s)
    return s


@dataclass(frozen=True)
class Outcome:
    status: str
    phase: str | None = None
    type: str | None = None
    message: str | None = None
    note: str | None = None          # why a claimed `fail` was reclassified
    origin: str | None = None        # the raising frame (path:line) when the adapter could read it
    entry: str | None = None         # the FIRST frame's function name (`def <name>(`) when readable

    @property
    def is_assertion_red(self) -> bool:
        return self.status == "fail"     # `fail` is only ever produced by _outcome after positive evidence

    def unproven_reason(self) -> str:
        """The UNPROVEN reason this outcome supports when it is NOT an assertion red."""
        if self.status == "skip":
            return "collection"
        if self.type in IMPORT_TYPES:
            return "compile"
        if self.phase in ("collection", "setup", "teardown"):
            return "collection"
        if self.type:
            return "compile"          # raised something that is not an assertion, in the body
        return "collection"           # a bare "error": did not fail by assertion, nothing more known

    def describe(self) -> str:
        bits = [self.status]
        if self.phase:
            bits.append(f"in {self.phase}")
        if self.type:
            bits.append(self.type)
        if self.message:
            bits.append(f"— {self.message.strip().splitlines()[0][:160]}")
        if self.note:
            bits.append(f"[{self.note}]")
        return " ".join(bits)


def _claimed_fail_is_assertion(phase: str | None, typ: str | None, msg: str | None) -> tuple[bool, str | None]:
    """(is_assertion, reason_if_not) for a DICT result claiming `fail`. Frame attribution is NOT
    decided here — it needs the test id, and it produces reason `contaminated`, not a reclassification."""
    if phase not in WITNESS_PHASES:
        return False, f"a failure in the {phase} phase is not an executed assertion (only the call phase may witness)"
    first = (msg or "").lstrip().splitlines()[0] if (msg or "").strip() else ""
    if first and _XPASS_RE.match(first):
        return False, "an XPASS(strict) is an xfail marker outcome, not an assertion red"
    if typ:
        if typ in ASSERTION_TYPES:
            return True, None
        return False, f"type {typ!r} is not an assertion type (exact match required)"
    if first:
        if _ASSERTION_MSG_RE.match(first):
            return True, None
        return False, "the message's first token is not an assertion shape"
    return False, "a dict `fail` with neither type nor message carries no evidence of an assertion"


def _outcome(raw: Any, key: str, test_id: str) -> Outcome:
    if isinstance(raw, str):
        status = raw
        if status not in STATUSES:
            raise ValueError(f"{key}[{test_id!r}] has unknown status {status!r}; expected exactly one of {STATUSES}")
        return Outcome(status)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{key}[{test_id!r}] must be a status string or a dict, got {type(raw).__name__}")
    status = raw.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError(f"{key}[{test_id!r}] has unknown status {status!r}; expected exactly one of {STATUSES}")
    phase = raw.get("phase")
    if phase is not None:
        if not isinstance(phase, str) or phase not in PHASES:
            raise ValueError(f"{key}[{test_id!r}].phase={phase!r} is not one of {PHASES}")
    typ = raw.get("type")
    typ = str(typ).strip() if typ is not None else None
    msg = raw.get("message")
    msg = str(msg) if msg is not None else None
    origin = raw.get("origin")
    if origin is not None and not isinstance(origin, str):
        raise TypeError(f"{key}[{test_id!r}].origin must be a str (`path:line`), got {type(origin).__name__}")
    origin = str(origin) if origin is not None else None
    entry = raw.get("entry")
    if entry is not None and not isinstance(entry, str):
        raise TypeError(f"{key}[{test_id!r}].entry must be a str (the first frame's function name), "
                        f"got {type(entry).__name__}")
    entry = entry.strip() if isinstance(entry, str) and entry.strip() else None
    if status == "fail":
        ok, why = _claimed_fail_is_assertion(phase, typ, msg)
        if not ok:
            return Outcome("error", phase or "call", typ, msg, note=f"claimed fail, reclassified: {why}",
                           origin=origin, entry=entry)
    return Outcome(status, phase, typ, msg, origin=origin, entry=entry)


def _normalize(raw: Any, key: str) -> dict[str, Outcome]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{key} must be a mapping of test_id -> result, got {type(raw).__name__}")
    out: dict[str, Outcome] = {}
    seen_raw: dict[str, Any] = {}
    for tid, v in raw.items():
        nid = normalize_test_id(tid, key)
        oc = _outcome(v, key, nid)
        if nid in out and out[nid] != oc:
            raise ValueError(f"{key}: ids {seen_raw[nid]!r} and {tid!r} normalise to {nid!r} with different results")
        out[nid] = oc
        seen_raw[nid] = tid
    return out


def _normalize_keys(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {normalize_test_id(k, key): v for k, v in raw.items()}


def _file_of(test_id: str) -> str:
    """The id's file part: everything before the FIRST depth-0 `::` (bracket-aware since the fifth
    cycle, so a parametrisation containing `::` cannot move the split — verifier 4, V4-11)."""
    return split_test_id(test_id)[0]


def _marker_for(test_id: str, markers: Mapping[str, Any]) -> str | None:
    if test_id in markers:
        return str(markers[test_id])
    f = _file_of(test_id)
    if f in markers:
        return str(markers[f])
    return None


def check_c1(inputs: Mapping[str, Any]) -> Verdict:
    base, head = str(inputs["base_sha"]), str(inputs["head_sha"])
    with_ = _normalize(inputs["test_results_with_change"], "test_results_with_change")
    without = _normalize(inputs["test_results_without_change"], "test_results_without_change")
    rerun_raw = inputs.get("test_results_without_change_rerun")
    rerun = _normalize(rerun_raw, "test_results_without_change_rerun") if rerun_raw is not None else None
    coll_wo = inputs.get("collection_errors_without_change") or {}
    coll_w = inputs.get("collection_errors_with_change") or {}
    markers = inputs.get("gaming_markers") or {}
    kinds_raw = inputs.get("test_failure_kinds_without_change") or {}
    notes_raw = inputs.get("execution_notes") or []
    contaminating_raw = inputs.get("contaminating_files") or {}
    family_raw = inputs.get("runner_family")
    if family_raw is not None and not isinstance(family_raw, str):
        raise TypeError(f"runner_family must be a str naming the runner that produced these results, got "
                        f"{type(family_raw).__name__}")
    runner_family = family_raw.strip().lower() if isinstance(family_raw, str) else None
    frame_waiver = parse_frame_attribution(inputs.get("frame_attribution"), runner_family)
    # TENTH CYCLE (the owner's ruling, 2026-09-07). `tracked_files` was OPTIONAL, and its absence took
    # a documented fallback: the ambiguity refusal compared against the test's own module name alone.
    # That is an INTENTIONAL REVERSAL of a residual this lane disclosed and defended in the eighth
    # cycle, recorded as a reversal rather than quietly overwritten. The owner's reasoning: a guard
    # that is unreachable when the caller omits an optional field depends on good behaviour rather
    # than enforcement. The whole point of the origin-ambiguity refusal is that `tests/test_a.py:9`
    # can name two different tracked files at once; a caller that supplies no tracked set gets a
    # verdict from a comparison that was never made, and the receipt said so in a line nobody had to
    # read. The key is now REQUIRED (runner.INPUT_SCHEMA), so an absent one is NOT_RUN — the check
    # says "I could not check" instead of answering a question it could not ask.
    tracked_raw = inputs["tracked_files"]
    if not isinstance(tracked_raw, (list, tuple, set, frozenset)) or any(not isinstance(f, str) for f in tracked_raw):
        raise TypeError("tracked_files must be a list of git-tracked repo-relative paths (str)")
    if not tracked_raw:
        # SEVENTH CYCLE (verifier 6, V6N-34). A PRESENT but EMPTY `tracked_files` used to behave
        # exactly like an ABSENT one: `frozenset()` is not None, so the ambiguity block ran, found no
        # reading tracked, and fell through to the plain module comparison — the fallback, reached
        # silently. That is the third appearance of one shape in this file: `gaming_markers_error=""`
        # (sixth cycle, F6), `runner_family=""` (seventh cycle, F3), and now this. The KEY'S PRESENCE
        # is the signal; an empty value is a caller stating it computed the repository's tracked set
        # and the answer was nothing, which cannot be true of a repository whose tests just ran.
        raise ValueError(
            "tracked_files is present but EMPTY. Its presence means the caller computed the repository's tracked "
            "paths, and a repository that produced these test results tracks at least the files the tests live in. "
            "An empty set makes the origin-ambiguity refusal unreachable while leaving the receipt claiming it was "
            "applied. Pass the real tracked set, or omit the key entirely to take the documented fallback.")
    tracked = frozenset(normalize_path_only(f) for f in tracked_raw)
    markers_error = inputs.get("gaming_markers_error")
    if markers_error is not None:
        if not isinstance(markers_error, str):
            raise TypeError("gaming_markers_error must be a str naming why C2's markers are unavailable")
        # SIXTH CYCLE (verifier 5, F6). `gaming_markers_error=""` used to fail OPEN: an empty string is
        # a str, so the type check passed, and it is FALSY, so both use sites below read the disarmed
        # guard as armed and returned PROVEN. The key's PRESENCE is the signal — its value only says
        # why — so a present-but-blank value is a guard reported as disarmed with no reason, which is a
        # CRASH, not a green. `parse_frame_attribution` already refuses a blank reason; this matches it.
        if not markers_error.strip():
            raise ValueError(
                "gaming_markers_error is present but blank. Its presence means C2's gaming markers could NOT be "
                "computed, so C1 has no marker source to certify a witness against; the value must name why, because "
                "that reason is what gets printed on the receipt. Omit the key entirely if the markers ARE available.")
    if not isinstance(contaminating_raw, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in contaminating_raw.items()):
        raise TypeError("contaminating_files must be a mapping of path -> reason (str -> str)")
    contaminating = {k: v for k, v in contaminating_raw.items()}
    for name, m in (("collection_errors_without_change", coll_wo), ("collection_errors_with_change", coll_w),
                    ("gaming_markers", markers), ("test_failure_kinds_without_change", kinds_raw)):
        if not isinstance(m, Mapping):
            raise TypeError(f"{name} must be a mapping, got {type(m).__name__}")
    if not isinstance(notes_raw, (list, tuple)) or any(not isinstance(n, str) for n in notes_raw):
        raise TypeError("execution_notes must be a list of str")
    coll_wo = _normalize_keys(coll_wo, "collection_errors_without_change")
    coll_w = _normalize_keys(coll_w, "collection_errors_with_change")
    markers = _normalize_keys(markers, "gaming_markers")
    kinds: dict[str, str] = {}
    for tid, raw_kind in kinds_raw.items():
        if not isinstance(raw_kind, str):
            raise TypeError(f"test_failure_kinds_without_change[{tid!r}] must be a str, got {type(raw_kind).__name__}")
        if tid == SESSION_KEY:
            kinds[SESSION_KEY] = raw_kind          # validated below, once, by the core's own parser
        else:
            kinds[normalize_test_id(tid, "test_failure_kinds_without_change")] = failure_kind_from_text(raw_kind)
    session_raw = kinds.pop(SESSION_KEY, None)
    session_kind, session_note = (None, None)
    if session_raw is not None:
        session_kind, session_note = parse_session_note(session_raw)
    # the failure kind is authoritative: a "fail" whose kind is not an assertion is not an assertion red
    for tid, kind in kinds.items():
        wo = without.get(tid)
        if wo is not None and wo.status == "fail" and kind != "assertion":
            # the KIND wins over the phase and type the adapter reported: it is the authoritative word on
            # what the reverted run actually did. Before the fourth cycle the reported phase won whenever
            # it was present, so a dict red carrying phase=call + type=AssertionError with kind=collection
            # came out UNPROVEN-compile instead of UNPROVEN-collection — the kind was authoritative in the
            # comment and advisory in the code.
            kind_phase = "collection" if kind in ("collection", "timeout") else (wo.phase or "call")
            kind_type = {"import": "ImportError", "timeout": "Timeout",
                         "collection": "CollectionError"}.get(kind, wo.type or "Error")
            without[tid] = Outcome("error", kind_phase, kind_type, wo.message, note=f"failure kind: {kind}")
        elif wo is not None and wo.status == "error" and kind == "import" and wo.type is None:
            without[tid] = Outcome("error", wo.phase, "ImportError", wo.message, note="failure kind: import")

    if not with_ and not without and not coll_w and not coll_wo:
        raise MissingInput("test_results_with_change (no tests: the PR changed no tests, or the runner collected none)")
    if not with_:
        raise MissingInput("test_results_with_change (no tests ran with the change; nothing can be a witness)")
    if not without and not coll_wo and session_note:
        raise MissingInput(f"test_results_without_change (the reverted-tree run produced no per-test results — "
                           f"[{session_kind}] {session_note[:300]})")

    ev: list[str] = [f"C1 red-on-revert: base {base[:12]} head {head[:12]} — {len(with_)} test(s) ran with the "
                     f"change, {len(without)} on the reverted tree"
                     + (f", {len(coll_wo)} file(s) failed to collect on the reverted tree" if coll_wo else "")]
    for n in notes_raw:
        ev.append(f"  {n}")
    for path, why in contaminating.items():
        ev.append(f"  SAFETY REFUSAL — infrastructure changed in this PR: {path} ({why}). PR-authored test infrastructure is "
                  f"never reverted, so it RUNS during the reverted phase; no witness from this run can be trusted. Rule: a C1 "
                  f"witness is trusted only when its assertion failure originates in the test's own call on a tree whose "
                  f"non-test, non-infrastructure code is the only thing reverted. This is a safety refusal, not a finding of intent.")
    # TENTH CYCLE (verifier 9, filed minor 2). This banner used to print whenever a waiver was
    # DECLARED, and to say "C1 did NOT confirm that the witness's assertion came from the test's own
    # function" even when every witness HAD supplied frames and the own-frame allowlist had been
    # applied to all of them in full. It understated rather than overstated, so it was safe — and it
    # was still a receipt describing work the check did not do, which is exactly what this product
    # sells against. The line is now written AFTER the per-test loop, from what the waiver actually
    # carried, and inserted here so a reader still meets it before the per-test lines.
    frame_banner_at = len(ev) if frame_waiver else None
    if markers_error:
        ev.append(f"  GUARD DISARMED — C2's gaming markers are unavailable ({markers_error.strip()[:200]}); a witness cannot be "
                  f"certified without them")

    witnesses: list[str] = []
    refused: list[str] = []
    frame_refused: list[str] = []          # per-test: the red was not the test's own frame
    flaky: list[str] = []
    gamed: list[str] = []
    reasons: dict[str, str] = {}          # test_id -> unproven reason (non-witnesses)
    red_with_change: list[str] = []
    uncomparable: list[str] = []
    waived: list[str] = []                # reds affirmed ONLY because the waiver applied to them

    for tid, w in with_.items():
        wo = without.get(tid)
        if wo is None:
            f = _file_of(tid)
            if f in coll_wo:
                wo = Outcome("error", "collection", None, str(coll_wo[f]))
            else:
                uncomparable.append(tid)
                continue
        marker = _marker_for(tid, markers)

        if w.status in ("fail", "error"):
            red_with_change.append(tid)
            ev.append(f"  {tid}: {w.describe()} WITH the change — not a witness (red on both trees)"
                      if wo.status in ("fail", "error") else
                      f"  {tid}: {w.describe()} WITH the change — not a witness")
            reasons[tid] = "green-on-revert"
            continue
        if w.status == "skip":
            ev.append(f"  {tid}: skipped WITH the change — never executed, not a witness")
            reasons[tid] = "collection"
            if marker:
                gamed.append(f"{tid} ({marker})")
            continue

        # w passed with the change
        if wo.is_assertion_red:
            affirmed, why_not = own_frame_affirmed(tid, wo, frame_waiver, tracked)
            if affirmed and frame_waiver and not wo.origin and not wo.entry:
                waived.append(tid)        # this one was taken at the adapter's word
            if not affirmed:
                frame_refused.append(tid)
                reasons[tid] = "contaminated"
                ev.append(f"  {tid}: pass with the change; {wo.describe()} on the reverted tree — a witness in FORM, "
                          f"REFUSED for safety: {why_not}. A C1 witness is trusted only when its assertion failure "
                          f"originates in the test's OWN function in its OWN module; every other frame — a plugin, a "
                          f"hook, a fixture, a helper module, a TEST METHOD INHERITED FROM A BASE CLASS IN ANOTHER "
                          f"FILE (`class TestAdd(AddContract)`: the assertion's origin is the base class's module, "
                          f"not this test's), the code under test — could have produced that red without the test "
                          f"asserting anything. Assert in the test body to be provable. This is a safety refusal, "
                          f"not a finding of intent")
                if marker:
                    gamed.append(f"{tid} ({marker})")
                continue
            if marker in VOIDING_MARKERS:
                # A REASON IS RECORDED HERE, always. Before the pivot this branch `continue`d with no
                # entry in `reasons`, because the `gamed` list was going to carry the verdict. Now that
                # a marker never decides the verdict, a voided witness with no reason would fall out of
                # every list and reach `MissingInput` -- NOT_RUN, "I could not check", for a run that
                # was measured in full. `contaminated` is the honest reason and is already the word for
                # "a witness in form, refused for safety": the PR's own diff put a token in the runner's
                # path, so the red cannot be attributed to the test.
                gamed.append(f"{tid} ({marker})")
                reasons[tid] = "contaminated"
                ev.append(f"  {tid}: pass with the change; {wo.describe()} on the reverted tree — BUT the diff gave this "
                          f"test a {marker} token (it can end the runner or write the report); its red is not trusted")
                continue
            if contaminating:
                refused.append(tid)
                reasons[tid] = "contaminated"
                ev.append(f"  {tid}: pass with the change; {wo.describe()} on the reverted tree — a witness in form, REFUSED "
                          f"for safety: {', '.join(contaminating)} changed in this PR and ran during the reverted phase")
                continue
            if rerun is not None:
                r = rerun.get(tid)
                if r is None or not r.is_assertion_red:
                    flaky.append(tid)
                    ev.append(f"  {tid}: pass with the change; {wo.describe()} on the reverted tree; "
                              f"rerun: {r.describe() if r else 'absent'} — the red did not reproduce (flaky)")
                    reasons[tid] = "flaky"
                    continue
                ev.append(f"  {tid}: pass with the change; {wo.describe()} on the reverted tree; "
                          f"rerun reproduced it — WITNESS")
            else:
                ev.append(f"  {tid}: pass with the change; {wo.describe()} on the reverted tree — WITNESS")
            witnesses.append(tid)
            if marker:
                ev.append(f"  {tid}: note — the diff added a {marker} marker here (C2 reports it); the test still went red")
            continue

        # passed with the change, did not fail by assertion on the reverted tree
        if wo.status == "pass":
            reasons[tid] = "green-on-revert"
            ev.append(f"  {tid}: pass on BOTH trees — green on revert, the test does not depend on the change")
        else:
            reasons[tid] = wo.unproven_reason()
            what = "skipped" if wo.status == "skip" else wo.describe()
            ev.append(f"  {tid}: pass with the change; {what} on the reverted tree — not an assertion red "
                      f"(UNPROVEN-{reasons[tid]})")
        if marker:
            gamed.append(f"{tid} ({marker})")

    if frame_banner_at is not None:
        # Owner's ruling, fifth cycle: the waiver is granted for jest/vitest, hardened, and LABELLED.
        # A reader must see at a glance whether THIS verdict is frame-verified — so the line says what
        # the waiver actually carried on this run, not merely that a waiver was declared.
        n_framed = sum(1 for oc in without.values() if oc.origin or oc.entry)
        if waived:
            line = (f"  FRAME-UNVERIFIED C1 — the runner declared it cannot attribute frames: {frame_waiver}. "
                    f"C1 did NOT confirm that the assertion came from the test's own function in its own module "
                    f"for {len(waived)} red(s) ({', '.join(sorted(waived))}); those are taken at the adapter's "
                    f"word, so this verdict is WEAKER than a pytest verdict, which is always frame-verified. The "
                    f"waiver is honoured only for a red whose reverted run supplied NEITHER an origin nor an entry "
                    f"frame: a runner that declared frames unavailable and then supplied one has contradicted "
                    f"itself, and the own-frame allowlist is applied to it in full. STATED RESIDUAL, printed on "
                    f"every receipt that uses it")
        else:
            line = (f"  FRAME WAIVER DECLARED AND NOT USED — the runner declared it cannot attribute frames: "
                    f"{frame_waiver}, but the reverted run supplied frames for {n_framed} of "
                    f"{len(without)} result(s) and the waiver applies only to a red carrying NEITHER an origin nor "
                    f"an entry frame. Every red this verdict rests on was checked against the own-frame allowlist "
                    f"IN FULL, so this verdict is frame-verified and is NOT weakened by the declaration. Said "
                    f"plainly because the alternative — printing the weaker sentence anyway — would be a receipt "
                    f"describing work the check did not do, in the safe direction, which is still a receipt that "
                    f"is not true")
        ev.insert(frame_banner_at, line)
    if frame_refused:
        ev.append(f"  frame attribution refused {len(frame_refused)} red(s) that were witnesses in form: "
                  f"{', '.join(frame_refused)}. KNOWN COST of the allowlist (checks/README.md, accepted residuals): "
                  f"an HONEST test whose assertion fires inside a shared helper module, or inside the code under "
                  f"test, reads UNPROVEN-contaminated here. Assert in the test body to be provable")
    for tid in uncomparable:
        ev.append(f"  {tid}: ran with the change but is ABSENT from the reverted run and no collection error "
                  f"names its file — cannot compare")

    if rerun is None:
        ev.append("  flakiness: not assessed — no rerun of the reverted tree was supplied")
    if session_note:
        # SIXTH CYCLE (verifier 5's V5-78, "a session-level abort note alongside a witness — PROVEN or
        # NOT_RUN?"). Answer, and the reasoning, because a by-design row has to be argued: INFORMATIONAL.
        # A `<session>` entry means the RUN said something about itself, and that is not the same as the
        # run saying its per-test outcomes are wrong. The tests in this snapshot DID execute and their
        # recorded outcomes are real; a teardown hook that crashes after the last test, or the benign
        # "1 warning during collection" this project's own contract test uses, invalidates none of them.
        # Refusing every witness on any session note was tried in this cycle and REVERTED: it turns a
        # warning into a refusal, and the only way to tell a fatal note from a benign one is to
        # classify free text, which is a heuristic — and required checks here are deterministic
        # comparisons, never a judgement call.
        #
        # The genuine gap V5-78 found was in the CONTRACT, not the code: nothing said whether this note
        # was load-bearing, so a reader could not tell a discarded signal from a deliberate one. It is
        # deliberate, it is now said here and in checks/README.md, and the receipt says so too — a
        # reader of the verdict must not have to guess whether this line changed anything.
        #
        # A session-level failure that DOES invalidate the run is reported the way the contract has
        # always specified: with NO per-test results, which is NOT_RUN a few lines above. That is the
        # path this project's own adapter uses (c1_runner sets `<session>` only where results are empty).
        ev.append(f"  {SESSION_KEY} note from the reverted run, kind `{session_kind}` (one of "
                  f"{'/'.join(SESSION_NOTE_KINDS)}, a closed vocabulary this check enforces), INFORMATIONAL — it "
                  f"did not affect this verdict: {session_note[:200]}. The per-test outcomes above were recorded by that same run and "
                  f"stand on their own; a run whose session failure invalidates its results reports NO per-test "
                  f"results, which is NOT_RUN, not a witness")

    approval = None
    if not witnesses and not flaky and not reasons and not gamed and not refused:
        raise MissingInput("test_results_without_change (no test id appears in both runs: "
                           + ", ".join(sorted(uncomparable)[:10]) + ")")

    if gamed:
        # C1 NO LONGER ACCUSES (the owner's decision of 2026-09-15; see verdict.ALLOWED_STATES["C1"]).
        # This used to `return Verdict("C1", "GAMED_SUSPECT", ...)`. The markers are C2's finding, and
        # C2 is not on the shipped path: the field sweep measured its accusatory precision at 12.3%
        # against a gate of >=85%, and every one of the nine idiom classes reached a user either as
        # C2's own verdict or as a marker that turned THIS verdict into an accusation.
        #
        # What replaces it is not silence and not a softer accusation -- it is the reason the same run
        # already computed. A marker aimed at a test that produced no trusted assertion red is, once
        # the marker is set aside, simply a test that did not go red: the loop above has already
        # written "collection" / "green-on-revert" / "contaminated" into `reasons` for exactly those
        # ids, and the fall-through below returns UNPROVEN with that reason. The ESCAPE side is
        # unchanged in both directions: PROVEN still requires a trusted assertion red, so nothing that
        # was GAMED becomes green, and nothing honest is accused.
        #
        # An OBSERVATION, never a finding (the decidability rule): what was seen is named,
        # and the line says plainly that it did not decide this verdict.
        ev.append(f"OBSERVATION (did not decide this verdict): the caller supplied marker(s) aimed at test(s) C1 "
                  f"needed, none of which produced a trusted assertion red: {', '.join(gamed)}. C1 reports what it "
                  f"could and could not PROVE and makes no finding about intent; the verdict below is the reason "
                  f"those tests gave on the reverted tree.")
    if refused:
        ev.append(f"UNPROVEN-contaminated: {len(refused)} witness(es) in form refused for safety — {', '.join(sorted(contaminating))} "
                  f"changed in this PR and ran during the reverted phase: {', '.join(refused)}. Not a finding of intent; the "
                  f"infrastructure change is reported by C2")
        return Verdict("C1", "UNPROVEN", base, head, approval, tuple(ev), reason="contaminated")
    if witnesses and markers_error:
        text = f"guard disarmed: C2's gaming markers are unavailable ({markers_error.strip()[:160]}) and {len(witnesses)} witness(es) are present — C1 cannot certify a witness without its marker source"
        ev.append(f"CRASHED: {text}")
        return Verdict("C1", "CRASHED", base, head, approval, tuple(ev), error=text)
    if witnesses:
        ev.append(f"PROVEN: {len(witnesses)} witness(es) failed by assertion on the reverted tree and pass with "
                  f"the change: {', '.join(witnesses)}"
                  + (f"; not witnesses (red with the change): {', '.join(red_with_change)}" if red_with_change else ""))
        # THIRTEENTH CYCLE, the owner's third bar (2026-09-07): the standing limit of red-on-revert is
        # NAMED on the receipt that carries the PROVEN, in its own words, every time. It is not a
        # weakness to hide -- it is the boundary of what a green here means, and a reader who does not
        # know it will read the green as more than it is.
        ev.append("  RESIDUAL what a witness proves is that the test EXERCISES the change: it fails when the change "
                  "is reverted and passes when it is applied. It does not prove the change is WANTED. A PR that "
                  "alters behaviour and updates its own test to match the new behaviour produces exactly this "
                  "shape, and no deterministic check can tell it from an intentional change -- only a human review "
                  "of the diff can. Corund proves the test is real, not that the behaviour is right")
        # FOURTEENTH CYCLE, addendum (the owner's ruling, 2026-09-08): the floor of what a witness proves,
        # in the owner's words, on the escape side, beside every PROVEN, once. A statement about what a
        # proven witness proves belongs beside the PROVEN it qualifies and nowhere a witness was not proven.
        ev.append("  RESIDUAL (escape side) " + X7_RESIDUAL)
        return Verdict("C1", "PROVEN", base, head, approval, tuple(ev))
    if flaky:
        ev.append(f"UNPROVEN-flaky: the only red(s) did not reproduce on the rerun: {', '.join(flaky)}")
        return Verdict("C1", "UNPROVEN", base, head, approval, tuple(ev), reason="flaky")
    for reason in ("contaminated", "compile", "collection", "green-on-revert"):
        if reason in reasons.values():
            named = [t for t, r in reasons.items() if r == reason]
            if reason == "green-on-revert":
                truly = [t for t in named if t not in red_with_change]
                parts = []
                if truly:
                    parts.append(f"{len(truly)} test(s) green on revert: {', '.join(truly)}")
                if red_with_change:
                    parts.append(f"{len(red_with_change)} test(s) red WITH the change (never a witness): "
                                 f"{', '.join(red_with_change)}")
                ev.append("UNPROVEN-green-on-revert (base case): no test passes with the change AND fails by "
                          "assertion without it; " + "; ".join(parts))
            else:
                ev.append(f"UNPROVEN-{reason}: no test failed by assertion on the reverted tree; "
                          f"{len(named)} test(s) carry this reason: {', '.join(named)}")
            return Verdict("C1", "UNPROVEN", base, head, approval, tuple(ev), reason=reason)  # type: ignore[arg-type]
    raise MissingInput("test_results_without_change (nothing comparable)")
