"""TWELFTH CYCLE (verifier 11). The runtime silencing rule: read the OUTCOME, never the syntax.

Verifier 11 refuted "zero escapes" with nine spellings that silence an EXISTING test while the PR
carries one genuine witness, so the receipt came back fully green. The nine differ only in HOW the
test is neutralised -- a `pytest.skip` reached through a dict literal, `del test_mul`, a shadowing
second `def`, a lambda rebind, `globals().pop`, a swallowing decorator, a `__code__` swap, an
assertion moved into a nested def nobody calls, an assertion moved into an unconsumed generator
expression. Eleven cycles of syntactic detection had answered each spelling with another pattern,
and a twelfth spelling always existed. The receipt already contradicted itself in one breath:

    C2| PROVEN: 0 finding(s) - ... and deletes or silences no assertion or test
    C1|   tests/test_mul.py::test_mul: skipped WITH the change - never executed, not a witness

THE OWNER'S RULE (2026-09-07): a test in the diff-needed set whose RUNTIME OUTCOME with the change
is skipped / not-executed / not-collected, and which is not on the loud-skip allowlist with a
reason, is a FINDING -- regardless of the silencing spelling.

WHAT THAT REQUIRES, AND WHY THE OBVIOUS READING IS NOT ENOUGH. Measured end to end on this tree by the
twelfth cycle's evidence run, the nine produce THREE runtime signatures, not one:

    N17                      status `skip` with the change                          (1 of 9)
    N2, N34                  the id is in NEITHER run -- never collected            (2 of 9)
    N33 N3 N31 N4 N14 N13    `pass` on BOTH trees -- "green on revert"              (6 of 9)

So "skipped or not-executed", read literally against the with-change run alone, reaches three of
nine; the other six EXECUTE and PASS, and their runtime record is byte-identical to an honest test
that happens to live in a changed test file and does not exercise the changed code -- the largest
honest class there is. Reading the with-change run alone cannot separate them, in either direction.

WHAT SEPARATES ALL NINE FROM HONEST CODE IS TEMPORAL, and C1's two trees never measured it: the
diff turned a test that RAN ITS ASSERTION into one that does not. So this rule adds the one run
nobody had: the BASE version of the modified test files, executed against THIS PR's code. Then

    the base version of this test FAILS BY ASSERTION against the PR's code   (it detects the change)
    AND the PR's version of that same test does not                          (it no longer does)

is a PROVEN silencing, and it is the same fact for all nine spellings because it never looks at one.
Both halves are runtime outcomes; nothing here parses a test.

THE SECOND HALF IS WHAT KEEPS THE RULE OFF ORDINARY CODE, and it is the whole false-positive story.
"The PR's version does not detect the change" is NOT "the PR's version passes". The most common pull
request in existence fixes a bug and updates the test that encoded the old behaviour: there, the base
version fails against the new code (first half TRUE) and the updated test passes with the change --
but it FAILS ON THE REVERTED TREE, which is exactly C1's witness. So the second half is:

    no case of this test function fails by assertion on the REVERTED tree  (it is not a witness)
    AND no case of it fails or errors WITH the change                      (the PR is not already red)

An honest bugfix-with-updated-test is a witness and is never accused. A test the PR left red is a
caught regression, not a silencing, and is never accused.

THE FIRST HALF IS WHAT KEEPS THE RULE OFF SKIPPED-BY-ENVIRONMENT CODE. It demands the base version
FAIL BY ASSERTION -- C1's own standard of proof, `Outcome.is_assertion_red`, applied to the base
version of the test. A `@pytest.mark.skipif(shutil.which('docker') is None)` integration test is
skipped in the base version too, so it never proves anything and is never accused; the same for
`pytest.importorskip`, for a platform gate, and for a test whose base version errors on the new code
because the PR changed a signature (a TypeError is not an assertion). A test the PR ADDS has no base
version at all and is out of scope by construction. Each of those is a mainstream shape and each is
excluded by the standard of proof rather than by an allowlist entry the customer has to write.

SCOPE IS DERIVED, NEVER SEARCHED. The set of test functions this rule judges is whatever
pytest itself collected from the BASE content of the test files the diff MODIFIES -- enumerated by
the runner, from git, not by a pattern in this file and not from the same list the implementation
iterates. `Report.scope` carries the sentence.

FAIL CLOSED. `state` is a CLOSED vocabulary the CORE enforces: `measured`, or one of
NOT_MEASURED_REASONS. An unknown word, a missing map, or a `measured` report with no probed files
RAISES -- which run_check turns into CRASHED, never into a green. When the comparison did not run,
the caller gets a Report that SAYS SO, and check_c2 prints it and strikes the runtime clause from
its own success sentence: absence never reads as absence-of-problem.

WHAT THIS DOES NOT DECIDE. It reports rows; it does not read the loud-skip allowlist and does not
build Findings. `check_c2` does both, so the allowlist stays parsed in exactly one place and a
silencing already reported by a syntactic detector is deduplicated there rather than counted twice.

FOURTEENTH CYCLE (verifier 13). THE CLEARANCE IS LINKED, NOT GLOBAL. The thirteenth cycle cleared a
name-absent base test whenever ANY test of a new name failed by assertion on the reverted tree,
anywhere in the PR's modified test files -- so `test_mul` (covering `mul`) was cleared by
`test_add_fixed` (covering `add`), and the twelfth cycle's N2 walked again behind an honest fix. Now a
base test with no counterpart is ACCOUNTED FOR, one test at a time, from runtime facts:

    relocated        a test of the same name lives under another path in this PR
    replaced         its base version DETECTS the change and entered {S}, the functions this PR
                     changed; this PR's tests of a NEW name that detect the change (a live witness, or
                     one that cannot even run on the reverted tree because what it calls does not
                     exist there) between them enter every function of {S}, on the SAME tree
    exercised        its base version does not detect the change (or this PR changes no code); it
                     entered {S}, and tests this PR still runs enter every function of {S}
    subject-deleted  its base version ERRORS in its own frame because what it called no longer
                     exists, and this PR adds no function in that file: a test deleted with what it
                     covered, which has no replacement to demand
    NOT DECIDED      its base version ERRORS because what it called no longer exists and this PR
                     ADDS functions in that file (a rename of the subject, or an unrelated addition --
                     the runtime record is the same); or it errors because the signature changed; or
                     it entered no function this PR changed at all. Stated on the receipt, by name,
                     never an accusation and never a green sentence

Anything else with a detecting base version and no counterpart is SILENCED, and its PROOF line names
the functions it exercised and says that no replacement witness exercised them. The SUBJECT of a test
is what it ENTERED during its call phase, recorded by the Action's own pytest plugin
(`corund_action/subject_trace.py`, `checks/corund_checks/subjects.py`) -- a runtime fact, compared on
the same tree, never a name. A PR that changes NO non-test code is now measured too (nothing can
detect, so only `relocated` / `exercised` / `subject-deleted` / NOT DECIDED apply), which is what
lets a pure `git mv` of a test file be read as the move it is instead of three deletions.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from . import subjects as _subjects
from .c1_red_on_revert import Outcome, _normalize, split_test_id
from .subjects import CANNOT_REACH_TYPES, RESIGNED_TYPES, ChangedFunctions, Subjects

# The tier-A syntactic kinds that READ a diff to guess at a fact this probe can MEASURE. Each is kept
# (it is the only answer when the probe did not run) and each defers to the probe under the exact
# conditions `Report.defer_reason` states. Closed set, so a reader can see what is subordinate to the
# measurement and what is not.
#
# FOURTEENTH CYCLE (verifier 13, E2/E3, D07): `mark-skipif` and `importorskip` join `skip-call`. They
# are the idiomatic spellings of the same intent -- a test that steps aside when a service, a binary
# or an optional dependency is absent -- and the probe's positive fact answers all three the same way:
# every base test in that file still EXECUTES with the change, so the added gate costs no existing
# test its run. `mark-xfail` / `xfail-call` are NOT here, on purpose: an xfail keeps the test running
# and discards its verdict in every environment, so "still executes" is not the fact that would clear
# it; the loud form for an xfail is the base-tree allowlist, and check_c2 says so beside the finding.
DEFER_TO_RUNTIME: frozenset[str] = frozenset({"test-deleted", "assertion-deleted", "skip-call", "mark-skipif", "importorskip"})
_SKIP_KINDS: frozenset[str] = frozenset({"skip-call", "mark-skipif", "importorskip"})
_DELETION_KINDS: frozenset[str] = frozenset({"test-deleted", "assertion-deleted"})

# The closed vocabulary of "the comparison did not run, and here is why". The CORE refuses any other
# word (guards are enforced by the core, never left to the caller: a free-text reason would let an
# adapter spell its way past the rule with a sentence nobody validates).
NOT_MEASURED_REASONS: dict[str, str] = {
    "not-supplied": "the caller supplied no runtime probe (a core-surface caller: a unit test, the "
                    "replay CLI, the app's dossier). The runtime comparison did not run",
    "no-modified-test-files": "the PR modifies no EXISTING test file, so no test has a base version "
                              "to compare against (tests it only ADDS are out of scope by construction)",
    "no-non-test-change": "the PR changes no non-test code, so there is no change for a test to detect",
    "runner-unsupported": "the runner family cannot be executed by this Action",
    "runner-error": "the probe run failed (the test command, the tree, or the report)",
    "no-report": "the probe run produced no machine-readable report, so no outcome could be read",
}

# THIRTEENTH CYCLE (verifier 12, D3). The WIDE half's own closed vocabulary. `not-needed` is a
# MEASUREMENT, not a silence: it says the wide base-tests run found no assertion red outside the files
# this PR touches, so there was no question for the two conditional phases to answer. Anything else is
# the comparison saying it did not happen, and the core refuses a green that rests on it.
SUITE_STATES: dict[str, str] = {
    "not-needed": "no test in a file this PR does not touch fails by assertion against this PR's code, so the "
                  "cross-file comparison had nothing to decide and its two extra phases did not run",
    "measured": "the two cross-file phases ran",
    "runner-error": "a cross-file phase failed to run",
    "no-report": "a cross-file phase produced no machine-readable report",
}

_HEAD_WORDS = {
    "absent": "is never collected",
    "skip": "is SKIPPED with the change",
    "pass": "is green on BOTH trees",
    "unrun": "runs NO case with the change",
    "cross-file": "is neutralised by THIS PR's changes to ANOTHER test file",
}
_HEAD_LONG = {
    "absent": "is never collected at all — the id appears in neither run of this PR's own tests",
    "skip": "is SKIPPED with the change and never reaches its assertion",
    "pass": "passes with the change AND passes with the change reverted, so it no longer detects the change",
    "unrun": "executes NO case at all: every case it owns is SKIPPED",
    "cross-file": "stops failing the moment THIS PR's own test files are put back at their head content, "
                  "while the code under test does not move at all",
}

# The closed vocabulary of how a base test with no counterpart was ACCOUNTED FOR (fourteenth cycle).
ACCOUNT_KINDS: dict[str, str] = {
    "relocated": "alive under another path in this PR",
    "replaced": "its base version detects the change and a witness of a new name exercises the same subject",
    "exercised": "what it exercised is still exercised by a test this PR runs",
    "subject-deleted": "what it called no longer exists on this PR's code and this PR adds nothing in its place",
    "undecided-successor": "what it called no longer exists on this PR's code and this PR ADDS functions in that file",
    "undecided-resigned": "its base version cannot call what it called (the signature changed)",
    "undecided-no-subject": "its base version entered no function this PR changed, so nothing can be linked to it",
}
UNDECIDED_KINDS: frozenset[str] = frozenset({"undecided-successor", "undecided-resigned", "undecided-no-subject"})


def _fns(fs) -> str:
    fs = sorted(fs)
    return ", ".join(fs[:4]) + (f" (+{len(fs) - 4} more)" if len(fs) > 4 else "") if fs else "nothing"


@dataclass(frozen=True)
class Silenced:
    """One PROVEN silencing: what the base version did, what the PR's version does."""
    path: str
    target: str                  # the test's own name (Class::method for a method), never the file
    base_evidence: str           # what the base version of this test did against the PR's code
    head_word: str               # the closed word for what the PR's version does
    head_evidence: str           # the outcome text behind that word
    moved_to: str | None = None  # the PR's version was found under a different path (a moved test)
    via_frame: str | None = None # the base red fired in THIS file, which the PR does not change
    proof_kind: str = "base-detects"   # "base-detects" | "ran-on-revert" | "cross-file" | "unreplaced"
    subject: tuple[str, ...] = ()      # the changed functions its base version entered (fourteenth cycle)
    linked: tuple[str, ...] = ()       # replacement candidates that exercise SOME of the subject
    uncovered: tuple[str, ...] = ()    # the part of the subject no candidate exercises
    note: str = ""                     # e.g. the subject trace was unavailable

    @property
    def test_id(self) -> str:
        return f"{self.path}::{self.target}"

    @property
    def _assertion(self) -> str:
        """The failing assertion's own words, short enough to survive the receipt's snippet window."""
        msg = self.base_evidence.split("—", 1)[-1].strip() if "—" in self.base_evidence else self.base_evidence
        return msg.strip()[:44] or "assertion failed"

    def snippet(self) -> str:
        if self.proof_kind == "cross-file":
            return ("SILENCED (measured, not parsed): this test is in a file THIS PR DOES NOT TOUCH; it detects "
                    "this PR's change until this PR's OTHER test files are restored, and then it does not")
        if self.proof_kind == "ran-on-revert":
            # THIRTEENTH CYCLE. The other half of the comparison: the base version proved nothing (it
            # was itself skipped), but the PR's OWN version of this test executed on the reverted tree
            # and executes nothing at all with the change. C1 already printed that fact as "cannot
            # compare"; this turns it into the finding it always was.
            # Short enough that `Finding.line()`'s 140-character window carries the WHOLE sentence: a
            # snippet whose second half is truncated away reads as an accusation with its proof cut off.
            return (f"SILENCED (measured, not parsed): this test RAN with the change REVERTED and "
                    f"{_HEAD_WORDS[self.head_word]}")
        if self.proof_kind == "unreplaced":
            return ("SILENCED (measured, not parsed): this test RAN at base, is never collected with the change, "
                    "and no test this PR runs still exercises what it exercised")
        return (f"SILENCED (measured, not parsed): base version FAILS BY ASSERTION on this PR's code "
                f"({self._assertion}); this PR's version {_HEAD_WORDS[self.head_word]}")

    def _linkage(self) -> str:
        """The fourteenth cycle's sentence: WHY no replacement witness cleared this test."""
        if self.note:
            return f" {self.note}"
        if not self.subject:
            return ""
        if self.linked:
            return (f" No replacement witness exercised its whole subject: its base version entered {_fns(self.subject)}, "
                    f"which this PR changes; this PR's tests of a new name that detect the change ({', '.join(self.linked[:3])}) "
                    f"exercise part of that, and {_fns(self.uncovered)} is exercised by none of them.")
        return (f" No replacement witness exercised its subject: its base version entered {_fns(self.subject)}, which this "
                f"PR changes, and no test of a new name that detects the change enters any of it.")

    def proof(self, allow_note: str) -> str:
        """The full proof, printed beneath the finding: both runs, and the allowlist's answer."""
        moved = f" (this PR has it under {self.moved_to})" if self.moved_to else ""
        if self.proof_kind == "cross-file":
            return (f"    PROOF {self.test_id}: this test lives in a file THIS PR DOES NOT TOUCH. Run against this "
                    f"PR's code with this PR's own test files RESTORED to their base content it {self.base_evidence} "
                    f"- it detects the change. Run against the SAME code with this PR's test files at their head "
                    f"content it {_HEAD_LONG[self.head_word]} ({self.head_evidence}). The code under test is identical "
                    f"in both runs, so the only thing that stops it failing is this PR's changes to ANOTHER TEST FILE. "
                    f"It was not already failing before this PR (the full-revert run says so), so this is not a PR "
                    f"that fixes cross-test pollution. {allow_note}")
        if self.proof_kind == "ran-on-revert":
            return (f"    PROOF {self.test_id}: THIS PR's own version of this test {self.base_evidence} when the "
                    f"non-test diff is REVERTED, and with the change it{moved} {_HEAD_LONG[self.head_word]} "
                    f"({self.head_evidence}). A test that ran before this diff and runs no case after it was "
                    f"silenced by the diff, whatever spelling did it, and it fails by assertion on neither tree, "
                    f"so it is no witness and no caught regression. {allow_note}")
        if self.proof_kind == "unreplaced":
            return (f"    PROOF {self.test_id}: the BASE version of this test EXECUTED against this PR's code (it "
                    f"{self.base_evidence}) and entered {_fns(self.subject)}. This PR's own version of it "
                    f"{_HEAD_LONG['absent']}, and of the tests this PR runs none enters {_fns(self.uncovered)}: its "
                    f"deletion removes the only execution of that code from the suite. This is measured, not read "
                    f"from the diff, and it does not depend on any change to the code under test. {allow_note}")
        via = ("" if not self.via_frame else
               f" Its assertion fired in {self.via_frame}, a file THIS PR DOES NOT CHANGE, so the red is base "
               f"content reacting to this PR's code and cannot have been authored to manufacture it.")
        return (f"    PROOF {self.test_id}: the BASE version of this test, run against THIS PR's code, "
                f"{self.base_evidence} — it detects the change.{via} This PR's own version of it{moved} "
                f"{_HEAD_LONG[self.head_word]} ({self.head_evidence}), and it fails by assertion on neither "
                f"tree, so it is no witness and no caught regression.{self._linkage()} {allow_note}")


@dataclass(frozen=True)
class Undecided:
    """The base version proves it detects the change, but the PR's counterpart cannot be identified
    (the name now exists in more than one place). C2 draws no conclusion and makes no finding."""
    path: str
    target: str
    base_evidence: str
    candidates: tuple[str, ...]

    def snippet(self) -> str:
        # Short enough to survive the receipt's 140-character snippet window WHOLE. An observation
        # whose second half is truncated away reads as an accusation with the retraction cut off.
        return (f"NOT DECIDED: this test's base version detects the change, but this PR holds "
                f"{len(self.candidates)} tests of that name — no counterpart, no finding")

    def proof(self) -> str:
        return (f"    NOT DECIDED {self.path}::{self.target}: the base version {self.base_evidence} against this "
                f"PR's code, but this PR's tests of that name are {', '.join(self.candidates)} and C2 cannot tell "
                f"which is its counterpart. It draws no conclusion and makes no finding")


@dataclass(frozen=True)
class Accounted:
    """FOURTEENTH CYCLE. How ONE base test with no counterpart of its own name was accounted for, from
    runtime facts. `kind` is one of ACCOUNT_KINDS; the undecided kinds are observations that DECIDE
    NOTHING and say so; the others are the positive facts the syntactic tiers defer to."""
    path: str
    target: str
    kind: str
    why: str                              # the whole sentence, printed on the receipt
    witnesses: tuple[str, ...] = ()       # the tests that answer for it (replaced / exercised / relocated)
    subject: tuple[str, ...] = ()         # the functions its base version entered (the linked ones)

    @property
    def undecided(self) -> bool:
        return self.kind in UNDECIDED_KINDS

    def line(self) -> str:
        tag = "NOT DECIDED" if self.undecided else "NOT SILENCED"
        return f"  {tag} {self.path}::{self.target}: {self.why}"

    def snippet(self) -> str:
        """For the observation Finding an undecided account becomes (140-character window)."""
        short = {
            "undecided-successor": "NOT DECIDED: what this test called no longer exists on this PR's code, and this PR adds "
                                   "functions in that file — a rename cannot be told from a deletion here",
            "undecided-resigned": "NOT DECIDED: this test's base version cannot call what it called (the signature "
                                  "changed), so what it exercised was not measured",
            "undecided-no-subject": "NOT DECIDED: this test's base version entered no function this PR changed, so no "
                                    "replacement can be linked to it — no finding",
        }
        return short.get(self.kind, f"NOT DECIDED: {self.why[:110]}")


@dataclass(frozen=True)
class Report:
    state: str                          # "measured" | one of NOT_MEASURED_REASONS
    rows: tuple[Silenced, ...] = ()
    undecided: tuple[Undecided, ...] = ()
    files: tuple[str, ...] = ()
    n_functions: int = 0                # base-side test FUNCTIONS enumerated in those files
    n_detecting: int = 0                # of those, how many detect this change (the proof standard)
    detail: str = ""
    accounted: tuple[Accounted, ...] = ()   # every name-absent base test that was accounted for
    n_ran_on_revert: int = 0            # of those, how many ran on the reverted tree and not with the change
    enumerated: tuple[tuple[str, str], ...] = ()   # every (path, target) this probe RAN, accused or not
    deleted_files: tuple[str, ...] = ()  # of `files`, the ones the PR DELETES (restored to be run)
    still_running: frozenset = frozenset()  # (path, target) that still execute >=1 case with the change
    suite_state: str = "not-needed"     # one of SUITE_STATES: the cross-file half's own answer
    n_suite_files: int = 0              # tracked test files the WIDE base-tests phase ran
    n_outside: int = 0                  # of those, tests OUTSIDE the PR's own test files it judged
    subjects_state: str = "unavailable" # one of subjects.SUBJECT_STATES
    subjects_detail: str = ""
    test_only: bool = False             # the PR changes no non-test code: nothing can detect, only account
    granularity: tuple[tuple[str, str], ...] = ()   # (path, how the changed functions were read)
    uncollectable: tuple[tuple[str, str], ...] = ()  # (modified/deleted test file, why its BASE content did not collect)
    head_by_name: tuple[tuple[str, tuple], ...] = ()  # bare test name -> the head function keys carrying it

    @property
    def measured(self) -> bool:
        return self.state == "measured"

    @property
    def accused_paths(self) -> frozenset[str]:
        return frozenset({r.path for r in self.rows} | {u.path for u in self.undecided})

    @property
    def accounts(self) -> dict[tuple[str, str], Accounted]:
        return {(a.path, a.target): a for a in self.accounted}

    @property
    def cleared(self) -> tuple[Accounted, ...]:
        """The positive accounts (not the undecided ones): what a reader wants to see as NOT SILENCED."""
        return tuple(a for a in self.accounted if not a.undecided)

    def defer_reason(self, kind: str, path: str, target: str | None) -> str | None:
        """Why a SYNTACTIC finding of `kind` about `path::target` should stand down, or None.

        Owner's ruling 2026-09-07: make the syntactic detectors DEFER TO THE RUNTIME FACT WHERE IT
        EXISTS. The decision lives HERE, in the core, and not as a condition check_c2 spells for
        itself: it is a guard, and a guard the caller composes is a guard defeated by a semicolon.

        It is deliberately narrow, and the narrowness is a MEASURED requirement rather than caution.
        A first cut deferred whenever the probe had run the file and accused nothing in it, and that
        re-opened a real escape (verifier 12's D6: an assertion deleted from a FIXTURE, whose red is
        a SETUP error and therefore outside this rule's standard of proof -- so the probe had no
        opinion, and "no opinion" was being read as "no problem"). The probe licenses a deferral only
        where it can point at the POSITIVE fact that answers the syntactic tier's guess:

            skip-call / mark-skipif / importorskip -> every base test the probe enumerated in that
                file still EXECUTES at least one case with the change, so the added gate costs no
                existing test its run.
            test-deleted / assertion-deleted -> the test named is ACCOUNTED FOR (fourteenth cycle):
                relocated, replaced by a witness of the same subject, still exercised, or deleted
                with the subject it covered; an UNDECIDED account defers too, because the decided
                disposition of an undecidable shape is an observation, never an accusation (the
                decidability rule), and the receipt names it as NOT DECIDED. A file-level finding
                defers only when EVERY base test in that file is accounted for.
        """
        if not self.measured or kind not in DEFER_TO_RUNTIME or path not in self.files:
            return None
        here = [pt for pt in self.enumerated if pt[0] == path]
        if kind in _SKIP_KINDS:
            if not here or path in self.accused_paths or any(pt not in self.still_running for pt in here):
                return None
            return (f"the runtime probe RAN all {len(here)} base test(s) in {path} against this PR's code and every "
                    f"one of them still EXECUTES with the change, so this skip costs no existing test its run")
        accounts = self.accounts
        accused = {(r.path, r.target) for r in self.rows} | {(u.path, u.target) for u in self.undecided}
        uncoll = dict(self.uncollectable)
        if target is not None:
            if (path, target) in accused:
                return None               # a relocated test can still be silenced: the row wins
            a = accounts.get((path, target))
            if a is None and path in uncoll:
                # FOURTEENTH CYCLE (verifier 13, D06): the BASE content of this file could not even be
                # COLLECTED against this PR's code -- the module it imports is gone or renamed -- so none
                # of its tests ran and none could be accounted for at runtime. A same-name test that
                # this PR runs under another path is the mainstream module-rename refactor (the
                # relocation rule, by name); otherwise NOT DECIDED, named, never an accusation.
                bare = target.split("::")[-1]
                same = dict(self.head_by_name).get(bare, ())
                if len(same) == 1 and split_test_id(same[0])[0] != path:
                    return (f"the BASE content of {path} did not collect against this PR's code ({uncoll[path][:120]}), "
                            f"so its tests never ran here; this PR runs a test of this name under another path "
                            f"({same[0]}) -- a module renamed with its tests, a relocation")
                return (f"NOT DECIDED -- the BASE content of {path} did not collect against this PR's code "
                        f"({uncoll[path][:120]}): what it imported no longer exists under that name, so whether "
                        f"{target} was moved, renamed or deleted could not be measured. No finding, no clearance")
            if a is None:
                return None
            return f"{'NOT DECIDED -- ' if a.undecided else ''}{a.why}"
        if here and all(pt in accounts and pt not in accused for pt in here):
            kinds = sorted({accounts[pt].kind for pt in here})
            return (f"every one of the {len(here)} base test(s) in {path} is accounted for at runtime "
                    f"({', '.join(kinds)}); see the NOT SILENCED / NOT DECIDED line for each")
        return None

    def scope_line(self) -> str:
        """Scope derived from X, N items enumerated, all N addressed or each exception named."""
        if not self.measured:
            return (f"  RUNTIME SILENCING PROBE: NOT RUN -- {NOT_MEASURED_REASONS[self.state]}"
                    + (f" ({self.detail})" if self.detail else "")
                    + ". A test silenced with no syntactic marker would NOT be found by this run")
        by_kind: dict[str, int] = {}
        for a in self.accounted:
            by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
        acc = ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items()))
        detect = (f"this PR changes no non-test code, so no test can detect a change here and only the ACCOUNTING "
                  f"half ran"
                  if self.test_only else
                  f"{self.n_detecting} detect this PR's change (their base version fails by assertion against it), of "
                  f"which {sum(1 for r in self.rows if r.proof_kind == 'base-detects')} no longer do")
        return (f"  RUNTIME SILENCING PROBE: scope derived from the runner's own collection of the BASE content "
                f"of {len(self.files)} modified/deleted test file(s) ({', '.join(self.files[:6])}"
                f"{' ...' if len(self.files) > 6 else ''}), {self.n_functions} test function(s) enumerated, "
                f"all {self.n_functions} addressed; {detect}"
                + (f"; {self.n_ran_on_revert} more ran on the reverted tree and run no case with the change" if self.n_ran_on_revert else "")
                + (f"; {sum(1 for r in self.rows if r.proof_kind == 'unreplaced')} ran at base and are neither collected nor replaced" if any(r.proof_kind == 'unreplaced' for r in self.rows) else "")
                + (f", {len(self.undecided)} undecided" if self.undecided else "")
                + (f"; {len(self.accounted)} base test(s) with no counterpart of their own name accounted for ({acc})" if self.accounted else "")
                + f". SUBJECT TRACE: {_subjects.SUBJECT_STATES[self.subjects_state]}"
                + (f" ({self.subjects_detail})" if self.subjects_detail and self.subjects_state != "measured" else "")
                + f". CROSS-FILE: all {self.n_suite_files} tracked test file(s) ran in that same phase; "
                + (f"{self.n_outside} test(s) in file(s) this PR does not touch were judged"
                   if self.suite_state == "measured" else SUITE_STATES[self.suite_state]))

    def residual_lines(self) -> tuple[str, ...]:
        """THIRTEENTH CYCLE, the owner's third bar: the standing limits of this comparison are NAMED
        on the receipt, not left in a document. FOURTEENTH CYCLE (verifier 13, F): BOTH directions --
        where this rule can over-flag AND where it can under-flag -- once each, in plain words."""
        if not self.measured:
            return (f"  RESIDUAL the runtime comparison did not run at all here ({self.state}), so nothing below "
                    f"rests on it and the verdict is a verdict on the diff's TEXT alone",)
        return (
            "  RESIDUAL a PR that changes behaviour and UPDATES ITS OWN TEST to match cannot be told from an "
            "intentional change by any deterministic check: the updated test fails on the reverted tree, which is "
            "exactly C1's witness, and that is the same shape whether the new behaviour is wanted or not. Corund "
            "proves the test still exercises the change; it does not and cannot judge whether the change is right",
            "  RESIDUAL this comparison judges the test files this PR MODIFIES or DELETES, plus - for the "
            "cross-file question only - every other test file git tracks under the caller's own test globs. A test "
            "the PR ADDS has no base version to compare against and is out of scope by construction; a test file "
            "git does not track, or one outside those globs, is never run here",
            "  RESIDUAL (escape side) a base version that does not FAIL BY ASSERTION against this PR's code proves "
            "nothing here: one skipped by its environment (a docker/platform gate, importorskip), one whose red "
            "fires in the SETUP phase (a fixture's assertion, not an executed test call), and one whose assertion "
            "fires in a frame belonging to a file THIS PR ALSO CHANGES (that red could have been authored). A "
            "silencing of such a test is not caught by this rule; the syntactic tiers are what answer for those",
            "  RESIDUAL (escape side) a test whose SUBJECT this PR renames or removes is NOT DECIDED, by name, on this "
            "receipt: its base version ERRORS because what it called no longer exists, and a test of the new name "
            "cannot be run on the reverted tree at all, so whether it still pins behaviour cannot be measured. The "
            "same is true of a base version that cannot call what it called because the signature changed. Corund "
            "does not accuse there and does not clear there; it says NOT DECIDED",
            "  RESIDUAL (escape side) a REPLACEMENT is a test of a new name that detects the change and ENTERS the "
            "functions the deleted test entered, on the same tree. A test that merely CALLS the deleted test's "
            "subject without asserting on it reads as its replacement here, and the receipt names which test "
            "cleared which; a deleted test that reached the changed code without entering a changed function (a "
            "constant, a module attribute, work done in a thread or a fixture) has no measurable subject and is "
            "NOT DECIDED rather than cleared or accused. A failing test's trace stops at its first failing "
            "assertion, so a deleted test's subject is completed from a run on the base tree where it passes; "
            "where that run did not happen, the part of a compound test after its first failing assertion is "
            "not demanded of its replacement",
            "  RESIDUAL (over-flag side) a deleted test whose base version was itself SKIPPED in this environment "
            "measured nothing, so the syntactic tiers keep the answer; and where the subject trace is unavailable "
            "(a runner other than pytest, or a run the plugin could not join) a renamed test whose old name "
            "detects the change is accused, because a rename cannot be told from a deletion without it. The "
            "loud-skip allowlist, with a reason, is the answer in both cases",
            "  RESIDUAL a PR that changes no non-test code is measured for what it still RUNS and still EXERCISES, "
            "never for what it still ASSERTS: with nothing to revert there is no change a test could detect, so a "
            "test moved and weakened in the same test-only PR is caught only if its assertion count drops",
        )


def _fkey(test_id: str) -> str:
    """The test FUNCTION an id belongs to: the id minus its `[parametrisation]`.

    The unit of this rule is the function, never the parametrised case, and that is a false-positive
    fix rather than a convenience: a PR that re-parametrises a test changes every id it owns, so a
    per-id comparison would read every honest re-parametrisation as a test that vanished."""
    cut = test_id.find("[")
    return test_id if cut < 0 else test_id[:cut]


def _target_of(fkey: str) -> str:
    """The name half of a function key: `test_a`, or `TestC::test_m` for a method."""
    parts = split_test_id(fkey)
    return "::".join(parts[1:]) if len(parts) > 1 else fkey


def _by_function(results: Mapping[str, Outcome]) -> dict[str, list[Outcome]]:
    out: dict[str, list[Outcome]] = {}
    for tid, oc in results.items():
        out.setdefault(_fkey(tid), []).append(oc)
    return out


def _ids_by_function(results: Mapping[str, Outcome]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tid in results:
        out.setdefault(_fkey(tid), []).append(tid)
    return out


def _origin_file(oc: Outcome) -> str | None:
    return split_test_id(oc.origin)[0].split(":")[0] if oc.origin else None


def _detects(outcomes: list[Outcome], path: str, target: str,
             changed: frozenset[str]) -> "tuple[Outcome, str | None] | None":
    """The base version of this function detects the PR's change: at least one of its cases fails BY
    ASSERTION -- C1's own standard (`Outcome.is_assertion_red`, which `_outcome` only ever grants
    after positive evidence that an assertion was executed and failed) -- in a frame THIS PR COULD
    NOT HAVE AUTHORED. Returns (the red, the foreign file its assertion fired in or None).

    THIRTEENTH CYCLE (verifier 12, D1/D7). Until now the test was `origin's file == the test's own
    file`, borrowed whole from C1's witness rule. It is the RIGHT rule for a witness and the WRONG
    rule here, and the difference is WHO WROTE THE FRAME. C1 refuses a witness whose red came from
    elsewhere because the PR authored that elsewhere and could have aimed it. Here the assertion
    being run is the BASE content of a test file -- content that predates the diff -- and the frame
    it fires in is another file at whatever content the PR left it. So the question is not "is this
    frame the test's own?" but "does this PR CHANGE the file this frame is in?":

        the frame's file is one the PR CHANGES  -> the red could have been authored. REFUSED.
        the frame's file is one the PR does not -> base content reacting to this PR's code. ACCEPTED.

    A shared assertion helper (`from tests.helpers import check_mul`), a `@contextlib.contextmanager`
    that asserts on exit, an assertion inside the code under test -- all mainstream, all previously a
    silent walk-through: move the helper one file over and the accusation vanished. The `entry` frame
    is still required to be the test's OWN function: the first frame is what pytest entered, and if
    that is not this test then this outcome is not about this test at all.

    Where the runner supplies no frames (jest/vitest) there is nothing to contradict and the red
    stands, exactly as before."""
    for oc in outcomes:
        if not oc.is_assertion_red:
            continue
        if oc.entry and oc.entry != target.split("::")[-1]:
            continue
        ofile = _origin_file(oc)
        if ofile is not None and ofile != path:
            if ofile in changed:
                continue                        # a frame THIS PR wrote: it could have been aimed
            return oc, ofile                    # untouched base content: the red is trustworthy
        return oc, None
    return None


def _cannot_reach(outcomes: list[Outcome], path: str, target: str, types: frozenset[str]) -> Outcome | None:
    """A case of this function that ERRORED in the CALL phase, IN ITS OWN FRAME, with one of `types`:
    the thing it called does not exist on that tree (the callee was never entered). Own frame is
    REQUIRED -- origin in this file, entry this function -- and an outcome with no frames does not
    qualify (fail closed: without the frame it could be an error from anywhere)."""
    fname = target.split("::")[-1]
    for oc in outcomes:
        if oc.status != "error" or oc.phase not in (None, "call") or oc.type not in types:
            continue
        if _origin_file(oc) != path or oc.entry != fname:
            continue
        return oc
    return None


def _executed(outcomes: list[Outcome]) -> list[Outcome]:
    """The cases that actually RAN: anything the runner did not skip. A test whose every case is
    `skip` executed nothing at all, whatever the collection said."""
    return [oc for oc in outcomes if oc.status != "skip"]


def _green(outcomes: list[Outcome]) -> bool:
    """Executes at least one case and none of them is red: the shape of a test that RUNS with the change."""
    return bool(_executed(outcomes)) and not any(oc.status in ("fail", "error") for oc in outcomes)


def analyse(raw: object, diff_text: str | None = None, files_after: Mapping[str, str] | None = None,
            files_before: Mapping[str, str] | None = None) -> Report:
    """The rule. Raises (-> CRASHED) on anything it cannot read; never returns a silent all-clear.

    `diff_text` / `files_after` / `files_before` are the PR's own diff and texts (check_c2 has them):
    from them the FUNCTIONS the non-test diff touches are read, at function granularity where the
    code under test parses. Without them the fallback is FILE granularity -- every function in a
    changed non-test file counts as changed -- which is the strict direction."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"silencing_probe must be a mapping, got {type(raw).__name__}")
    state = raw.get("state")
    if not isinstance(state, str):
        raise TypeError(f"silencing_probe['state'] must be str, got {type(state).__name__}")
    if state != "measured" and state not in NOT_MEASURED_REASONS:
        raise ValueError(f"silencing_probe['state'] is {state!r}, which is not one of the closed vocabulary "
                         f"{('measured',) + tuple(NOT_MEASURED_REASONS)}")
    detail = str(raw.get("detail") or "")
    if state != "measured":
        return Report(state, detail=detail)

    files_raw = raw.get("files")
    if not isinstance(files_raw, (list, tuple)) or not all(isinstance(p, str) for p in files_raw):
        raise TypeError("silencing_probe['files'] must be a list of the modified test file paths")
    files = tuple(sorted(set(files_raw)))
    if not files:
        # THE KEY'S PRESENCE IS THE SIGNAL, and present-but-empty is a VALUE (the shape this codebase
        # has now met four times: gaming_markers_error='', tracked_files=[], protection_after=None).
        # A `measured` probe over no files measured nothing; saying so is `no-modified-test-files`.
        raise ValueError("silencing_probe['state'] is 'measured' but 'files' is empty: a probe that ran over no "
                         "file measured nothing. Say `no-modified-test-files` rather than claiming a measurement")
    changed_raw = raw.get("changed_files")
    if not isinstance(changed_raw, (list, tuple)) or not all(isinstance(p, str) for p in changed_raw):
        # REQUIRED, not optional (guards are enforced by the core, never left to the caller). Without
        # it `_detects` cannot tell a frame the PR AUTHORED from one that is untouched base content,
        # and the twelfth cycle answered that question by refusing both -- which let D1 and D7 walk.
        raise TypeError("silencing_probe['changed_files'] must be a list of every path this PR changes; "
                        "without it a base red cannot be told from a frame this PR authored")
    changed = frozenset(changed_raw)
    suite_files = raw.get("suite_files")
    if suite_files is None:
        suite_files = list(files)
    if not isinstance(suite_files, (list, tuple)) or not all(isinstance(q, str) for q in suite_files):
        raise TypeError("silencing_probe['suite_files'] must be a list of the tracked test files the wide phase ran")
    suite_state = raw.get("suite_state", "not-needed")
    if not isinstance(suite_state, str) or suite_state not in SUITE_STATES:
        raise ValueError(f"silencing_probe['suite_state'] is {suite_state!r}, which is not one of the closed "
                         f"vocabulary {tuple(SUITE_STATES)}")
    test_only = raw.get("no_non_test_change", False)
    if not isinstance(test_only, bool):
        raise TypeError("silencing_probe['no_non_test_change'] must be a bool")
    coll_raw = raw.get("base_collection_errors") or {}
    if not isinstance(coll_raw, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in coll_raw.items()):
        raise TypeError("silencing_probe['base_collection_errors'] must be a mapping of test file -> the collection error "
                        "its BASE content raised against this PR's code")
    uncollectable = {k: v for k, v in coll_raw.items() if k in set(files_raw)}
    # FOURTEENTH CYCLE. REQUIRED of a measured probe: the subject trace, or the stated reason it is
    # not available. A measured probe that says nothing about it is a caller claiming a measurement
    # it did not take, and the rename clearance would have to guess.
    subj = _subjects.parse_subjects(raw.get("subjects"))
    base = _normalize(raw.get("base_tests_on_head"), "silencing_probe['base_tests_on_head']")
    head = _normalize(raw.get("head_tests_on_head"), "silencing_probe['head_tests_on_head']")
    reverted = _normalize(raw.get("head_tests_on_base"), "silencing_probe['head_tests_on_base']")

    test_paths = set(files) | set(suite_files)
    non_test_changed = [p for p in sorted(changed) if p not in test_paths]
    if diff_text is not None:
        chf = _subjects.changed_functions(diff_text, files_after, files_before, test_paths)
        # the diff may be narrower than `changed_files` (a caller handing over part of it): the strict
        # direction is to count every changed non-test path the diff did not describe at file level
        missing = [p for p in non_test_changed if p not in chf.paths]
        if missing:
            chf = ChangedFunctions(chf.head, chf.base, chf.wildcard_head | frozenset(missing),
                                   chf.wildcard_base | frozenset(missing), chf.added, chf.removed,
                                   chf.paths | frozenset(missing),
                                   {**chf.granularity, **{p: "file (not in the diff text supplied)" for p in missing}})
    else:
        chf = ChangedFunctions.file_level(non_test_changed)

    base_fn = _by_function(base)
    head_fn = _by_function(head)
    rev_fn = _by_function(reverted)
    base_ids = _ids_by_function(base)
    head_ids = _ids_by_function(head)
    rev_ids = _ids_by_function(reverted)
    in_scope = {k: v for k, v in base_fn.items() if split_test_id(k)[0] in files}

    def is_subject(fn: str) -> bool:
        return fn.split("::", 1)[0] not in test_paths

    def s_base_head(fkey: str) -> frozenset[str]:
        return frozenset(f for f in subj.of("base_tests_on_head", base_ids.get(fkey, ())) if is_subject(f))

    def s_head_head(fkey: str) -> frozenset[str]:
        return frozenset(f for f in subj.of("head_tests_on_head", head_ids.get(fkey, ())) if is_subject(f))

    def s_head_base(fkey: str) -> frozenset[str]:
        return frozenset(f for f in subj.of("head_tests_on_base", rev_ids.get(fkey, ())) if is_subject(f))

    def s_base_full(fkey: str) -> frozenset[str]:
        """The base test's subject with the TRUNCATION repaired: what it entered on this PR's code, plus
        what it entered on the base tree (where it passes and runs to the end) that is still a def on
        head. A function it reached only on base and that no longer exists on head (renamed away) is not
        added: no head test could enter it, and demanding it would accuse every rename."""
        on_head = s_base_head(fkey)
        if _subjects.REPAIR_KEY not in subj.runs or not subj.traced(_subjects.REPAIR_KEY, base_ids.get(fkey, ())):
            return on_head
        on_base = frozenset(f for f in subj.of(_subjects.REPAIR_KEY, base_ids.get(fkey, ())) if is_subject(f))
        return on_head | frozenset(f for f in on_base if f in chf.head_defs)

    # An index of the PR's tests by BARE NAME, so a test the PR MOVED to another file is recognised as
    # the same test rather than accused of vanishing (an honest, mainstream refactor).
    by_name: dict[str, list[str]] = {}
    for k in head_fn:
        by_name.setdefault(_target_of(k).split("::")[-1], []).append(k)
    # The WIDE base-tests run keeps files the PR ADDS at head content (they have no base version), so
    # a test in one of them is NOT a base name: it is exactly the kind of NEW-name test the linkage is
    # looking for (verifier 13's case (d): a rename INTO an added file, clearable now).
    added_test_files = {q for q in suite_files if q in changed and q not in files}
    base_names = {_target_of(k).split("::")[-1] for k in base_fn if split_test_id(k)[0] not in added_test_files}

    # ---- THE REPLACEMENT CANDIDATES (fourteenth cycle) ------------------------------------------
    # Every test this PR runs, classified by what its two runs say:
    #   live          fails BY ASSERTION on the reverted tree in an acceptable frame -- C1's witness
    #   cannot-reach  green with the change; on the reverted tree it errors IN ITS OWN FRAME because
    #                 what it calls does not exist there (AttributeError / NameError / ImportError) or
    #                 has another signature (TypeError at the call site), AND it entered no function of
    #                 a changed non-test file before erroring -- it could not reach the changed code at
    #                 all. That is the runtime shape of a test whose SUBJECT this PR renamed or re-signed
    #   executes      green with the change (what a test-only PR's tests, and kept tests, can show)
    # A `replaced` account demands live or cannot-reach candidates OF A NEW NAME; an `exercised`
    # account (nothing detects) accepts any executing test.
    cands: dict[str, tuple[str, bool]] = {}          # head fkey -> (kind, new name?)
    for k in sorted(head_fn):
        kpath, ktarget = split_test_id(k)[0], _target_of(k)
        new_name = ktarget.split("::")[-1] not in base_names
        head_ocs, rev_ocs = head_fn[k], rev_fn.get(k, [])
        if not test_only and _detects(rev_ocs, kpath, ktarget, changed) is not None:
            cands[k] = ("live", new_name)
        elif _green(head_ocs):
            cr = None if test_only else _cannot_reach(rev_ocs, kpath, ktarget, CANNOT_REACH_TYPES | RESIGNED_TYPES)
            if cr is not None and not any(chf.in_changed_file(f) for f in s_head_base(k)):
                cands[k] = ("cannot-reach", new_name)
            else:
                cands[k] = ("executes", new_name)

    def cover(sigma: frozenset[str], kinds: frozenset[str], need_new: bool) -> tuple[tuple[str, ...], frozenset[str]]:
        """(the candidates that exercise part of `sigma`, the part none of them exercises)."""
        linked: list[str] = []
        covered: set[str] = set()
        for k, (kind, new_name) in cands.items():
            if kind not in kinds or (need_new and not new_name):
                continue
            s = s_head_head(k)
            if s & sigma:
                linked.append(k)
                covered |= s & sigma
        return tuple(linked), frozenset(sigma - covered)

    deleted_raw = raw.get("deleted_files") or []
    if not isinstance(deleted_raw, (list, tuple)) or not all(isinstance(q, str) for q in deleted_raw):
        raise TypeError("silencing_probe['deleted_files'] must be a list of the test files this PR DELETES")
    deleted_files = tuple(sorted(set(deleted_raw) & set(files)))

    rows: list[Silenced] = []
    undecided: list[Undecided] = []
    accounted: list[Accounted] = []
    enumerated: list[tuple[str, str]] = []
    still_running: set[tuple[str, str]] = set()
    n_detecting = 0
    n_ran_on_revert = 0
    trace_note = ("" if subj.measured else
                  f"The subject trace is unavailable here ({subj.detail}), so a replacement of the same subject could "
                  f"not be looked for and a rename cannot be told from a deletion: Corund accuses (fail closed).")

    def gone_account(path: str, target: str, err: Outcome, fkey: str) -> Accounted:
        """The base version errored IN ITS OWN FRAME because what it called is not on this PR's code."""
        if err.type in RESIGNED_TYPES:
            return Accounted(path, target, "undecided-resigned",
                             f"its base version cannot call what it called on this PR's code ({err.describe()[:120]}) -- "
                             f"the signature changed -- so which function it exercised was never measured, and whether "
                             f"this PR's tests still pin it cannot be decided. No finding, no clearance")
        added_here = sorted(f for f in chf.added if f.split("::", 1)[0] in {p for p in chf.paths})
        unread = sorted(p for p in chf.paths if p in chf.wildcard_head and p.lower().endswith((".py", ".pyi")))
        if unread and not added_here:
            # "adds no function" can only be said of a file that was READ at function level; a file
            # counted wholesale (no head text, or unparsable) may add anything, and saying otherwise
            # would clear a rename behind a fallback (fail closed: NOT DECIDED)
            return Accounted(path, target, "undecided-successor",
                             f"its base version ERRORS in its own frame on this PR's code ({err.describe()[:120]}): what it "
                             f"called no longer exists, and this PR's change to {_fns(unread)} could not be read at function "
                             f"level, so whether it adds a successor in its place is unknown. NOT DECIDED",
                             (), ())
        if not added_here:
            return Accounted(path, target, "subject-deleted",
                             f"its base version ERRORS in its own frame on this PR's code ({err.describe()[:120]}): what it "
                             f"called no longer exists, and this PR adds no function in the file(s) it changes "
                             f"({_fns(chf.paths)}) -- a test deleted with what it covered, which has no replacement to demand")
        # a successor may exist: is any added function exercised by a test that distinguishes the trees?
        succ = [k for k, (kind, _n) in cands.items() if kind in ("live", "cannot-reach") and s_head_head(k) & set(added_here)]
        if succ:
            return Accounted(path, target, "undecided-successor",
                             f"its base version ERRORS in its own frame on this PR's code ({err.describe()[:120]}): what it "
                             f"called no longer exists. This PR adds {_fns(added_here)} in that file, exercised by "
                             f"{', '.join(succ[:3])}, which cannot run on the reverted tree -- so whether that test pins "
                             f"the new behaviour cannot be measured, and whether the added function IS this test's "
                             f"subject under a new name cannot be told from an unrelated addition. NOT DECIDED",
                             tuple(succ), tuple(added_here))
        return Accounted(path, target, "undecided-successor",
                         f"its base version ERRORS in its own frame on this PR's code ({err.describe()[:120]}): what it "
                         f"called no longer exists. This PR adds {_fns(added_here)} in that file and no test this PR "
                         f"runs exercises them in a way that distinguishes the trees -- whether one of them is this "
                         f"test's subject under a new name cannot be told from an unrelated addition. NOT DECIDED",
                         (), tuple(added_here))

    for fkey in sorted(in_scope):
        path = split_test_id(fkey)[0]
        target = _target_of(fkey)
        enumerated.append((path, target))
        base_ocs = in_scope[fkey]
        found = None if test_only else _detects(base_ocs, path, target, changed)
        red, via = found if found else (None, None)
        if red is not None:
            n_detecting += 1

        moved_to = None
        ambiguous: tuple[str, ...] = ()
        counterpart = fkey if fkey in head_fn else None
        if counterpart is None:
            same = sorted(by_name.get(target.split("::")[-1], []))
            if len(same) == 1:
                counterpart, moved_to = same[0], split_test_id(same[0])[0]
            elif len(same) > 1:
                ambiguous = tuple(same)

        key = counterpart or fkey
        head_ocs = head_fn.get(key, [])
        rev_ocs = rev_fn.get(key, [])
        if moved_to and moved_to != path:
            accounted.append(Accounted(path, target, "relocated",
                                       f"alive under another path in this PR ({moved_to}) -- a relocation, not a deletion",
                                       (key,)))
        if _executed(head_ocs):
            still_running.add((path, target))
        if any(oc.status in ("fail", "error") for oc in head_ocs):
            continue                      # red with the change: a caught regression, not a silencing
        if any(oc.is_assertion_red for oc in rev_ocs):
            continue                      # still fails on the reverted tree: it is a WITNESS, honest

        gone = _cannot_reach(base_ocs, path, target, CANNOT_REACH_TYPES | RESIGNED_TYPES)

        if red is not None:
            if ambiguous:
                undecided.append(Undecided(path, target, red.describe(), ambiguous))
                continue
            if counterpart is None:
                # FOURTEENTH CYCLE: the linked clearance. Its subject is what it entered among the
                # functions this PR changed; a replacement is a NEW-name candidate that detects the
                # change and enters that subject, compared on the SAME tree (both with the change).
                if not subj.measured:
                    rows.append(Silenced(path, target, red.describe(), "absent",
                                         "the id appears in neither run of this PR's tests", None, via,
                                         note=trace_note))
                    continue
                sigma = frozenset(f for f in s_base_full(fkey) if chf.touches_head(f))
                if not sigma:
                    accounted.append(Accounted(path, target, "undecided-no-subject",
                                               f"its base version detects this PR's change but entered no function this PR "
                                               f"changed (it entered {_fns(s_base_head(fkey))}), so no replacement can be "
                                               f"linked to it and none can be demanded. NOT DECIDED: no finding, no clearance",
                                               (), ()))
                    continue
                linked, uncovered = cover(sigma, frozenset({"live", "cannot-reach"}), need_new=True)
                if linked and not uncovered:
                    kinds = sorted({cands[k][0] for k in linked})
                    accounted.append(Accounted(path, target, "replaced",
                                               f"its base version detects this PR's change and entered {_fns(sigma)}, which "
                                               f"this PR changes; this PR holds {len(linked)} test(s) of a NEW name that "
                                               f"enter the same function(s) with the change and detect the change "
                                               f"({', '.join(linked[:3])}: {'/'.join(kinds)}) -- a live witness is never "
                                               f"a silencing, and this one exercises the same subject",
                                               linked, tuple(sorted(sigma))))
                    continue
                rows.append(Silenced(path, target, red.describe(), "absent",
                                     "the id appears in neither run of this PR's tests", None, via,
                                     subject=tuple(sorted(sigma)), linked=linked, uncovered=tuple(sorted(uncovered))))
                continue
            if any(oc.status == "skip" for oc in head_ocs):
                word = "skip"
                ev = next(oc for oc in head_ocs if oc.status == "skip").describe()
            else:
                word, ev = "pass", "pass with the change; pass on the reverted tree"
            rows.append(Silenced(path, target, red.describe(), word, ev, moved_to, via))
            continue

        # ---- the base version did NOT detect the change (or nothing can, on a test-only PR) ----
        if gone is not None and not ambiguous and counterpart is None:
            accounted.append(gone_account(path, target, gone, fkey))
            continue
        if gone is not None and counterpart is not None and not test_only:
            # D5's family, name kept: the base version could not call what it called; the PR's own
            # version runs. Whether it still pins anything is not measurable (see residual_lines).
            accounted.append(gone_account(path, target, gone, fkey))
            continue
        if counterpart is None and not ambiguous and subj.measured and _executed(base_ocs):
            # FOURTEENTH CYCLE, the accounting half. A deleted test that detects nothing (or a test-only
            # PR, where nothing can): is what it exercised still exercised by anything this PR runs?
            sigma = s_base_head(fkey)
            if not sigma:
                accounted.append(Accounted(path, target, "undecided-no-subject",
                                           f"its base version ran against this PR's code and entered no function outside "
                                           f"the test files, so there is nothing measurable it stopped exercising. NOT "
                                           f"DECIDED: no finding, no clearance", (), ()))
                continue
            linked, uncovered = cover(sigma, frozenset({"live", "cannot-reach", "executes"}), need_new=False)
            if linked and not uncovered:
                accounted.append(Accounted(path, target, "exercised",
                                           f"its base version {'does not detect this PR' + chr(39) + 's change and ' if not test_only else ''}"
                                           f"entered {_fns(sigma)}; every one of those is still entered by a test this PR "
                                           f"runs ({', '.join(linked[:3])}), so nothing it exercised has lost its execution",
                                           linked, tuple(sorted(sigma))))
                continue
            rows.append(Silenced(path, target, f"{base_ocs[0].describe()}", "absent",
                                 "the id appears in neither run of this PR's tests", None, None, "unreplaced",
                                 subject=tuple(sorted(sigma)), linked=linked, uncovered=tuple(sorted(uncovered))))
            continue

        # THIRTEENTH CYCLE (verifier 12, D4). The base version proved nothing -- it was itself skipped,
        # because the PR emptied the collection-time data it is parametrised over, in a file that is
        # not this test file. But the PR's OWN version of this test EXECUTED on the reverted tree and
        # executes NOTHING with the change. The measurement that "did not happen" IS the finding: C1
        # already prints "ran with the change but is ABSENT from the reverted run -- cannot compare",
        # and this is that sentence pointed the other way round, where it decides something.
        # An environment-gated test is skipped on BOTH trees and never reaches this.
        # FOURTEENTH CYCLE: on a test-only PR there is no reverted run; the base version's own run on
        # the same code is the "before".
        ran = _executed(rev_ocs) if not test_only else _executed(base_ocs)
        if ran and not _executed(head_ocs):
            if ambiguous:
                undecided.append(Undecided(path, target, "ran on the reverted tree", ambiguous))
                continue
            n_ran_on_revert += 1
            word = "absent" if not head_ocs else "unrun"
            ev = ("the id appears in neither run of this PR's tests" if not head_ocs
                  else "; ".join(sorted({oc.describe() for oc in head_ocs}))[:160])
            rows.append(Silenced(path, target,
                                 f"EXECUTED {len(ran)} case(s) ({ran[0].describe()})",
                                 word, ev, moved_to, None, "ran-on-revert"))

    # ---- THE CROSS-FILE HALF (thirteenth cycle, verifier 12's D3) -------------------------------
    # A modified test file can neutralise a test in a file the PR never touches -- `import test_mul;
    # test_mul.core = <fake>` at module scope is one spelling of it and there is no last spelling. The
    # twelfth cycle's scope was the modified files, so the silenced test was never run and the receipt
    # said "0 detect this PR's change" over a live regression. Two files cannot be compared without
    # being co-run, so the base-tests phase is now WIDE and this is what reads it.
    #
    # The proof, and every clause of it is load-bearing:
    #   * the test is in a file the PR does NOT modify or delete, so its own text is identical on both
    #     sides and nothing about the test itself can explain a change of outcome;
    #   * with this PR's TEST files at BASE content it FAILS BY ASSERTION against this PR's code;
    #   * with this PR's TEST files at HEAD content, against the SAME code, it does not;
    #   * it was NOT already failing before this PR at all (the full-revert run) -- without that clause
    #     a PR that FIXES cross-test pollution shows the identical signature and would be accused, and
    #     fixing test pollution is honest work.
    outside_scope = {k: v for k, v in base_fn.items()
                     if split_test_id(k)[0] in set(suite_files) and split_test_id(k)[0] not in files
                     and split_test_id(k)[0] not in added_test_files}
    n_outside = 0
    if suite_state == "measured":
        suite_head = _by_function(_normalize(raw.get("suite_head_on_head"), "silencing_probe['suite_head_on_head']"))
        suite_base = _by_function(_normalize(raw.get("suite_base_on_base"), "silencing_probe['suite_base_on_base']"))
        for fkey in sorted(outside_scope):
            path = split_test_id(fkey)[0]
            target = _target_of(fkey)
            found = _detects(outside_scope[fkey], path, target, changed)
            if found is None:
                continue
            n_outside += 1
            enumerated.append((path, target))
            head_ocs = suite_head.get(fkey, [])
            if not head_ocs:
                continue                  # not collected in the head-content run either: not this rule's shape
            if any(oc.status in ("fail", "error") for oc in head_ocs):
                continue                  # still red with this PR's own test files: a caught regression
            if any(oc.is_assertion_red for oc in suite_base.get(fkey, [])):
                continue                  # ALREADY failing before this PR: not a red this diff caused
            red, via = found
            word = "cross-file"
            ev = "; ".join(sorted({oc.describe() for oc in head_ocs}))[:120]
            rows.append(Silenced(path, target, red.describe(), word, ev, None, via, "cross-file"))
    elif outside_scope and suite_state != "not-needed":
        # The wide run found reds outside the PR's own test files and the deciding phases did not run.
        # A comparison that did not happen must never read as a green sentence (the owner's rule this
        # cycle, stated for D4 and true here): say so and let C2 narrow its claim.
        raise ValueError(f"silencing_probe['suite_state'] is {suite_state!r} while the wide base-tests run enumerated "
                         f"{len(outside_scope)} test function(s) outside this PR's own test files: the cross-file "
                         f"comparison did not happen and must not be reported as though it had")

    accused_now = {(r.path, r.target) for r in rows} | {(u.path, u.target) for u in undecided}
    accounted = [a for a in accounted if (a.path, a.target) not in accused_now]
    return Report("measured", tuple(rows), tuple(undecided), files, len(in_scope), n_detecting, detail,
                  tuple(accounted), n_ran_on_revert, tuple(enumerated), deleted_files,
                  frozenset(still_running), suite_state, len(suite_files), n_outside,
                  subj.state, subj.detail, test_only, tuple(sorted(chf.granularity.items())),
                  tuple(sorted(uncollectable.items())), tuple(sorted(by_name.items())))


# `checks/tests/harness_sabotage.py` replaces `analyse` to prove the gates notice when this rule is
# switched off. It keeps this handle so the sabotaged run still produces a TRUTHFUL scope line and
# differs from the real one in exactly one way: the findings are gone. A sabotage that also broke the
# receipt would be caught by the wrong assertion.
_analyse_real = analyse
