"""`run_check` — the ONE seam through which a check's outcome may be decided from an exception.

    run_check(check, inputs) -> Verdict      NEVER raises.
      unknown check id / non-mapping inputs   -> CRASHED (error names the problem)
      a required key absent, or None unless
        the key is declared NULLABLE          -> NOT_RUN (evidence names each missing key)
      the check raised ANYTHING               -> CRASHED, error = "<ExcType>: <the exception's text>"
                                                 (BaseException included: a hostile mapping whose
                                                 `.get` raises SystemExit, a value whose __str__
                                                 raises KeyboardInterrupt — never re-raised)
      the check returned a non-Verdict, a Verdict for another check, or a state it may not speak
                                              -> CRASHED (an internal invariant broke; never green)

The guard is inside the thing it protects (this function IS the public entry point), not beside it,
so a future check implementation cannot bypass it ("a guard consulted by convention is
a guard defeated by a semicolon"). INPUT_SCHEMA documents the REQUIRED keys per check;
OPTIONAL_INPUTS the optional ones. Both are data the README is checked against, not prose.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from . import c1_red_on_revert, c2_skip_audit, c3_gate_fold, c4_approval_binding
from .verdict import ALLOWED_STATES, MissingInput, Verdict

INPUT_SCHEMA: dict[str, dict[str, str]] = {
    "C1": {
        "base_sha": "SHA of the base commit the PR's non-test diff was reverted onto",
        "head_sha": "SHA of the PR head whose new/changed tests were run",
        "test_results_with_change": "{test_id: status | {status, phase?, type?, message?}} from the "
                                    "run WITH the PR's non-test diff applied (status: pass|fail|skip|error)",
        "test_results_without_change": "{test_id: ...} from the run with the PR's non-test diff REVERTED",
        # TENTH CYCLE (the owner's ruling, 2026-09-07: guards are enforced by the core, never left to
        # the caller). This key was OPTIONAL, with a documented fallback. An INTENTIONAL REVERSAL of a
        # residual this lane disclosed and defended, recorded as one: the origin-ambiguity refusal
        # exists because `tests/test_a.py:9` can name two different tracked files at once, and it is
        # UNREACHABLE without this set — so omitting the key bought a verdict from a comparison that
        # was never made. Absent is now NOT_RUN.
        "tracked_files": "[path] the paths git TRACKS in the repository under test (NOT the PR's changed files). "
                         "Without it the origin-ambiguity refusal cannot run at all, so an absent key is NOT_RUN "
                         "rather than a verdict from an unmade comparison; present-but-empty is CRASHED",
    },
    "C2": {
        "base_sha": "SHA of the merge-base the diff was taken from",
        "head_sha": "SHA of the PR head the diff was taken to",
        "diff": "unified diff text (git diff --no-renames <base> <head>)",
        "skip_allowlist": "[pattern | 'pattern  reason'] — the loud-skip allowlist AS OF THE BASE TREE",
        # TWELFTH CYCLE (verifier 11's nine spellings; the owner's rule of 2026-09-07). REQUIRED for
        # the same reason `tracked_files` became required for C1 in the tenth: without it C2's success
        # sentence affirmed "deletes or silences no assertion or test" from a comparison that had never
        # been made, and nine silencing spellings walked through it with a fully green receipt. An
        # OMITTED key is NOT_RUN; a `state` outside the closed vocabulary is CRASHED; a stated
        # `not-measured` reason is honoured and PRINTED, and it strikes the runtime clause from C2's
        # success sentence. Absence never reads as absence-of-problem.
        "silencing_probe": "{state, files, base_tests_on_head, head_tests_on_head, head_tests_on_base} — the "
                           "runtime silencing comparison. `state` is `measured`, or one of "
                           "corund_checks.runtime_silencing.NOT_MEASURED_REASONS "
                           "(not-supplied | no-modified-test-files | no-non-test-change | runner-unsupported | "
                           "runner-error | no-report) with the reason PRINTED on the receipt. When measured, the "
                           "three maps are `{test_id: status | {status, phase?, type?, message?, origin?, entry?}}` "
                           "for the BASE version of the modified test files run against the PR's code, the PR's own "
                           "tests with the change, and the PR's own tests on the reverted tree; `files` names the "
                           "modified test files probed. A test whose base version fails BY ASSERTION and whose "
                           "PR version neither fails nor witnesses is a PROVEN silencing",
    },
    "C3": {
        "protection_before": "branch-protection snapshot (GitHub API shape) before the PR",
        "protection_after": "branch-protection snapshot after the PR (same shape)",
        "workflow_files_before": "{path: text} of every .github/workflows file at the base",
        "workflow_files_after": "{path: text} of every .github/workflows file at the head",
    },
    "C4": {
        "approvals": "[{reviewer_id, commit_sha, state}] in submission order (GitHub review states)",
        "head_sha": "the exact head SHA an approval must bind to (merge_sha accepted as a fallback key)",
    },
}

# NINTH CYCLE (verifier 8, FILED). `None` and ABSENT are the same thing at this seam, and for almost
# every key that is right. For a branch-protection snapshot it is not: `protection_after = None` is
# GitHub's own way of saying the branch has NO protection any more — the most severe gate fold there
# is — and the uniform rule swallowed it into `I could not check`. checks/README.md's C3 input table
# has always said the key may be `None`, and `_required_contexts()` handles None deliberately, so the
# code and the published contract disagreed and the code's answer was the useless one. It failed
# closed, which is why verifier 8 filed it rather than calling it an escape.
#
# This is the THIRD appearance of one shape in this codebase — `gaming_markers_error = ''` (sixth
# cycle), `tracked_files = []` (eighth), and now this: THE KEY'S PRESENCE IS THE SIGNAL, and a
# present-but-empty value is a VALUE. The fix is a per-key declaration rather than dropping the None
# rule, because an OMITTED key really is a missing input and must still be NOT_RUN — a caller that
# forgot to fetch the after-snapshot must never get a verdict from that omission.
NULLABLE_INPUTS: dict[str, dict[str, str]] = {
    "C3": {
        "protection_before": "None means the base branch had NO branch protection at all",
        "protection_after": "None means the PR REMOVED branch protection wholesale — every required "
                            "context is gone, which C3 reports as a fold, never as a missing input",
    },
}

OPTIONAL_INPUTS: dict[str, dict[str, str]] = {
    "C1": {
        "test_results_without_change_rerun": "{test_id: ...} second run on the reverted tree; a red "
                                             "that does not reproduce is UNPROVEN-flaky. Absent = flakiness not assessed (stated)",
        "collection_errors_without_change": "{path: message} files that failed to collect/import on "
                                            "the reverted tree — their tests are UNPROVEN-collection",
        "collection_errors_with_change": "{path: message} same, for the run with the change",
        "test_failure_kinds_without_change": "{test_id: assertion|import|collection|error|timeout (or free text the kind is "
                                             "derived from)} on the reverted tree — authoritative for whether a red is an "
                                             "assertion; a `<session>` entry with no per-test results = the run itself produced "
                                             "nothing -> NOT_RUN with that reason",
        "gaming_markers": "{test_id or path: kind} skips/xfails/constant-true/runner-escape the PR's own diff added, "
                          "as found by C2 — aimed at a test C1 needed = GAMED_SUSPECT; a `runner-escape` marker on a "
                          "witness voids the witness",
        "execution_notes": "[str] lines the runner adds about HOW the runs were executed (isolation, reconciliation, "
                           "test-infrastructure files kept, the reverted diff's composition) — echoed on the receipt verbatim",
        "contaminating_files": "{path: reason} test-infrastructure files the PR adds or modifies that can affect outcomes (a "
                               "conftest hook, an autouse fixture, a plugin); such code is never reverted and RUNS during the "
                               "reverted phase, so every witness is refused for safety: UNPROVEN-contaminated naming the file",
        "gaming_markers_error": "str: why C2's gaming markers could not be computed (a C2 crash). With a witness present C1 is "
                                "CRASHED 'guard disarmed', never PROVEN; without one the ordinary verdict stands",
        "frame_attribution": "'available' (or absent) = STRICT, the default: a witness must affirm own-frame (its `origin` "
                             "names the test's OWN module and its `entry` names the test's OWN function), else it is "
                             "UNPROVEN-contaminated. 'unavailable: <why>' = the adapter's report carries no frames at all "
                             "(jest/vitest); the allowlist is waived and the declaration is PRINTED on the receipt. "
                             "REQUIRES `runner_family`, and is REFUSED (CRASHED) for a family whose report always "
                             "carries frames — C1 makes that refusal itself rather than trusting the caller's map",
        "runner_family": "str: the runner that produced these results (`pytest`, `jest`, `vitest`, ...). Required "
                         "whenever `frame_attribution` waives the own-frame allowlist, because the waiver is granted "
                         "per runner; `pytest` may never waive it",
    },
    "C2": {
        "c1_needed_tests": "[test_id or path::name] the tests C1 needed; a skip aimed at one is GAMED_SUSPECT. "
                           "Absent = derived from the diff (tests the PR adds or modifies)",
        "skip_allowlist_path": "repo path of the allowlist file, so additions to it in this same diff are named",
        "files_after": "{path: text} full NEW-side text of changed files at the head — enables the AST tier for Python "
                       "and the token tier for JS over whole files (absent = reconstructed from the diff where it parses; "
                       "the receipt says which)",
        "files_before": "{path: text} full OLD-side text of changed files at the base — enables the before/after "
                        "collectability comparison (tests silenced by structure: class rename, indentation, __test__ = False)",
    },
    "C3": {
        "required_contexts": "[names] override of the required-check contexts to compare (default: "
                             "derived from protection_before)",
    },
    "C4": {
        "author_id": "the PR author's reviewer_id; an approval bound only by the author is FAILED (self-approval)",
        "merge_sha": "fallback binding target when head_sha is absent",
    },
}

_DISPATCH: dict[str, Callable[[Mapping[str, Any]], Verdict]] = {
    "C1": c1_red_on_revert.check_c1,
    "C2": c2_skip_audit.check_c2,
    "C3": c3_gate_fold.check_c3,
    "C4": c4_approval_binding.check_c4,
}


_MISSING = object()      # `None` is a VALUE for a nullable key, so absence needs its own sentinel


def _crashed(check: str, text: str, base: str | None = None, head: str | None = None) -> Verdict:
    cid = check if check in ALLOWED_STATES else "C1"  # a Verdict needs a real id; the text names the bad one
    return Verdict(check=cid, state="CRASHED", base_sha=base, head_sha=head, approval_sha=None,
                   evidence=(f"CRASHED inside {check}: {text}",), error=text)


def _exc_text(exc: BaseException) -> str:
    try:
        s = str(exc)
    except BaseException:  # noqa: BLE001 — even the exception's own text may be hostile
        s = ""
    return f"{type(exc).__name__}: {s}" if s else type(exc).__name__


def _sha(inputs: Any, key: str) -> str | None:
    try:
        v = inputs.get(key) if isinstance(inputs, Mapping) else None
    except BaseException:  # noqa: BLE001 — a raising mapping must not take the crash path down with it
        return None
    return v if isinstance(v, str) else None


def run_check(check: str, inputs: Mapping[str, Any]) -> Verdict:
    """NEVER raises. See the module docstring for the outcome table."""
    try:
        return _run_check(check, inputs)
    except BaseException as exc:  # noqa: BLE001 — the last belt: nothing leaves this function
        try:
            cid = check if isinstance(check, str) and check in ALLOWED_STATES else "C1"
        except BaseException:  # noqa: BLE001
            cid = "C1"
        return _crashed(cid, f"escaped the check runner: {_exc_text(exc)}")


def _run_check(check: str, inputs: Mapping[str, Any]) -> Verdict:
    if not isinstance(check, str) or check not in _DISPATCH:
        return _crashed(str(check) if isinstance(check, str) else "C1",
                        f"unknown check id {check!r}; expected one of {sorted(_DISPATCH)}")
    if not isinstance(inputs, Mapping):
        return _crashed(check, f"TypeError: inputs must be a Mapping, got {type(inputs).__name__}")
    base, head = _sha(inputs, "base_sha"), _sha(inputs, "head_sha")

    # Required keys, named one by one. `merge_sha` may stand in for C4's head_sha (contract).
    try:
        nullable = NULLABLE_INPUTS.get(check, {})
        missing = []
        for k in INPUT_SCHEMA[check]:
            if k in nullable:
                # THE KEY'S PRESENCE IS THE SIGNAL: `None` here is a value the check reads, and only
                # an OMITTED key is a missing input. Asked through `.get` with a SENTINEL, never
                # through `in`: `.get` is the accessor every other key already goes through, so a
                # hostile mapping (one whose `get` raises, or whose `__contains__` disagrees with it)
                # behaves here exactly as it did before this key became nullable, and the
                # raising-comparator path still reaches CRASHED instead of NOT_RUN.
                if inputs.get(k, _MISSING) is _MISSING:
                    missing.append(k)
            elif inputs.get(k) is None:
                missing.append(k)
        if check == "C4" and "head_sha" in missing and inputs.get("merge_sha") is not None:
            missing.remove("head_sha")
    except BaseException as exc:  # noqa: BLE001
        return _crashed(check, _exc_text(exc), base, head)
    if missing:
        return Verdict(check=check, state="NOT_RUN", base_sha=base, head_sha=head, approval_sha=None,
                       evidence=tuple(f"NOT_RUN: missing input `{k}` — {INPUT_SCHEMA[check][k]}"
                                      for k in missing))

    try:
        verdict = _DISPATCH[check](inputs)
    except MissingInput as mi:
        return Verdict(check=check, state="NOT_RUN", base_sha=base, head_sha=head, approval_sha=None,
                       evidence=tuple(f"NOT_RUN: missing or unusable input `{n}`" for n in mi.names))
    except BaseException as exc:  # noqa: BLE001 — classified into CRASHED, never swallowed, never green, never re-raised
        return _crashed(check, _exc_text(exc), base, head)

    if not isinstance(verdict, Verdict):
        return _crashed(check, f"check returned {type(verdict).__name__}, not a Verdict", base, head)
    if verdict.check != check:
        return _crashed(check, f"check returned a Verdict for {verdict.check}, not {check}", base, head)
    if verdict.state not in ALLOWED_STATES[check]:  # belt: Verdict already refuses this
        return _crashed(check, f"{check} spoke {verdict.state}, which it may not", base, head)
    return verdict
