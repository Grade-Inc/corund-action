"""The receipt: conclusion mapping under the owner's posture (OBSERVE default, BLOCK per rule),
the check-run title, the Markdown rendering (PR comment + step summary; no emoji), the C1 inputs
block the hosted App reads from the C1 check run's text, and the hash-and-suppress redactor for
anything that might carry a secret. NEW module."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping

from corund_checks import CHECK_IDS, CHECK_NAMES, Verdict
from corund_checks.c1_red_on_revert import X7_RESIDUAL
from corund_checks.runner import INPUT_SCHEMA, OPTIONAL_INPUTS
from corund_checks.runtime_silencing import NOT_MEASURED_REASONS

COMMENT_MARKER = "<!-- corund-receipt -->"

# --- what this release RUNS, and what it says about what it does not -----------------------------
# THE ONE SOURCE. `entrypoint.OWNED_CHECKS` is this tuple, and `parse_block` refuses everything
# outside it, so the set of checks the Action opens a check run for, the set it may block on, and the
# set the receipt renders cannot drift apart. A hand list in either place would be the "hardcoded
# list is the default suspect" shape.
RUNNABLE_CHECKS: tuple[str, ...] = ("C1",)

# The owner's product sentence, 2026-09-15, in the owner's words.
PRODUCT_SENTENCE = "Require the PR's new tests to fail on the old code."

# Why each check the contract still declares is not run here. A `block:` naming one of these is
# REFUSED with this text: an un-runnable id silently ignored is a user who believes a gate is armed
# when nothing is -- the fake-green shape this product exists to catch.
NOT_RUN_BY_THE_ACTION: dict[str, str] = {
    "C2": "C2 skip-audit is experimental and observe-only: it is not run by the Action in this "
          "release. It is still in the codebase and reachable through `corund replay`",
    "C3": "C3 is not run by the Action in this release",
    "C4": "C4 is not run by the Action in this release",
}

# ONE line per surface, no more (the owner's wording rule for this release).
C2_FROZEN_LINE = ("C2 skip-audit is experimental and observe-only: it is not run by the Action in "
                  "this release, and is available via `corund replay`.")
# The `skip-allowlist` input is C2's alone. It stays ACCEPTED so that no existing workflow errors on
# upgrade -- and an input that silently does nothing is a lie, so every run says it does nothing.
SKIP_ALLOWLIST_INERT_LINE = ("The `skip-allowlist` input is accepted and INERT in this release: it is "
                             "not read and selects nothing, because C2 is not run by the Action in "
                             "this release.")
# The named receipt residual, IMPORTED from the core rather than retyped, so the owner's sentence of
# 2026-09-08 exists in exactly one place: "a PR whose code and test are wrong IN AGREEMENT has no
# deterministic oracle ... it is a NAMED RECEIPT RESIDUAL". The core also prints it
# beside every PROVEN; here it is a standing statement of what this check decides at all, which is
# what a reader of an UNPROVEN or a CRASHED receipt also needs.
RESIDUAL_LINE = X7_RESIDUAL

# Every standing sentence this release puts on every receipt, in order. Rendered once per surface:
# the PR comment / step summary (render_markdown), the check-run text (entrypoint._summary_output),
# and the receipt JSON (build_receipt's `notes`).
STANDING_NOTES: tuple[str, ...] = (PRODUCT_SENTENCE, RESIDUAL_LINE, C2_FROZEN_LINE,
                                   SKIP_ALLOWLIST_INERT_LINE)

# BLOCK-mode conclusions. Only PROVEN is ever `success`; UNPROVEN is never a block by itself
# (owner's ruling); CRASHED / NOT_RUN block but are distinct from failure — a crash is never green.
_BLOCK_CONCLUSION = {
    "PROVEN": "success", "FAILED": "failure", "GAMED_SUSPECT": "failure", "UNPROVEN": "neutral",
    "CRASHED": "action_required", "NOT_RUN": "action_required",
}


def parse_block(raw: str | None) -> frozenset[str]:
    """The BLOCK opt-in, parsed FAIL-CLOSED against what this release actually runs.

    An id the Action does not run is REFUSED here, by name, rather than accepted and quietly
    ignored: accepting `block: c2` would hand a user a check run that never appears and a gate they
    believe is armed. "A green result on a path that compared nothing is the absence of a question,
    not the answer to one" -- so this raises, the Action's own crash handler turns it
    into CRASHED on every check it owns, and the step exits non-zero."""
    out = set()
    for tok in (raw or "").replace(";", ",").split(","):
        t = tok.strip().upper()
        if not t:
            continue
        if t not in CHECK_IDS:
            raise ValueError(f"block: unknown check id {tok.strip()!r}; expected any of {', '.join(CHECK_IDS)}")
        if t not in RUNNABLE_CHECKS:
            raise ValueError(
                f"block: {tok.strip()!r} cannot be blocked on -- {NOT_RUN_BY_THE_ACTION.get(t, f'{t} is not run by the Action in this release')}. "
                f"This Action runs {', '.join(RUNNABLE_CHECKS)} only, so blocking on {t} would arm nothing. "
                f"Remove it from `block:`.")
        out.add(t)
    return frozenset(out)


def conclusion_for(verdict: Verdict, *, block: bool) -> str:
    if not block:
        return "neutral"
    return _BLOCK_CONCLUSION[verdict.state]


def check_run_name(check: str) -> str:
    return f"Corund {check} {CHECK_NAMES[check]}"


def check_run_title(verdict: Verdict) -> str:
    return f"{check_run_name(verdict.check)}: {verdict.display_state}"


def redact(text: str, secrets: Iterable[str]) -> str:
    out = text
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, f"<redacted sha256:{hashlib.sha256(s.encode()).hexdigest()[:16]}>")
    return out


def build_receipt(*, repo: str | None, pr_number: int | None, base_sha: str | None, head_sha: str | None,
                  block: frozenset[str], runner: dict, verdicts: list[Verdict], runs: list[dict],
                  internal_error: str | None, post_errors: list[str], version: str,
                  check_run_ids: dict | None = None, measured_at: str | None = None) -> dict:
    return {
        "corund_action": version,
        "measured_at": measured_at,
        "repo": repo, "pr_number": pr_number, "base_sha": base_sha, "head_sha": head_sha,
        "posture": {"mode": "OBSERVE" if not block else "BLOCK (per rule)", "block": sorted(block)},
        "runner": runner,
        "verdicts": [v.to_dict() | {"conclusion": conclusion_for(v, block=v.check in block)} for v in verdicts],
        "runs": runs,
        "check_run_ids": dict(check_run_ids or {}),
        "internal_error": internal_error,
        "post_errors": list(post_errors),
        # The standing sentences, machine-readable, on every receipt whatever the verdict -- so a
        # reader parsing the JSON meets the same disclosure a reader of the comment meets.
        "notes": list(STANDING_NOTES),
    }


def render_markdown(receipt: dict) -> str:
    lines = [COMMENT_MARKER, "## Corund receipt", ""]
    b, h = receipt.get("base_sha") or "?", receipt.get("head_sha") or "?"
    blk = receipt["posture"]["block"]
    lines.append(f"base `{b[:12]}` -> head `{h[:12]}`  |  posture: **{receipt['posture']['mode']}**"
                 + (f" on {', '.join(blk)}; every other check is OBSERVE (neutral)" if blk else
                    " (every check-run conclusion is neutral; BLOCK is a per-rule opt-in)"))
    lines.append("")
    lines.append("| check | verdict | conclusion |")
    lines.append("|---|---|---|")
    for v in receipt["verdicts"]:
        lines.append(f"| {v['check']} {v['check_name']} | **{v['display_state']}** | {v['conclusion']} |")
    lines.append("")
    for v in receipt["verdicts"]:
        lines.append(f"### {v['check']} {v['check_name']}: {v['display_state']}")
        if v.get("error"):
            lines.append(f"error: `{v['error']}`")
        lines.append("")
        lines.append("```")
        lines.extend(v["evidence"])
        lines.append("```")
        lines.append("")
    if receipt.get("internal_error"):
        lines.append(f"**The Action itself crashed**: `{receipt['internal_error']}` — the checks it did not finish are "
                     f"reported CRASHED above, never green.")
        lines.append("")
    if receipt.get("post_errors"):
        lines.append("posting errors: " + "; ".join(receipt["post_errors"]))
        lines.append("")
    r = receipt.get("runner") or {}
    if r:
        lines.append(f"runner: {r.get('family', '?')} — `{' '.join(r.get('command', [])) if isinstance(r.get('command'), list) else r.get('command', '')}`")
    lines.append("")
    for note in receipt.get("notes") or STANDING_NOTES:
        lines.append(note)
        lines.append("")
    lines.append(f"Corund never reports a check it did not run. corund-action {receipt.get('corund_action', '?')}")
    text = "\n".join(lines) + "\n"
    return re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", text)


# --- the C1 inputs block (the hosted App's C1 input source) --------------------------------------
# The App cannot run a customer's tests. It finds C1's inputs by scanning the check runs on the PR's
# head SHA for ONE block in some run's `output.text`, delimited by these two markers, and parses the
# JSON between them; app/corund_app/check_adapters.py is the reader of record, and
# action/tests/test_c1_inputs_marker.py carries its regex and key list VERBATIM as the Action-side
# tripwire. An HTML comment: invisible in rendered markdown, machine-readable. The block rides on the
# `Corund C1 red-on-revert` check run ALWAYS -- whatever C1's verdict -- so the App never mistakes an
# Action that ran for one that is not installed.
C1_INPUTS_MARKER_START = "<!-- corund-c1-inputs"
C1_INPUTS_MARKER_END = "-->"
CHECK_RUN_TEXT_LIMIT = 65535          # the Checks API's cap on output.text
_EVIDENCE_RESERVE = 4096              # human text the block never squeezes below while it still has options
_MESSAGE_KEEP = 160                   # a per-test message's first line, as far as the core prints it (Outcome.describe)
_RED = ("fail", "error")
_SKIP_MARKERS = ("skip", "skipif", "todo")
# The App derives this one itself (the Git trees API): a repository's tracked-file list can exceed the
# check-run text on its own, so it never travels in the block.
C1_APP_SIDE_KEYS = frozenset({"tracked_files"})
# Every C1 input the core declares, required and optional, under the core's own names and in the
# core's own shapes -- DERIVED from the core's schema, never typed here, so an optional input is
# published the moment the Action has a value for it. The classification below (scalar / prose /
# evidence) is pinned against the live schema by action/tests/test_c1_inputs_marker.py, so a new
# core input is loud on this side too.
C1_KEYS_FROM_CORE = tuple(k for k in (*INPUT_SCHEMA["C1"], *OPTIONAL_INPUTS["C1"]) if k not in C1_APP_SIDE_KEYS)
_RESULTS_MAP_KEYS = ("test_results_with_change", "test_results_without_change", "test_results_without_change_rerun")
_C1_SCALAR_KEYS = ("base_sha", "head_sha", "runner_family", "frame_attribution", "gaming_markers_error")
_C1_PROSE_KEYS = ("execution_notes",)        # echoed on the receipt verbatim; decides nothing
# The EVIDENCE SET: the witnesses and every guard that judges them (the rerun, the failure kinds, the
# collection errors, the gaming markers, the contaminating files). They travel together or not at all:
# a block carrying results without their guards would let the App's C1 be LESS strict than the
# Action's. Anything the core adds later lands here by default -- the strict direction.
C1_EVIDENCE_KEYS = tuple(k for k in C1_KEYS_FROM_CORE if k not in _C1_SCALAR_KEYS and k not in _C1_PROSE_KEYS)
_APP_ONLY_KEYS = ("flaky_test_ids", "skips_added_for")
_PROBE_PER_TEST_KEYS = ("base_tests_on_head", "head_tests_on_head", "head_tests_on_base",
                        "suite_head_on_head", "suite_base_on_base", "subjects", "base_collection_errors")
_PROBE_LIST_KEYS = ("files", "changed_files", "suite_files", "deleted_files")


def _status(raw) -> str:
    """The status word of a per-test result -- the runner's `{status, phase?, type?, message?, origin?,
    entry?}` dict, or a bare word. Used here only to derive `flaky_test_ids`."""
    if isinstance(raw, Mapping):
        raw = raw.get("status")
    return str(raw)


def results_as_handed(results: Mapping | None) -> dict:
    """The three results maps travel EXACTLY as the Action hands them to its own core: the runner's
    per-test dicts, `origin` and `entry` included. The core's own-frame allowlist reads those two to
    affirm that a pytest witness's assertion fired in the test's own function; a status word alone made
    the hosted C1 refuse every honest witness -- UNPROVEN-contaminated on a PR the Action PROVED, which
    is an accusation of honest code -- so nothing is flattened here (the owner's decision on this
    lane's F2). A value outside the core's vocabulary passes through as it is: the core refuses it
    loudly, and nothing here spells it into a green."""
    return {str(tid): (dict(r) if isinstance(r, Mapping) else r) for tid, r in (results or {}).items()}


def c1_inputs_payload(*, base_sha: str | None, head_sha: str | None, inputs: Mapping, probe: Mapping,
                      markers_computed: bool) -> dict:
    """The block's content: the required C1 inputs (the two results maps as the runner's per-test dicts,
    exactly as handed to the core; empty when the Action never ran them), every optional C1 input the
    Action has a value for, under the core's names and shapes (the rerun map like its siblings; nothing
    renamed, nothing flattened), and
    `silencing_probe` (the core's probe structure for this run, as it is; its `state` is `measured` or
    one of the core's NOT_MEASURED words -- anything else is refused here, never rewritten). Two keys
    for the App's reader beside those: `flaky_test_ids` when a rerun of the reverted tree was made (a
    red that did not reproduce; ABSENT means flakiness was not assessed, never "none flaky") and
    `skips_added_for` when C2's gaming markers were computed without error (the tests or files the
    PR's own diff aims a skip at). `truncated` is written only by the composer, only when the block
    had to shrink."""
    state = probe.get("state") if isinstance(probe, Mapping) else None
    if state != "measured" and state not in NOT_MEASURED_REASONS:
        raise ValueError(f"silencing_probe['state'] is {state!r}, which is not one of the closed vocabulary "
                         f"{('measured',) + tuple(NOT_MEASURED_REASONS)}; the block carries the core's words only")
    payload: dict = {"base_sha": base_sha, "head_sha": head_sha}
    for k in C1_KEYS_FROM_CORE:
        if k in ("base_sha", "head_sha"):
            continue
        if k in _RESULTS_MAP_KEYS:
            if k in inputs or k in INPUT_SCHEMA["C1"]:
                payload[k] = results_as_handed(inputs.get(k))
        elif inputs.get(k) is not None:
            payload[k] = inputs[k]
    payload["silencing_probe"] = dict(probe)
    without = inputs.get("test_results_without_change") or {}
    rerun = inputs.get("test_results_without_change_rerun")
    if rerun is not None:
        payload["flaky_test_ids"] = sorted(t for t, r in without.items()
                                           if _status(r) in _RED and t in rerun and _status(rerun[t]) not in _RED)
    if markers_computed:
        markers = inputs.get("gaming_markers") or {}
        payload["skips_added_for"] = sorted(k for k, kind in markers.items() if kind in _SKIP_MARKERS)
    return payload


def _json_default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)


def render_c1_inputs_block(payload: Mapping) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default)
    # `<`, `>` and `&` can only occur inside JSON strings, and as \u escapes they decode to the same
    # characters -- so no `-->`, `<!--` or `--!>` exists before the terminator, and neither a browser
    # nor the App's non-greedy regex can close the comment early on a test id or a message.
    body = body.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f"{C1_INPUTS_MARKER_START} {body} {C1_INPUTS_MARKER_END}"


def _size(v) -> int:
    try:
        return len(v)
    except TypeError:
        return 1


def _reduce(payload: Mapping, level: int) -> dict:
    """Level 0: whole. 1: the probe's per-test maps dropped -- the probe becomes `no-report` (the core's
    own word for "no machine-readable outcome reached the reader"), its detail saying the Action DID
    measure it and why the maps are absent. 2: the prose (`execution_notes`) and the two App-only lists
    dropped. 3: every per-test `message` in the three results maps cut to its first line, 160
    characters -- exactly what the core prints (`Outcome.describe`); the core decides an assertion red
    on `type` before it reads the message, and `origin` / `entry` are untouched, so no verdict moves.
    4: the EVIDENCE SET dropped as one unit -- the results maps become empty and every guard goes with
    them, so the App's C1 has nothing to judge and says NOT_RUN, never a verdict from witnesses whose
    guards were cut. 5: the probe's file lists dropped as well; the block is then bounded by
    construction. Every drop is written under `truncated` with what it held."""
    p = dict(payload)
    if level <= 0:
        return p
    truncated: dict = {}
    probe = dict(p.get("silencing_probe") or {})
    dropped = {k: len(probe[k]) for k in _PROBE_PER_TEST_KEYS if isinstance(probe.get(k), Mapping) and probe[k]}
    if dropped:
        counts = ", ".join(f"{n} in {k}" for k, n in dropped.items())
        probe = {"state": "no-report",
                 "detail": (f"measured by the Action ({counts}), but the per-test maps did not fit the "
                            f"{CHECK_RUN_TEXT_LIMIT}-character check-run text this block travels in and were not "
                            f"published here; the Action's own C2 read them in full"),
                 **{k: probe[k] for k in _PROBE_LIST_KEYS if k in probe}}
        truncated["silencing_probe"] = dropped
    if level >= 2:
        for k in (*_C1_PROSE_KEYS, *_APP_ONLY_KEYS):
            if k in p:
                truncated[k] = _size(p[k])
                del p[k]
    if level >= 3:
        cut = 0
        for k in _RESULTS_MAP_KEYS:
            m = p.get(k)
            if not isinstance(m, Mapping):
                continue
            new = {}
            for tid, r in m.items():
                msg = r.get("message") if isinstance(r, Mapping) else None
                if isinstance(msg, str) and msg:
                    first = msg.strip().splitlines()[0][:_MESSAGE_KEEP] if msg.strip() else ""
                    if first != msg:
                        r = {**r, "message": first}
                        cut += 1
                new[tid] = r
            p[k] = new
        if cut:
            truncated["message"] = {"entries": cut, "kept": f"first line, {_MESSAGE_KEEP} characters (what the core prints)"}
    if level >= 4:
        for k in C1_EVIDENCE_KEYS:
            if k not in p:
                continue
            truncated[k] = _size(p[k])
            if k in INPUT_SCHEMA["C1"]:
                p[k] = {}                     # required: present and empty, which the core reads as "no tests"
            else:
                del p[k]
    if level >= 5:
        for k in _PROBE_LIST_KEYS:
            if k in probe:
                truncated.setdefault("silencing_probe", {})[k] = len(probe[k])
                del probe[k]
    p["silencing_probe"] = probe
    if truncated:
        p["truncated"] = truncated
    return p


def _notice(omitted: int, limit: int) -> str:
    return (f"\n\n[check-run text truncated: {omitted} character(s) of evidence omitted so the machine-readable C1 "
            f"inputs block below fits the {limit}-character limit of a check run's text; the receipt JSON, the step "
            f"summary and the PR comment carry the evidence in full]")


def _trim(evidence: str, room: int, limit: int) -> str:
    if len(evidence) <= room:
        return evidence
    keep = room - len(_notice(len(evidence), limit))       # the real count has no more digits than this one
    if keep <= 0:
        return ""
    cut = evidence[:keep]
    nl = cut.rfind("\n")
    if nl > keep // 2:
        cut = cut[:nl]
    cut = cut.rstrip()
    return cut + _notice(len(evidence) - len(cut), limit)


def _join(evidence: str, block: str) -> str:
    return f"{evidence}\n\n{block}\n" if evidence else f"{block}\n"


def check_run_text_with_c1_inputs(evidence: str, payload: Mapping, *, redact: Callable[[str], str],
                                  limit: int = CHECK_RUN_TEXT_LIMIT) -> str:
    """`evidence` (already redacted) + a blank line + the block, within `limit`, the block ALWAYS present
    and whole. When everything does not fit, in this order: the human text is trimmed (with a notice;
    the receipt JSON, the step summary and the PR comment carry it in full) but never below
    _EVIDENCE_RESERVE characters; then the probe's per-test maps go; then the prose and the App-only
    lists; then per-test messages are cut to what the core prints; then the evidence set as one unit
    (the App's C1 then says NOT_RUN); then the probe's file lists. `redact` runs over every rendered
    block, since redaction can lengthen text."""
    b0 = redact(render_c1_inputs_block(payload))
    if len(evidence) + (2 if evidence else 0) + len(b0) + 1 <= limit:
        return _join(evidence, b0)
    room = limit - len(b0) - 3
    if room >= _EVIDENCE_RESERVE:
        return _join(_trim(evidence, room, limit), b0)
    last = b0
    for level in (1, 2, 3, 4, 5):
        b = redact(render_c1_inputs_block(_reduce(payload, level)))
        if b == last:
            continue
        last = b
        room = limit - len(b) - 3
        if room >= min(len(evidence), _EVIDENCE_RESERVE):
            return _join(_trim(evidence, room, limit), b)
    room = max(0, limit - len(last) - 3)
    return _join(_trim(evidence, room, limit), last)
