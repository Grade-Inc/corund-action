"""C4 — approval-SHA binding. A merge word binds to a SHA, not a PR: an approval older
than the head SHA fails the check.

    PROVEN   a reviewer whose LATEST review is APPROVED binds to EXACTLY the head SHA (case-insensitive
             full-string compare; never a prefix match), and is not the PR's own author or a bot
    FAILED   an approval EXISTS and does not bind: every approval is stale (a post-approval push
             without re-approval — a force-push, a rebase, or any later push to the same PR),
             dismissed, superseded by a later CHANGES_REQUESTED/COMMENTED from the same reviewer,
             missing its SHA, a prefix, the only binding approver is the PR author (`author_id`
             supplied), or the only binding approver is a bot (`is_bot` supplied on its record)
    NOT_RUN  ZERO approvals submitted (R4 RULING 1, owner, 2026-09-08 — REVERSES the prior design).
             "No approvals yet" is not a failure: nothing has been gamed and no approval has gone
             stale, there is simply nothing yet to judge — the same absence-is-NOT_RUN move as
             `c3_gate_fold.py`'s zero-required-checks fix, and for the same adoption reason: a
             freshly-opened, unreviewed PR must not show a red check before anyone has reviewed it.
             THE BOUNDARY, load-bearing: this is ONLY the empty-list case. The moment a single review
             exists — however it turns out — this check resumes ordinary judgement, and a review
             that exists but does not bind is still FAILED, exactly as before. Also (runner) no
             head_sha and no merge_sha
    CRASHED  (runner) an unknown review state, a malformed approval record; or (here) a malformed
             `is_bot`

Reviews are ordered by `submitted_at` when every record carries one, else by list order (the
caller passes them in submission order). WHICHEVER ORDERING WAS USED IS NAMED ON THE RECEIPT, and
`submitted_at` is PARSED before it is compared — see `_parsed_order`. NEW module.

BOT APPROVALS (R4 RULING 3, owner, 2026-09-08 — a known limitation, fixed here because it was
cheap: no new required input, no shared-schema change, contained entirely in this module).
`approvals[i].is_bot` is an OPTIONAL bool. Absent (the default; every existing caller is
unaffected) means "not known to be a bot" — this check never GUESSES bot-ness from a reviewer id
that merely looks like one (`dependabot[bot]` supplied with no `is_bot` still binds, because the
core does not decide a fact only the adapter can observe — the guard rule). When the
caller DOES supply `is_bot: true` on the binding review(s), a bot-only binding is excluded exactly
like an author-only one, and a genuine human co-approval still binds."""
from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from typing import Any

from .verdict import Verdict

REVIEW_STATES = ("APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED", "PENDING")


def _parse_submitted_at(raw: Any, i: int) -> _dt.datetime:
    """`submitted_at` as an INSTANT, or a refusal naming the record and the value.

    EIGHTH CYCLE (verifier 7, F-C4 — the first finding ever raised against C4). This check used to
    order reviews with `sort(key=lambda rec: (str(rec["at"]), rec["i"]))`: a LEXICAL sort of
    unvalidated free text, standing in for the documented "in submission order" contract. Same
    submission order, same verdict-relevant content, only the timestamp populated, and the verdict
    FLIPPED — a reviewer who APPROVED and then requested changes was read as approving, because
    `2026-01-01T21:00:00-05:00` (02:00Z the next day, the truly later review) sorts lexically BEFORE
    `2026-01-01T23:00:00+00:00`. Free text sorted too ('i' > '9'); so did integers (`str(10)` <
    `str(9)`). Nothing validated the format and nothing refused an unparseable one, so C4's
    guarantee held only because GitHub's API happens to emit a sortable form — a guard backed by the
    adapter behaving, which is precisely the state the seventh cycle's F8 fix existed to END one
    check over.

    STRICT on purpose, and narrow on purpose:

      * only a `str` is a timestamp. An int is refused rather than read as an epoch, because
        reading it means GUESSING a unit — seconds, milliseconds, a sequence number — and C4 has a
        verdict for "I could not check" that is not a guess;
      * `datetime.fromisoformat` is the whole grammar (it takes GitHub's `...Z` wire format and
        every offset spelling beside it). Anything it rejects is unparseable, and unparseable is
        CRASHED with the value quoted, never a silent fallback to the old lexical order — a
        fallback would restore the defect for exactly the inputs that provoke it;
      * the SHA comparison one screen down is a full-string compare, never a prefix. The ordering
        is held to the same standard: no partial credit for a value that merely looks like a date."""
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise ValueError(f"approvals[{i}].submitted_at={raw!r} is of type {type(raw).__name__}, not an ISO-8601 "
                         f"timestamp string. Reviews are ordered by this field, so a value C4 cannot parse "
                         f"is a verdict C4 cannot reach — supply an ISO-8601 string, or omit the field on "
                         f"every record to be ordered by submission order instead")
    try:
        return _dt.datetime.fromisoformat(raw.strip())
    except ValueError as e:
        raise ValueError(f"approvals[{i}].submitted_at={raw!r} is not an ISO-8601 timestamp ({e}). Reviews are "
                         f"ordered by this field, so a value C4 cannot parse is a verdict C4 cannot reach — "
                         f"supply an ISO-8601 string, or omit the field on every record to be ordered by "
                         f"submission order instead") from None


def _parsed_order(records: list[dict]) -> str:
    """Sort `records` into submission order IN PLACE, and return the ordering's name for the receipt.

    `submitted_at` decides only when EVERY record carries one — the documented contract, unchanged.
    A set that mixes offset-aware and naive timestamps has NO total order (Python raises on the
    comparison, and defaulting the naive ones to UTC would invent a fact), so it is refused rather
    than ordered. `rec["i"]` breaks ties, which keeps two reviews at the same instant in the
    submission order they arrived in instead of letting the sort decide the verdict."""
    if not records or not all(rec["at"] is not None for rec in records):
        return "submission order (the order the caller supplied; `submitted_at` absent on at least one review)"
    parsed = [_parse_submitted_at(rec["at"], rec["i"]) for rec in records]
    aware = {p.utcoffset() is not None for p in parsed}
    if len(aware) > 1:
        raise ValueError("approvals mix offset-aware and naive `submitted_at` values, which have no total "
                         "order — assuming UTC for the naive ones would invent the fact the ordering turns "
                         "on. Supply an offset on every review, or on none")
    for rec, p in zip(records, parsed):
        rec["ts"] = p
    records.sort(key=lambda rec: (rec["ts"], rec["i"]))
    kind = "offset-aware" if True in aware else "naive"
    return f"`submitted_at`, parsed as ISO-8601 ({kind}) and compared as instants, ties in submission order"


def _norm_sha(s: Any) -> str | None:
    if s is None:
        return None
    s = str(s).strip().lower()
    return s or None


def check_c4(inputs: Mapping[str, Any]) -> Verdict:
    approvals = inputs["approvals"]
    if not isinstance(approvals, (list, tuple)):
        raise TypeError(f"approvals must be a list of review records, got {type(approvals).__name__}")
    head_raw = inputs.get("head_sha")
    target_key = "head_sha" if head_raw is not None else "merge_sha"
    target = _norm_sha(head_raw if head_raw is not None else inputs.get("merge_sha"))
    if not target:
        raise ValueError(f"{target_key} is empty")
    author = inputs.get("author_id")
    author_s = None if author is None else str(author)

    # R4 RULING 1 (owner, 2026-09-08). THE BOUNDARY: this is the ONLY place this check returns
    # NOT_RUN for approvals -- it fires solely on a literally EMPTY list, before a single record is
    # built. Any non-empty `approvals`, however every record turns out (stale, dismissed, bot-only,
    # self-only), falls through to the ordinary comparison below and can still be FAILED.
    if len(approvals) == 0:
        return Verdict("C4", "NOT_RUN", None, target, None,
                       (f"C4 approval-SHA binding: 0 reviews submitted; binding target {target_key} {target[:12]}",
                        "NOT_RUN: no approvals at all — there is nothing yet to judge as bound or "
                        "stale. Not a failure: nobody has reviewed this PR, so nothing has been "
                        "gamed and no approval has gone stale. The moment any review is submitted, "
                        "this check resumes its ordinary judgement, and a review that exists but "
                        "does not bind is still FAILED, exactly as before."))

    records = []
    for i, r in enumerate(approvals):
        if not isinstance(r, Mapping):
            raise TypeError(f"approvals[{i}] must be a mapping, got {type(r).__name__}")
        if r.get("reviewer_id") is None:
            raise ValueError(f"approvals[{i}] has no reviewer_id")
        state = str(r.get("state", "")).strip().upper()
        if state not in REVIEW_STATES:
            raise ValueError(f"approvals[{i}].state={r.get('state')!r} is not one of {REVIEW_STATES}")
        raw_bot = r.get("is_bot")
        if raw_bot is not None and not isinstance(raw_bot, bool):
            raise ValueError(f"approvals[{i}].is_bot={raw_bot!r} must be a bool or absent, not "
                             f"{type(raw_bot).__name__} — a value that merely looks truthy is not read as one")
        records.append({"i": i, "reviewer": str(r["reviewer_id"]), "sha": _norm_sha(r.get("commit_sha")),
                        "state": state, "at": r.get("submitted_at"), "is_bot": bool(raw_bot)})
    ordering = _parsed_order(records)

    latest: dict[str, dict] = {}
    for rec in records:
        latest[rec["reviewer"]] = rec

    head_label = f"{target_key} {target[:12]}"
    ev = [f"C4 approval-SHA binding: {len(records)} review(s) from {len(latest)} reviewer(s); binding target {head_label}",
          # Which ordering decided "latest" is the whole verdict when one reviewer reviewed twice, so it is
          # PRINTED rather than assumed (a success message is not evidence that anything happened).
          f"  ordered by: {ordering}"]
    binding: list[dict] = []
    for reviewer, rec in latest.items():
        if rec["state"] != "APPROVED":
            ev.append(f"  reviewer {reviewer}: latest review is {rec['state']} at {(rec['sha'] or 'no sha')[:12]} — not an approval")
            continue
        if not rec["sha"]:
            ev.append(f"  reviewer {reviewer}: APPROVED but the review carries no commit_sha — cannot bind")
            continue
        if rec["sha"] == target:
            ev.append(f"  reviewer {reviewer}: APPROVED at {rec['sha'][:12]} == {head_label} — binds")
            binding.append(rec)
        elif target.startswith(rec["sha"]) or rec["sha"].startswith(target):
            ev.append(f"  reviewer {reviewer}: APPROVED at {rec['sha']} is a prefix of / prefixed by the target — "
                      f"binding requires the exact full SHA")
        else:
            ev.append(f"  reviewer {reviewer}: APPROVED at {rec['sha'][:12]} != {head_label} — a post-approval push landed "
                      f"without re-approval (stale word)")

    if binding and author_s is not None:
        non_author = [b for b in binding if b["reviewer"] != author_s]
        if not non_author:
            ev.append(f"  the only binding approval is by the PR author ({author_s}) — self-approval does not count")
            binding = []
        elif len(non_author) < len(binding):
            ev.append(f"  author {author_s}'s own approval ignored; {len(non_author)} other binding approval(s) remain")
            binding = non_author

    # R4 RULING 3 (owner, 2026-09-08). A bot's approval is excluded exactly like an author's own —
    # only when the CALLER supplied `is_bot: true` on it (see the module docstring: absence is never
    # read as "not a bot" being decided FOR the caller by a name that merely looks like one).
    if binding:
        non_bot = [b for b in binding if not b["is_bot"]]
        if not non_bot:
            who = ", ".join(b["reviewer"] for b in binding)
            ev.append(f"  the only binding approval(s) are from bot account(s) ({who}) — a bot's "
                      f"approval does not count as reviewer sign-off")
            binding = []
        elif len(non_bot) < len(binding):
            bots = ", ".join(b["reviewer"] for b in binding if b["is_bot"])
            ev.append(f"  bot approval(s) ignored ({bots}); {len(non_bot)} other binding approval(s) remain")
            binding = non_bot

    if binding:
        who = ", ".join(b["reviewer"] for b in binding)
        ev.append(f"PROVEN: {len(binding)} non-dismissed APPROVED review(s) bind to the exact {head_label}: reviewer(s) {who}")
        return Verdict("C4", "PROVEN", None, target, target, tuple(ev))
    n_appr = sum(1 for rec in latest.values() if rec["state"] == "APPROVED")
    ev.append(f"FAILED: no APPROVED review binds to {head_label} ({n_appr} approval(s) examined, 0 binding; "
              f"{len(records)} review(s) total)")
    return Verdict("C4", "FAILED", None, target, None, tuple(ev))
