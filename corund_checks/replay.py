"""Replay — run ALL checks over a repo's merged PRs and report "would have flagged" per rule with
true/false-positive accounting (the onboarding centrepiece, before any enforcement can
be switched on). Pure: the caller (the Action's `corund replay` CLI, or the app on install) builds
one ReplayItem per merged PR from git history and read-only API calls; this module runs
`run_check` per available input, never raises, posts nothing, and renders the report.

A check the caller could not build inputs for is NOT_RUN with the caller's stated reason — Corund
never omits a check it did not run. `redact_verdict` is the hash-and-suppress redactor for results
files that may carry commit SHAs only (evidence lines are replaced by their sha256; finding KINDS,
which are Corund vocabulary, survive as counts). NEW module.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .runner import run_check
from .verdict import CHECK_IDS, CHECK_NAMES, Verdict

MARKS = ("tp", "fp")
_KIND_RE = re.compile(r"\[[A-D]/([a-z0-9-]+)\]")


@dataclass(frozen=True)
class ReplayItem:
    pr_id: str
    base_sha: str | None
    head_sha: str | None
    inputs: Mapping[str, Any]                          # check id -> input snapshot
    not_run_reasons: Mapping[str, str] = field(default_factory=dict)
    crashed_reasons: Mapping[str, str] = field(default_factory=dict)   # the gatherer's own exception text
    meta: Mapping[str, Any] = field(default_factory=dict)   # informational only; never decides a verdict


@dataclass(frozen=True)
class ReplayRow:
    pr_id: str
    base_sha: str | None
    head_sha: str | None
    verdicts: tuple[Verdict, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def verdict(self, check: str) -> Verdict:
        for v in self.verdicts:
            if v.check == check:
                return v
        raise KeyError(check)


@dataclass(frozen=True)
class RuleAccount:
    check: str
    total: int
    ran: int
    flagged: int
    proven: int
    unproven: int
    not_run: int
    crashed: int
    true_positive: int
    false_positive: int
    unmarked: int

    @property
    def false_positive_rate(self) -> float | None:
        marked = self.true_positive + self.false_positive
        return None if marked == 0 else self.false_positive / marked


@dataclass(frozen=True)
class ReplayReport:
    rows: int
    accounts: tuple[RuleAccount, ...]
    flagged_prs: dict[str, tuple[str, ...]]


def replay_one(item: ReplayItem) -> ReplayRow:
    verdicts: list[Verdict] = []
    inputs = item.inputs if isinstance(item.inputs, Mapping) else {}
    reasons = item.not_run_reasons if isinstance(item.not_run_reasons, Mapping) else {}
    crashed = item.crashed_reasons if isinstance(item.crashed_reasons, Mapping) else {}
    for check in CHECK_IDS:
        snap = inputs.get(check) if isinstance(inputs, Mapping) else None
        if check in crashed:
            text = str(crashed[check]) or "gatherer crashed without a message"
            verdicts.append(Verdict(check, "CRASHED", item.base_sha, item.head_sha, None,
                                    (f"CRASHED while gathering inputs: {text}",), error=text))
            continue
        if snap is None:
            why = reasons.get(check) or "inputs not built by the caller for this PR"
            verdicts.append(Verdict(check, "NOT_RUN", item.base_sha, item.head_sha, None,
                                    (f"NOT_RUN: {why}",)))
            continue
        verdicts.append(run_check(check, snap))
    return ReplayRow(item.pr_id, item.base_sha, item.head_sha, tuple(verdicts), dict(item.meta or {}))


def replay(items: Iterable[ReplayItem]) -> tuple[ReplayRow, ...]:
    return tuple(replay_one(it) for it in items)


def summarize(rows: Iterable[ReplayRow], marks: Mapping[str, Mapping[str, str]] | None = None) -> ReplayReport:
    rows = tuple(rows)
    marks = marks or {}
    for pr, per_check in marks.items():
        for check, mark in per_check.items():
            if mark not in MARKS:
                raise ValueError(f"mark for {pr}/{check} must be one of {MARKS}, got {mark!r}")
    accounts = []
    flagged_prs: dict[str, tuple[str, ...]] = {}
    for check in CHECK_IDS:
        vs = [(r.pr_id, r.verdict(check)) for r in rows]
        flagged = [pr for pr, v in vs if v.flagged]
        tp = sum(1 for pr in flagged if marks.get(pr, {}).get(check) == "tp")
        fp = sum(1 for pr in flagged if marks.get(pr, {}).get(check) == "fp")
        accounts.append(RuleAccount(
            check=check, total=len(vs),
            ran=sum(1 for _, v in vs if v.state not in ("NOT_RUN", "CRASHED")),
            flagged=len(flagged),
            proven=sum(1 for _, v in vs if v.state == "PROVEN"),
            unproven=sum(1 for _, v in vs if v.state == "UNPROVEN"),
            not_run=sum(1 for _, v in vs if v.state == "NOT_RUN"),
            crashed=sum(1 for _, v in vs if v.state == "CRASHED"),
            true_positive=tp, false_positive=fp, unmarked=len(flagged) - tp - fp,
        ))
        flagged_prs[check] = tuple(flagged)
    return ReplayReport(rows=len(rows), accounts=tuple(accounts), flagged_prs=flagged_prs)


def render_report(report: ReplayReport, *, tree_sha: str, repo_label: str = "repo") -> str:
    lines = [f"tree {tree_sha} — corund replay over {report.rows} merged PR(s) of {repo_label}",
             "posture: read-only; nothing was posted. A rule would have flagged a PR when its verdict is "
             "FAILED or GAMED-SUSPECT. Mark each flagged PR true/false positive before enabling BLOCK for a rule.",
             ""]
    for a in report.accounts:
        rate = "n/a (nothing marked)" if a.false_positive_rate is None else f"{a.false_positive_rate:.0%}"
        lines.append(f"{a.check} {CHECK_NAMES[a.check]}: ran {a.ran}/{a.total}; would have flagged {a.flagged}; "
                     f"PROVEN {a.proven}; UNPROVEN {a.unproven}; NOT_RUN {a.not_run}; CRASHED {a.crashed}; "
                     f"marked true positive {a.true_positive}, false positive {a.false_positive}, unmarked {a.unmarked}; "
                     f"false positive rate {rate}")
        if report.flagged_prs.get(a.check):
            lines.append(f"    flagged: {', '.join(report.flagged_prs[a.check])}")
    return "\n".join(lines) + "\n"


def rows_to_dicts(rows: Iterable[ReplayRow]) -> Iterable[dict]:
    for r in rows:
        yield {"pr_id": r.pr_id, "base_sha": r.base_sha, "head_sha": r.head_sha,
               "verdicts": [v.to_dict() for v in r.verdicts], "meta": dict(r.meta)}


def redact_verdict(v: Verdict) -> Verdict:
    """Hash-and-suppress: every evidence line becomes its sha256 (verifiable later against the real
    line, revealing nothing); the finding kinds seen are kept as counts. State/reason/SHAs stay."""
    kinds: dict[str, int] = {}
    for line in v.evidence:
        for k in _KIND_RE.findall(line):
            kinds[k] = kinds.get(k, 0) + 1
    hashed = tuple("sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest()[:24] for line in v.evidence)
    head = [f"evidence redacted: {len(v.evidence)} line(s) hashed (hash-and-suppress)"]
    if kinds:
        head.append("finding kinds: " + ", ".join(f"{k} x{n}" for k, n in sorted(kinds.items())))
    error = None
    if v.error is not None:
        error = "redacted: " + hashlib.sha256(v.error.encode("utf-8")).hexdigest()[:24]
    return Verdict(v.check, v.state, v.base_sha, v.head_sha, v.approval_sha, tuple(head) + hashed, error, v.reason)
