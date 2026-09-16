"""C3 — gate-fold: did the PR fold a gate? Branch protection before/after, plus every workflow
file before/after, compared structurally.

Design ruling (2026-09-05): C3 FAILS only on the MECHANICALLY-CERTAIN classes —
    * a required check context removed from (or renamed in) branch protection
    * the gate job that produces a required context deleted or renamed away (or its workflow
      file deleted)
    * `continue-on-error: true` added to a gate job
    * `if: <constant false>` added to a gate job (false / 0 / ${{ false }} / quoted forms)
— and REPORTS everything else as `OBSERVATION` lines on a PROVEN receipt: strict-mode off, review
count lowered, dismiss-stale off, enforce-admins off, force-push/deletion allowed, a non-constant
`if:` added or changed on a gate job, step-level continue-on-error, a paths filter added to a
gating trigger, a gate job losing steps, a required context no supplied workflow produces.

Context mapping: a workflow job produces the context "<workflow name> / <job display name or key>"
(matrix variants start with that prefix and " ("). A bare job name is accepted too. Workflow text
is parsed by corund_checks.workflow_yaml (stdlib); a file it cannot parse is CRASHED, never
skipped. NEW module.

NOT_RUN when zero required contexts exist on BOTH the base and the head (R4 fix, 2026-09-08): a
repo that has never turned on branch protection has no gate for a PR to fold, so C3 has nothing to
compare and says so rather than FAILING an ordinary PR for a gate it never had. When the base has
zero required contexts but the head gains some (the PR turns protection ON), that is not a fold
either and the ordinary comparison below reaches PROVEN, naming the gained context(s).

A required context's rename is TRACED, not folded, when the SAME workflow job (by (path, key)
identity, not by name) persists into the after-tree and now produces a context protection newly
requires in the same PR (R4 fix, 2026-09-08) — a job renamed and its protection entry updated
together, correctly, in one change. The traced job is still held to the full job-level standard
(continue-on-error, if:, steps, triggers); tracing a rename is never a way to also hide a fold.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .verdict import Verdict
from .workflow_yaml import Job, Workflow, is_constant_false, parse_workflow

GATING_EVENTS = ("pull_request", "push", "merge_group")


def _required_contexts(prot: Any, label: str) -> list[str]:
    if prot is None:
        return []
    if not isinstance(prot, Mapping):
        raise TypeError(f"{label} must be a mapping (GitHub branch-protection shape) or None, got {type(prot).__name__}")
    rsc = prot.get("required_status_checks")
    if rsc is None:
        return []
    if not isinstance(rsc, Mapping):
        raise TypeError(f"{label}.required_status_checks must be a mapping or None")
    out: list[str] = []
    for c in rsc.get("contexts") or []:
        out.append(str(c))
    for c in rsc.get("checks") or []:
        if isinstance(c, Mapping) and c.get("context") is not None:
            out.append(str(c["context"]))
        elif isinstance(c, str):
            out.append(c)
    seen: list[str] = []
    for c in out:
        if c not in seen:
            seen.append(c)
    return seen


def _flag(prot: Any, *path: str, default=None):
    cur = prot
    for p in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


def _protection_observations(before: Any, after: Any) -> list[str]:
    obs: list[str] = []
    b_strict, a_strict = _flag(before, "required_status_checks", "strict"), _flag(after, "required_status_checks", "strict")
    if b_strict is True and a_strict is not True:
        obs.append("required_status_checks.strict turned off — branches no longer need to be up to date with the base")
    b_adm, a_adm = _flag(before, "enforce_admins", "enabled"), _flag(after, "enforce_admins", "enabled")
    if b_adm is True and a_adm is not True:
        obs.append("enforce_admins turned off — administrators may bypass the gates")
    b_rev, a_rev = _flag(before, "required_pull_request_reviews"), _flag(after, "required_pull_request_reviews")
    if b_rev is not None and a_rev is None:
        obs.append("required_pull_request_reviews removed")
    else:
        b_n = _flag(before, "required_pull_request_reviews", "required_approving_review_count", default=0)
        a_n = _flag(after, "required_pull_request_reviews", "required_approving_review_count", default=0)
        if isinstance(b_n, int) and isinstance(a_n, int) and a_n < b_n:
            obs.append(f"required_approving_review_count lowered {b_n} -> {a_n}")
        b_d = _flag(before, "required_pull_request_reviews", "dismiss_stale_reviews")
        a_d = _flag(after, "required_pull_request_reviews", "dismiss_stale_reviews")
        if b_d is True and a_d is not True:
            obs.append("dismiss_stale_reviews turned off — an approval now survives a new push (see C4)")
    for key, what in (("allow_force_pushes", "force pushes"), ("allow_deletions", "branch deletion")):
        if _flag(before, key, "enabled") is not True and _flag(after, key, "enabled") is True:
            obs.append(f"{key} turned on — {what} now allowed on the protected branch")
    b_c, a_c = _flag(before, "required_conversation_resolution", "enabled"), _flag(after, "required_conversation_resolution", "enabled")
    if b_c is True and a_c is not True:
        obs.append("required_conversation_resolution turned off")
    return obs


def _parse_all(files: Any, label: str) -> dict[str, Workflow]:
    if not isinstance(files, Mapping):
        raise TypeError(f"{label} must be a mapping of path -> text, got {type(files).__name__}")
    out: dict[str, Workflow] = {}
    for path, text in files.items():
        if not isinstance(text, str):
            raise TypeError(f"{label}[{path!r}] must be str")
        try:
            out[str(path)] = parse_workflow(text)
        except Exception as exc:
            raise ValueError(f"{label}[{path}] could not be parsed as a workflow: {exc}") from exc
    return out


def _context_map(wfs: Mapping[str, Workflow]) -> dict[str, list[tuple[str, str]]]:
    """context name -> [(path, job_key)] for every context the supplied workflows can produce."""
    m: dict[str, list[tuple[str, str]]] = {}
    for path, wf in wfs.items():
        for key, job in wf.jobs.items():
            for ctx in job.contexts:
                for name in ((f"{wf.name} / {ctx}",) if wf.name else ()) + (ctx,):
                    m.setdefault(name, []).append((path, key))
    return m


def _lookup(cmap: Mapping[str, list[tuple[str, str]]], ctx: str) -> list[tuple[str, str]]:
    if ctx in cmap:
        return cmap[ctx]
    hits = []
    for name, locs in cmap.items():
        if ctx.startswith(name + " ("):
            hits.extend(locs)
    return hits


def _names_for(wf: Workflow, key: str) -> list[str]:
    """Every context name the job `key` in `wf` can produce today — the same naming scheme
    `_context_map` uses, but for one job, so a traced rename can ask "what does THIS job produce
    now" without rebuilding the whole map."""
    job = wf.jobs.get(key)
    if job is None:
        return []
    out: list[str] = []
    for ctx in job.contexts:
        if wf.name:
            out.append(f"{wf.name} / {ctx}")
        out.append(ctx)
    return out


def _job_level_checks(ctx_label: str, path: str, key: str, job_a: Job, job_b: Job | None,
                      wf_after: Mapping[str, Workflow], wf_before: Mapping[str, Workflow],
                      folds: list[str], obs: list[str]) -> None:
    """The mechanically-certain / observation checks for ONE gate job, before vs after. Shared by
    the direct-match path and the traced-rename path (R4 fix) so a legitimately renamed job is held
    to exactly the same standard as one whose name never changed."""
    if job_a.continue_on_error and not (job_b and job_b.continue_on_error):
        folds.append(f"gate job {path}:{key} (`{ctx_label}`): continue-on-error: true added (any constant spelling: true / "
                     f"True / 'true' / ${{{{ true }}}}) — the job cannot fail")
    if job_a.continue_on_error_expr and not (job_b and job_b.continue_on_error_expr == job_a.continue_on_error_expr):
        obs.append(f"gate job {path}:{key} (`{ctx_label}`): continue-on-error: {job_a.continue_on_error_expr} — an expression, "
                   f"not a constant; when it evaluates true the job cannot fail; not mechanically certain, verify by hand")
    if is_constant_false(job_a.if_expr) and not (job_b and is_constant_false(job_b.if_expr)):
        folds.append(f"gate job {path}:{key} (`{ctx_label}`): if: {job_a.if_expr} added — the job never runs, and a skipped "
                     f"required check counts as passing")
    elif job_a.if_expr is not None and (job_b is None or job_b.if_expr != job_a.if_expr):
        obs.append(f"gate job {path}:{key} (`{ctx_label}`): if: {job_a.if_expr} added/changed — a condition that skips a "
                   f"required check makes it count as passing; not mechanically certain, verify by hand")
    new_step_coe = [i for i in job_a.step_continue_on_error if not job_b or i not in job_b.step_continue_on_error]
    if new_step_coe:
        obs.append(f"gate job {path}:{key} (`{ctx_label}`): continue-on-error: true on step(s) {new_step_coe} — "
                   f"a step that cannot fail inside a gate job")
    if job_b and job_a.step_count < job_b.step_count:
        obs.append(f"gate job {path}:{key} (`{ctx_label}`): steps {job_b.step_count} -> {job_a.step_count}")
    wa, wb = wf_after[path], wf_before.get(path)
    for evn in wa.paths_filtered_events:
        if evn in GATING_EVENTS and (wb is None or evn not in wb.paths_filtered_events):
            obs.append(f"{path}: `{evn}` trigger gained a paths filter — the required check will not report on "
                       f"PRs outside those paths (stalls or, with paths-ignore on push, never runs)")
    if wb is not None:
        lost = sorted((wb.events & set(GATING_EVENTS)) - wa.events)
        if lost:
            obs.append(f"{path}: gating trigger(s) removed: {', '.join(lost)}")


def check_c3(inputs: Mapping[str, Any]) -> Verdict:
    p_before, p_after = inputs["protection_before"], inputs["protection_after"]
    wf_before = _parse_all(inputs["workflow_files_before"], "workflow_files_before")
    wf_after = _parse_all(inputs["workflow_files_after"], "workflow_files_after")
    override = inputs.get("required_contexts")

    before_ctx = _required_contexts(p_before, "protection_before")
    after_ctx = _required_contexts(p_after, "protection_after")
    if override is not None:
        if not isinstance(override, (list, tuple)):
            raise TypeError("required_contexts must be a list of context names")
        before_ctx = [str(c) for c in override]
        after_ctx = list(before_ctx)
        src = "caller override"
    else:
        src = "protection_before"

    ev: list[str] = [f"C3 gate-fold: {len(before_ctx)} required context(s) from {src}: "
                     f"{', '.join(before_ctx) if before_ctx else '(none)'}; workflow files: "
                     f"{len(wf_before)} before / {len(wf_after)} after"]

    # R4 FIX (2026-09-08, this lane). Formerly: an empty `before_ctx` unconditionally appended a
    # FOLD and the run came back FAILED — an accusation with nothing behind it whenever the base
    # branch had never had branch protection, AND (a distinct bug) whenever a PR ADDED protection
    # for the first time (after_ctx non-empty), which is the opposite of a fold. FAILED is a
    # FLAGGING state (verdict.FLAGGING_STATES: "this check would have flagged the PR"); there is no
    # fold to flag when there was never a gate. Two outcomes now, not one:
    #   * before AND after both carry zero required contexts -> NOT_RUN. Nothing gates on either
    #     side, so gate-fold has no comparison to make (a result on a path that compared
    #     nothing is the absence of a question, not the answer to one).
    #   * before is empty but after is not -> falls through to the normal comparison below, which
    #     reports the gained context(s) and reaches PROVEN; a PR turning protection ON is not folding
    #     a gate it never had.
    if not before_ctx and not after_ctx:
        ev.append("NOT_RUN: no required status checks are configured on the base or the head — there is "
                  "no gate for this PR to have folded, so gate-fold has nothing to compare. A repo that "
                  "has never turned on branch protection is not accused of removing a gate it never had.")
        return Verdict("C3", "NOT_RUN", None, None, None, tuple(ev))

    folds: list[str] = []
    obs: list[str] = []
    cmap_b, cmap_a = _context_map(wf_before), _context_map(wf_after)

    # (1) contexts removed / renamed in protection — TRACED before being folded (R4 fix, 2026-09-08).
    #
    # Formerly: any before_ctx entry absent from after_ctx was folded, full stop — so a LEGITIMATE
    # rename (a job's `name:` changed, and protection was updated in the SAME PR to require the new
    # name) was indistinguishable from one done wrong (the job renamed away, leaving a stale required
    # context nothing can ever satisfy). Both produced "removed from protection (or renamed)". That is
    # exactly the false-positive the decidability rule exists to close: whether the SAME
    # workflow job (identified by its (path, key) — not by name, which is the very thing that moved)
    # persisted into the after-tree and now produces a context that protection newly requires is
    # MECHANICALLY DECIDABLE from the inputs C3 already has; it needs no new input surface.
    #
    # A rename is TRACED, not folded, only when: the job that produced the old context in `wf_before`
    # still exists at the same (path, key) in `wf_after`, AND it now produces some context that (a) IS
    # in `after_ctx` and (b) was NOT already a before_ctx entry (so it is the "gained" side of this
    # exact rename, not a coincidence with an unrelated already-required name). The traced job is then
    # held to the FULL job-level standard below (continue-on-error, if:, steps, triggers) exactly like
    # an unrenamed one — a rename is not a way to also sneak a fold past this check.
    traced: dict[str, tuple[str, str, str]] = {}   # old_ctx -> (new_ctx, path, key)
    for ctx in before_ctx:
        if ctx in after_ctx:
            continue
        hit = None
        for path, key in _lookup(cmap_b, ctx):
            if path in wf_after and key in wf_after[path].jobs:
                candidates = [n for n in _names_for(wf_after[path], key) if n in after_ctx and n not in before_ctx]
                if candidates:
                    hit = (candidates[0], path, key)
                    break
        if hit:
            traced[ctx] = hit
        else:
            folds.append(f"required check `{ctx}` removed from protection (or renamed — no context of that name remains)")

    renamed_targets = {new for new, _p, _k in traced.values()}
    added = [c for c in after_ctx if c not in before_ctx and c not in renamed_targets]
    if added:
        ev.append(f"  protection gained required context(s): {', '.join(added)}")
    for old, (new, path, key) in traced.items():
        ev.append(f"  RENAMED (traced): `{old}` -> `{new}` — same job {path}:{key} persisted across the change, and "
                  f"protection was updated to require the new name in the same PR; not a fold")

    # (2)-(4) gate jobs in the workflow files: direct-name matches, exactly as before.
    for ctx in before_ctx:
        if ctx not in after_ctx:
            continue
        locs_b = _lookup(cmap_b, ctx)
        locs_a = _lookup(cmap_a, ctx)
        if not locs_b:
            obs.append(f"`{ctx}`: no workflow file supplied produces it (external app or matrix job) — job not compared")
            continue
        if not locs_a:
            missing_files = sorted({p for p, _ in locs_b if p not in wf_after})
            if missing_files:
                folds.append(f"gate for `{ctx}`: workflow file {', '.join(missing_files)} deleted")
            else:
                was = ", ".join(f"{p}:{k}" for p, k in locs_b)
                folds.append(f"gate job for `{ctx}` deleted or renamed away (was {was}) — the required context can never report")
            continue
        for path, key in locs_a:
            job_a: Job = wf_after[path].jobs[key]
            job_b: Job | None = wf_before[path].jobs.get(key) if path in wf_before else None
            _job_level_checks(ctx, path, key, job_a, job_b, wf_after, wf_before, folds, obs)
            ev.append(f"  `{ctx}` <- {path}:{key}: compared")

    # (2)-(4) again, for each TRACED rename: the SAME standard, keyed by the (path, key) the trace
    # already identified rather than by a name lookup (the name changed; the job identity did not).
    for old, (new, path, key) in traced.items():
        job_a = wf_after[path].jobs[key]
        job_b = wf_before[path].jobs.get(key)
        _job_level_checks(f"{old}` -> `{new}", path, key, job_a, job_b, wf_after, wf_before, folds, obs)
        ev.append(f"  `{old}` -> `{new}` <- {path}:{key}: compared (traced rename)")

    obs.extend(_protection_observations(p_before, p_after) if override is None else [])
    for o in obs:
        ev.append(f"  OBSERVATION: {o}")
    for f in folds:
        ev.append(f"  FOLD: {f}")

    if folds:
        ev.append(f"FAILED: {len(folds)} fold(s) in the mechanically-certain classes; {len(obs)} observation(s)")
        return Verdict("C3", "FAILED", None, None, None, tuple(ev))
    ev.append(f"PROVEN: 0 folds in the mechanically-certain classes across {len(before_ctx)} required context(s); "
              f"{len(obs)} observation(s) reported, not blocked")
    return Verdict("C3", "PROVEN", None, None, None, tuple(ev))
