"""The Action's orchestrator: `corund run` on one pull request.

    C1  the revert-run (c1_runner)  +  the contamination set from the PR's own diff -> run_check("C1")

ONE CHECK. The owner's decision of 2026-09-15: Corund ships C1 alone, and C1 does not ACCUSE.
"Require the PR's new tests to fail on the old code."

The reason is measured, not stylistic. The field sweep of 1,079 merged PRs across 27 repositories
put C2's false-accusation rate at 6.58% and its accusatory precision at 12.3%, against gates of <3%
and >=85%, across nine systematic idiom classes -- issue #184. Every one of those classes reached a user in exactly two
ways: as C2's own verdict, or as a C2 gaming marker that turned C1's verdict into GAMED_SUSPECT,
C1's only accusatory verdict. So C2 comes off the shipped path AND C1 is denied its markers: that
removes 100% of the measured false-accusation risk from the shipped product rather than deferring
it. What is left is PROVEN / UNPROVEN-with-a-named-reason / CRASHED / NOT_RUN. The canonical fake
green ("green-on-revert") is UNPROVEN, not an accusation; a skip is UNPROVEN reason "collection".
Where the check cannot PROVE, it withholds.

C2 IS FROZEN, NOT REMOVED: its code, its tests and its corpora stay, and it is reachable through
`corund replay`. It is simply not run here.

WHAT DID NOT LEAVE WITH C2: the contamination refusal. `contamination_from_diff` is a PURE function
over the PR's own diff that this adapter calls itself -- it is not C2's verdict -- and it is what
makes the design invariant true (a PR that adds or modifies a revert-protected file gets
every C1 witness REFUSED). It yields UNPROVEN-contaminated, a safety refusal, never an accusation.
If it cannot be computed, C1 is CRASHED: a disarmed guard never leaves the gate open.

Posture (owner's ruling): OBSERVE by default — the check run concludes NEUTRAL and the receipt is
posted as ONE PR comment (upserted); BLOCK is a per-rule opt-in via --block c1. `--block` naming a
check this release does not run is REFUSED (receipt.parse_block), because an un-runnable id quietly
ignored is a gate the user believes is armed and is not. One check run per check, created
`in_progress` first and completed with the verdict, so a check that could not run is posted as
NOT_RUN / CRASHED with its reason — never omitted, never green.

Honest about its own crash: main() wraps everything; any internal exception concludes every check
it owns that has no verdict yet as CRASHED with the exception's text, still writes the receipt
JSON, still tries to post, and exits 2. The token is read from CORUND_GITHUB_TOKEN (env, never
argv) and hash-and-suppressed out of every string that leaves this process. NEW module.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import sys
import traceback
from collections.abc import Mapping
from typing import Any, Callable

from corund_checks import Verdict, run_check
# `contamination_from_diff` lives in the c2 module but is C1's own guard: a pure function over the
# PR's diff, taking no C2 verdict and producing none. The gaming-marker function is deliberately NOT
# imported -- it is the accusation source this release removes.
from corund_checks.c2_skip_audit import contamination_from_diff

from . import __version__, c1_runner, gitio, receipt, runners
from .github_api import GitHubApi

# DERIVED from receipt.RUNNABLE_CHECKS, never a second hand-written list: the set of checks that get
# a check run, the set `--block` will accept, and the set the receipt renders are one set.
OWNED_CHECKS = receipt.RUNNABLE_CHECKS
# ISSUE #202: ONE default, not two. `c1_runner` has to know whether the caller chose their own globs
# before it may honour the repository's own `testpaths`, and a second copy of this string here would
# be a guard that holds only while two literals happen to agree.
DEFAULT_GLOBS = c1_runner.DEFAULT_TEST_GLOBS


class _Ctx:
    def __init__(self) -> None:
        self.verdicts: dict[str, Verdict] = {}
        self.check_run_ids: dict[str, int] = {}
        self.post_errors: list[str] = []
        self.runs: list[dict] = []
        self.internal_error: str | None = None
        self.api = None
        self.block: frozenset[str] = frozenset()
        self.details_url: str | None = None
        self.secrets: list[str] = []
        self.base: str | None = None
        self.head: str | None = None
        self.pr_number: int | None = None
        self.repo: str | None = None
        self.runner: dict = {}
        # The C1 inputs block (receipt.C1_INPUTS_MARKER_START) the hosted App reads from the C1 check
        # run's text: the probe and C1's input snapshot once the gather has run, and C2's gaming markers
        # when they were computed without error. None / {} until then, and the block says so.
        self.probe: dict | None = None
        self.c1_inputs: dict = {}
        self.c1_markers: dict | None = None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="corund", description="Corund — require the PR's new tests to fail on the old code")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run C1 red-on-revert on one pull request and post the receipt")
    r.add_argument("--repo-dir", default=".")
    r.add_argument("--base", help="base SHA (default: from GITHUB_EVENT_PATH pull_request.base.sha)")
    r.add_argument("--head", help="head SHA (default: pull_request.head.sha, else GITHUB_SHA)")
    r.add_argument("--pr-number", type=int, help="PR number for the comment (default: from the event)")
    r.add_argument("--test-command", default="python -m pytest", help="runner command; the report flag and files are appended")
    r.add_argument("--test-globs", default=DEFAULT_GLOBS)
    r.add_argument("--runner", default="auto", help="pytest | jest | vitest | auto")
    r.add_argument("--skip-allowlist", default="tests/loud_skips.txt",
                   help="ACCEPTED AND INERT in this release: the loud-skip allowlist is C2's input and C2 is not run by "
                        "the Action. Kept so existing workflows do not error on upgrade; every receipt says it does nothing")
    r.add_argument("--block", default="",
                   help=f"comma list of check ids that may conclude failure (BLOCK opt-in); default OBSERVE. "
                        f"This release runs {', '.join(receipt.RUNNABLE_CHECKS)}; naming any other id is REFUSED, never ignored")
    r.add_argument("--timeout-minutes", type=float, default=20.0)
    r.add_argument("--out", default="corund-receipt.json")
    r.add_argument("--details-url", default=None)
    r.add_argument("--no-post", action="store_true", help="compute and write the receipt; call no API")
    return ap


def _event(env: Mapping[str, str]) -> dict:
    p = env.get("GITHUB_EVENT_PATH")
    if p and os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}
    return {}


def _c1_inputs_payload(ctx: _Ctx) -> dict:
    """What the C1 check run publishes for the App: C1's own base (the merge-base its diff was reverted
    onto) when C1 got that far, else the PR base; the two results maps flattened to status words; the
    probe as the gather left it, or -- when C1 completed before the gather (a shallow checkout, the
    Action's own crash) -- `runner-error`, the core's word for a probe run that did not happen, with
    the reason in `detail`."""
    probe = ctx.probe or {
        "state": "runner-error",
        "detail": ("the Action did not reach the test runs: "
                   + (ctx.internal_error or "C1 completed before them (its evidence says why)"))[:300]}
    return receipt.c1_inputs_payload(
        base_sha=ctx.c1_inputs.get("base_sha") or ctx.base, head_sha=ctx.head, inputs=ctx.c1_inputs, probe=probe,
        markers_computed=ctx.c1_markers is not None)


def _summary_output(v: Verdict, ctx: _Ctx) -> dict:
    text = "\n".join(v.evidence)
    if v.error:
        text = f"error: {v.error}\n\n{text}"
    # The standing sentences ride on the CHECK RUN too, not only on the PR comment: a reader who opens
    # the check and never scrolls the comment must still meet the product sentence, the named residual,
    # the one line freezing C2, and the one line saying `skip-allowlist` is inert. They go BEFORE the
    # C1 inputs block, which is what gets protected when the text has to be trimmed.
    text = text + "\n\n" + "\n".join(receipt.STANDING_NOTES)
    text = receipt.redact(text, ctx.secrets)
    if v.check == "C1":
        # The App's C1 input source rides on THIS check run's text, whatever C1's verdict. If the block
        # cannot be built (a probe outside the core's vocabulary is the only way), the check run is still
        # completed -- with the plain text, NO block, and a posting error that names it, so the step
        # exits non-zero and the App reads "no marker" (NOT_RUN) rather than a block that was spelled past
        # the rule.
        try:
            text = receipt.check_run_text_with_c1_inputs(text, _c1_inputs_payload(ctx),
                                                         redact=lambda s: receipt.redact(s, ctx.secrets))
        except Exception as exc:  # noqa: BLE001 — recorded, never hidden, never a fake block
            ctx.post_errors.append(receipt.redact(f"c1 inputs block not published: {type(exc).__name__}: {exc}", ctx.secrets))
            text = text[:receipt.CHECK_RUN_TEXT_LIMIT]
    else:
        text = text[:receipt.CHECK_RUN_TEXT_LIMIT]
    return {"title": receipt.check_run_title(v)[:255],
            "summary": receipt.redact(f"{v.display_state} — posture {'BLOCK' if v.check in ctx.block else 'OBSERVE'}; "
                                      f"base {(v.base_sha or ctx.base or '?')[:12]} head {(v.head_sha or ctx.head or '?')[:12]}",
                                      ctx.secrets)[:receipt.CHECK_RUN_TEXT_LIMIT],
            "text": text}


def _open_check_runs(ctx: _Ctx) -> None:
    if ctx.api is None or not ctx.head:
        return
    for check in OWNED_CHECKS:
        try:
            ctx.check_run_ids[check] = ctx.api.create_check_run(
                name=receipt.check_run_name(check), head_sha=ctx.head, status="in_progress",
                output={"title": f"{receipt.check_run_name(check)}: running", "summary": "Corund is running this check."},
                details_url=ctx.details_url)
        except Exception as exc:  # noqa: BLE001 — recorded, never hidden
            ctx.post_errors.append(receipt.redact(f"create_check_run {check}: {type(exc).__name__}: {exc}", ctx.secrets))


def _finalize(ctx: _Ctx, v: Verdict) -> None:
    ctx.verdicts[v.check] = v
    if ctx.api is None:
        return
    cid = ctx.check_run_ids.get(v.check)
    conclusion = receipt.conclusion_for(v, block=v.check in ctx.block)
    try:
        if cid is None:
            if not ctx.head:
                raise RuntimeError("no head SHA to attach the check run to")
            ctx.check_run_ids[v.check] = ctx.api.create_check_run(
                name=receipt.check_run_name(v.check), head_sha=ctx.head, status="completed",
                conclusion=conclusion, output=_summary_output(v, ctx), details_url=ctx.details_url)
        else:
            ctx.api.update_check_run(check_run_id=cid, status="completed", conclusion=conclusion,
                                     output=_summary_output(v, ctx), details_url=ctx.details_url)
    except Exception as exc:  # noqa: BLE001
        ctx.post_errors.append(receipt.redact(f"complete check run {v.check}: {type(exc).__name__}: {exc}", ctx.secrets))


def _not_run(check: str, ctx: _Ctx, why: str) -> Verdict:
    return Verdict(check, "NOT_RUN", ctx.base, ctx.head, None, (f"NOT_RUN: {why}",))


def _crashed(check: str, ctx: _Ctx, text: str) -> Verdict:
    return Verdict(check, "CRASHED", ctx.base, ctx.head, None, (f"CRASHED inside the Action: {text}",), error=text)


def _do_run(args: argparse.Namespace, env: Mapping[str, str], ctx: _Ctx) -> None:
    repo_dir = os.path.abspath(args.repo_dir)
    ev = _event(env)
    pr = ev.get("pull_request") if isinstance(ev, dict) else None
    ctx.base = args.base or ((pr or {}).get("base") or {}).get("sha")
    ctx.head = args.head or ((pr or {}).get("head") or {}).get("sha") or env.get("GITHUB_SHA")
    ctx.pr_number = args.pr_number if args.pr_number is not None else (pr or {}).get("number")
    ctx.block = receipt.parse_block(args.block)
    ctx.details_url = args.details_url or (
        f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/{ctx.repo}/actions/runs/{env['GITHUB_RUN_ID']}"
        if ctx.repo and env.get("GITHUB_RUN_ID") else None)
    if not ctx.base or not ctx.head:
        raise RuntimeError("base and head SHAs are required (--base/--head, or a pull_request event payload)")

    _open_check_runs(ctx)

    if not gitio.has_commit(repo_dir, ctx.base) or not gitio.has_commit(repo_dir, ctx.head):
        why = (f"base {ctx.base[:12]} or head {ctx.head[:12]} is not in the checkout — a shallow clone; "
               f"set `fetch-depth: 0` on actions/checkout so the merge-base exists")
        for check in OWNED_CHECKS:
            _finalize(ctx, _not_run(check, ctx, why))
        return

    mb = gitio.merge_base(repo_dir, ctx.base, ctx.head)
    family = args.runner if args.runner != "auto" else (runners.detect_family(repo_dir) or "unknown")
    test_command = shlex.split(args.test_command)
    ctx.runner = {"family": family, "command": test_command, "globs": args.test_globs, "merge_base": mb}

    # ---- the PR's own diff, for C1's contamination guard -----------------------------------------
    # renames OFF: a test file renamed away must show as a deletion; paths unquoted (core.quotePath=false)
    # `args.skip_allowlist` is NOT read: it is C2's input, C2 is not run here, and the receipt says so
    # rather than the value quietly selecting nothing.
    diff_text = gitio.diff(repo_dir, mb, ctx.head, binary=False, renames=False)
    files_after, files_before = _changed_texts(repo_dir, mb, ctx.head)

    # ---- GATHER -----------------------------------------------------------------------------------
    # `run_c1` takes no C2 input of any kind, and C1's own guard comes from `contamination_from_diff`
    # -- a PURE function over the diff that this adapter calls itself -- never from C2's verdict.
    # Every path below sets `probe`, and its `state` always speaks the core's closed vocabulary; the
    # probe still travels on the C1 inputs block because the hosted App reads it.
    c1_verdict: Verdict | None = None
    out: c1_runner.C1Outcome | None = None
    if family not in runners.SUPPORTED_FAMILIES:
        probe = {"state": "runner-unsupported", "detail": f"runner family {family!r}"}
        c1_verdict = _not_run("C1", ctx, f"runner family {family!r} is NOT SUPPORTED (launch support: "
                                         f"{', '.join(runners.SUPPORTED_FAMILIES)}); pass --runner or add a supported runner")
    else:
        try:
            out = c1_runner.run_c1(repo_dir=repo_dir, base_sha=ctx.base, head_sha=ctx.head, test_globs=args.test_globs,
                                   family=family, test_command=test_command, timeout_s=int(args.timeout_minutes * 60))
        except c1_runner.RunnerError as exc:
            probe = {"state": "runner-error", "detail": f"C1 CRASHED: {exc}"[:300]}
            c1_verdict = _crashed("C1", ctx, f"RunnerError: {exc}")
        except SystemExit:
            raise
        except BaseException as exc:  # noqa: BLE001 — recorded on the receipt, never swallowed
            # The Action's own crash, now that the gather runs BEFORE C2. C2's audit does not depend
            # on this run, so it still gets its own verdict — exactly the two states this produced
            # before the hoist, when C2 had already posted — and the receipt still says the Action
            # crashed. C1 is CRASHED with the text; the probe says the comparison did not run.
            text = f"{type(exc).__name__}: {exc}"
            ctx.internal_error = receipt.redact(text + " | " + traceback.format_exc().strip().splitlines()[-1],
                                                ctx.secrets)
            probe = {"state": "runner-error", "detail": receipt.redact(text, ctx.secrets)[:300]}
            c1_verdict = _crashed("C1", ctx, receipt.redact(text, ctx.secrets))
        else:
            probe = out.probe
            ctx.runs = out.runs
            if out.kind == "not_run":
                c1_verdict = _not_run("C1", ctx, out.reason)
    ctx.probe = probe
    ctx.c1_inputs = dict(out.inputs) if out is not None else {}

    # ---- C1's contamination guard ----------------------------------------------------------------
    # The design invariant (owner's ruling 2026-09-05): a C1 witness is trusted only when
    # its assertion failure originates in the test's own call on a tree whose non-test,
    # non-infrastructure code is the only thing reverted. A PR that adds or modifies a
    # revert-protected file (conftest.py, a plugin module, anything defining a pytest hook or an
    # autouse fixture) gets every witness REFUSED -- UNPROVEN-contaminated, worded as a safety
    # refusal, never as an accusation. That is C1's guard, not C2's verdict, and it stays.
    #
    # If it cannot be computed, C1 is CRASHED and never PROVEN: a scanner that can skip is not a
    # gate, and a disarmed guard must not leave the gate open. The message names the guard --
    # it is NOT spelled as `gaming_markers_error`, which would print "C2's gaming markers are
    # unavailable" on a receipt where C2 was never asked for anything.
    contaminating: dict = {}
    try:
        if diff_text.strip():
            contaminating = contamination_from_diff(diff_text, files_after, files_before)
    except BaseException as exc:  # noqa: BLE001 — never swallowed into an unguarded run
        if c1_verdict is None:
            c1_verdict = _crashed("C1", ctx, f"C1's contamination guard could not be computed "
                                             f"({type(exc).__name__}: {exc}); a witness cannot be trusted without it")
    # C2's gaming markers are NOT computed here: C2 is not run by the Action in this release, so there
    # is no marker source and C1 is told so by being handed an EMPTY marker set. `ctx.c1_markers` stays
    # None, which keeps `skips_added_for` off the published block -- the Action must not publish an
    # empty skip set as if it had computed one.
    ctx.c1_markers = None

    # ---- C1 -----------------------------------------------------------------------------------
    # Decided during the gather above; posted here. The input snapshot is assembled on every path --
    # it is what the C1 check run publishes for the App, guards included, whether or not the core is
    # asked for a verdict here.
    inputs = dict(out.inputs) if out is not None else {}
    # EXPLICIT, not omitted. `gaming_markers={}` states that no marker is in effect; an omitted key
    # would mean the same thing to the core but would leave the published block silent, and the App
    # forwards whatever the block carries into its own C1. An empty mapping denies the hosted C1 its
    # markers too, from this Action's own words.
    inputs["gaming_markers"] = {}
    # `gaming_markers_error` is NEVER set. Its PRESENCE is the signal that C2's markers could not be
    # COMPUTED -- which, with a witness present, is C1 CRASHED (guard disarmed). C2 not being run is
    # not a failure to compute, and spelling it as one would turn every proven PR in this release into
    # a crash. This is the line the whole pivot turns on.
    if contaminating:
        inputs["contaminating_files"] = contaminating
    ctx.c1_inputs = inputs
    if c1_verdict is not None or out is None:
        _finalize(ctx, c1_verdict or _crashed("C1", ctx, "no C1 outcome and no reason — the gather fell through"))
        return
    _finalize(ctx, run_check("C1", inputs))


_TEXT_EXT = (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".ini", ".toml", ".cfg", ".json",
             ".yml", ".yaml", ".txt")


def _changed_texts(repo_dir: str, base: str, head: str) -> tuple[dict[str, str], dict[str, str]]:
    """Full NEW-side and OLD-side text of every changed text file, for C2's AST/token tiers and its
    before/after collectability comparison. Read from git at exact SHAs; never from the working tree."""
    after: dict[str, str] = {}
    before: dict[str, str] = {}
    for st, path in gitio.changed_files(repo_dir, base, head):
        if not path.lower().endswith(_TEXT_EXT):
            continue
        if st != "D":
            t = gitio.show_file(repo_dir, head, path)
            if t is not None:
                after[path] = t
        if st != "A":
            t = gitio.show_file(repo_dir, base, path)
            if t is not None:
                before[path] = t
    return after, before


def _write_receipt(args: argparse.Namespace | None, ctx: _Ctx) -> dict:
    rec = receipt.build_receipt(
        repo=ctx.repo, pr_number=ctx.pr_number, base_sha=ctx.base, head_sha=ctx.head, block=ctx.block,
        runner=ctx.runner, verdicts=[ctx.verdicts[c] for c in OWNED_CHECKS if c in ctx.verdicts],
        runs=ctx.runs, internal_error=ctx.internal_error, post_errors=ctx.post_errors, version=__version__,
        check_run_ids=ctx.check_run_ids, measured_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    text = receipt.redact(json.dumps(rec, indent=2, sort_keys=True), ctx.secrets)
    out = getattr(args, "out", None) or "corund-receipt.json"
    try:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except OSError as exc:
        sys.stderr.write(f"corund: could not write the receipt to {out}: {exc}\n{text}\n")
    return json.loads(text)


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None,
         api_factory: Callable[[str, str], Any] | None = None) -> int:
    env = os.environ if env is None else env
    ctx = _Ctx()
    ctx.repo = env.get("GITHUB_REPOSITORY")
    token = env.get("CORUND_GITHUB_TOKEN") or ""
    ctx.secrets = [token] if token else []
    args = None
    try:
        args = build_parser().parse_args(argv)
        if args.cmd != "run":
            raise RuntimeError(f"unknown command {args.cmd!r}")
        if not args.no_post:
            if not token:
                raise RuntimeError("no token: set CORUND_GITHUB_TOKEN (the Action passes inputs.token through the environment) "
                                   "or pass --no-post")
            if not ctx.repo:
                raise RuntimeError("GITHUB_REPOSITORY is not set")
            factory = api_factory or (lambda t, r: GitHubApi(token=t, repo=r))
            ctx.api = factory(token, ctx.repo)
        _do_run(args, env, ctx)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — the Action's own crash, made visible on every owned check
        text = f"{type(exc).__name__}: {exc}"
        ctx.internal_error = receipt.redact(text + " | " + traceback.format_exc().strip().splitlines()[-1], ctx.secrets)
        for check in OWNED_CHECKS:
            if check not in ctx.verdicts:
                _finalize(ctx, _crashed(check, ctx, receipt.redact(text, ctx.secrets)))

    rec = _write_receipt(args, ctx)
    md = receipt.render_markdown(rec)
    summary_path = env.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(md)
        except OSError:
            pass
    if ctx.api is not None and ctx.pr_number is not None:
        try:
            ctx.api.upsert_pr_comment(pr_number=int(ctx.pr_number), marker=receipt.COMMENT_MARKER, body=md)
        except Exception as exc:  # noqa: BLE001
            ctx.post_errors.append(receipt.redact(f"pr comment: {type(exc).__name__}: {exc}", ctx.secrets))
            rec = _write_receipt(args, ctx)
    sys.stdout.write(md)
    for line in ctx.post_errors:
        sys.stderr.write(f"corund: posting error: {line}\n")

    if ctx.internal_error:
        return 2
    if ctx.post_errors:
        return 1
    blocked = [v for v in ctx.verdicts.values()
               if v.check in ctx.block and receipt.conclusion_for(v, block=True) in ("failure", "action_required")]
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
