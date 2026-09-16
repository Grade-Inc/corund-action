"""`corund replay` — the onboarding centrepiece: run ALL checks over the repo's last N
merged PRs, read-only, post nothing, and report "would have flagged" per rule with true/false-
positive accounting BEFORE any enforcement is switched on.

Per merged PR (derived from first-parent history: `Merge pull request #N` merge commits or squash
commits whose subject ends in `(#N)`):
  C2  git diff base..head + the base tree's allowlist                     always
  C1  the revert-run over history                                          only with --run-tests
  C3  the CURRENT branch protection (read-only GET; stated as such) for both before and after,
      workflow files at base and head from git                             only with a token
  C4  the PR's reviews (read-only GET)                                     only with a token + PR number
A check that could not be built is NOT_RUN with the reason. `--mark '#12:C2:fp'` records the
operator's judgement in the marks file; the report shows per-rule counts and the false-positive
rate. `--redact` applies hash-and-suppress to every evidence line (results files that may carry
commit SHAs only). NEW module; the engine is corund_checks.replay.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Mapping
from typing import Any, Callable

from corund_checks import replay as engine

from . import c1_runner, gitio, runners
from .github_api import GitHubApi


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="corund replay", description=__doc__.splitlines()[0])
    ap.add_argument("--repo-dir", default=".")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--last", type=int, default=50)
    ap.add_argument("--out", default="corund-replay.jsonl")
    ap.add_argument("--report", default="corund-replay.txt")
    ap.add_argument("--marks", default=None, help="JSON file of operator marks {pr_id: {check: tp|fp}}")
    ap.add_argument("--mark", action="append", default=[], help="record a mark: '#12:C2:fp' (repeatable)")
    ap.add_argument("--repo", default=None, help="owner/name for read-only GETs (default GITHUB_REPOSITORY)")
    ap.add_argument("--skip-allowlist", default="tests/loud_skips.txt")
    ap.add_argument("--run-tests", action="store_true", help="also run C1's revert-run per PR (slow; needs the runner)")
    ap.add_argument("--test-command", default="python -m pytest")
    ap.add_argument("--test-globs", default="tests/**/test_*.py,**/test_*.py,**/*_test.py,**/*.test.js,**/*.test.ts,**/__tests__/**")
    ap.add_argument("--runner", default="auto")
    ap.add_argument("--timeout-minutes", type=float, default=20.0)
    ap.add_argument("--redact", action="store_true", help="hash-and-suppress evidence lines in the JSONL")
    ap.add_argument("--header-comment", default=None, help="a second `# ...` header line for the JSONL (e.g. a grep-proof allow)")
    ap.add_argument("--label", default=None, help="how the report names the repo (default: --repo, else the directory name)")
    return ap


def _load_marks(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _apply_marks(marks: dict, specs: list[str]) -> dict:
    for spec in specs:
        try:
            pr, check, mark = spec.rsplit(":", 2)
        except ValueError:
            raise ValueError(f"--mark must look like '#12:C2:fp', got {spec!r}") from None
        if mark not in engine.MARKS:
            raise ValueError(f"--mark {spec!r}: mark must be one of {engine.MARKS}")
        marks.setdefault(pr, {})[check.upper()] = mark
    return marks


def _workflow_files(repo_dir: str, ref: str) -> dict[str, str]:
    out = {}
    for p in gitio.ls_tree_paths(repo_dir, ref, ".github/workflows"):
        if p.endswith((".yml", ".yaml")):
            txt = gitio.show_file(repo_dir, ref, p)
            if txt is not None:
                out[p] = txt
    return out


def build_items(args: argparse.Namespace, repo_dir: str, prs: list[dict], api: Any, protection: dict | None,
                protection_note: str | None) -> list[engine.ReplayItem]:
    family = args.runner if args.runner != "auto" else (runners.detect_family(repo_dir) or "unknown")
    test_command = shlex.split(args.test_command)
    items = []
    for pr in prs:
        pr_id = f"#{pr['pr']}" if pr.get("pr") else pr["sha"][:12]
        base, head = pr["base_sha"], pr["head_sha"]
        inputs: dict[str, Any] = {}
        reasons: dict[str, str] = {}
        crashed: dict[str, str] = {}
        try:
            mb = gitio.merge_base(repo_dir, base, head)
        except gitio.GitError as exc:
            mb = base
            reasons["C2"] = f"merge-base failed: {exc}"
        if "C2" not in reasons:
            allow = gitio.show_file(repo_dir, mb, args.skip_allowlist)
            inputs["C2"] = {"base_sha": mb, "head_sha": head, "diff": gitio.diff(repo_dir, mb, head, binary=False),
                            "skip_allowlist": allow.splitlines() if allow is not None else [],
                            "skip_allowlist_path": args.skip_allowlist,
                            # replay re-runs the CHECKS over history, never the test suite, so the
                            # runtime silencing comparison is not available here and says so.
                            "silencing_probe": {"state": "not-supplied"}}
        # C1
        if not args.run_tests:
            reasons["C1"] = "replay ran without --run-tests (the revert-run over history is opt-in; slow)"
        elif family not in runners.SUPPORTED_FAMILIES:
            reasons["C1"] = f"runner family {family!r} is NOT SUPPORTED"
        else:
            try:
                out = c1_runner.run_c1(repo_dir=repo_dir, base_sha=base, head_sha=head, test_globs=args.test_globs,
                                       family=family, test_command=test_command, timeout_s=int(args.timeout_minutes * 60))
                if out.kind == "inputs":
                    inputs["C1"] = out.inputs
                else:
                    reasons["C1"] = out.reason
            except c1_runner.RunnerError as exc:
                crashed["C1"] = f"RunnerError: {exc}"
        # C3
        if api is None:
            reasons["C3"] = "no token: branch protection is a read-only GET that needs one (CORUND_GITHUB_TOKEN)"
        elif protection is None and protection_note:
            reasons["C3"] = protection_note
        else:
            inputs["C3"] = {"protection_before": protection, "protection_after": protection,
                            "workflow_files_before": _workflow_files(repo_dir, mb),
                            "workflow_files_after": _workflow_files(repo_dir, head)}
        # C4
        if api is None:
            reasons["C4"] = "no token: PR reviews are a read-only GET that needs one"
        elif not pr.get("pr"):
            reasons["C4"] = "no PR number derivable from the merge commit subject"
        else:
            try:
                inputs["C4"] = {"approvals": api.get_pr_reviews(int(pr["pr"])), "head_sha": head}
            except Exception as exc:  # noqa: BLE001
                reasons["C4"] = f"reviews GET failed: {type(exc).__name__}: {exc}"
        items.append(engine.ReplayItem(pr_id=pr_id, base_sha=base, head_sha=head, inputs=inputs, not_run_reasons=reasons,
                                       crashed_reasons=crashed, meta={"kind": pr["kind"], "merge_sha": pr["sha"]}))
    return items


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None,
         api_factory: Callable[[str, str], Any] | None = None) -> int:
    env = os.environ if env is None else env
    args = build_parser().parse_args(argv)
    repo_dir = os.path.abspath(args.repo_dir)
    tree = gitio.rev_parse(repo_dir, args.branch)
    token = env.get("CORUND_GITHUB_TOKEN") or ""
    repo = args.repo or env.get("GITHUB_REPOSITORY")
    api = None
    if token and repo:
        api = (api_factory or (lambda t, r: GitHubApi(token=t, repo=r)))(token, repo)
    protection, protection_note = None, None
    if api is not None:
        try:
            protection = api.get_branch_protection(args.branch)
            if protection is None:
                protection_note = f"branch {args.branch} has no protection (404) — C3 cannot compare gates that do not exist"
        except Exception as exc:  # noqa: BLE001
            protection_note = f"branch protection GET failed: {type(exc).__name__}: {exc}"

    prs = gitio.merged_prs(repo_dir, args.branch, args.last)
    marks = _apply_marks(_load_marks(args.marks), args.mark)
    if args.marks:
        with open(args.marks, "w", encoding="utf-8") as fh:
            json.dump(marks, fh, indent=2, sort_keys=True)

    label = args.label or repo or os.path.basename(repo_dir)
    if not prs:
        text = (f"tree {tree} — corund replay over 0 merged PR(s) of {label}: no merge commits and no `(#N)` subjects "
                f"in the first-parent history of {args.branch}; nothing was replayed (this is not a pass).\n")
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(f"# tree {tree} corund replay: 0 merged PRs derived\n")
        sys.stdout.write(text)
        return 2

    items = build_items(args, repo_dir, prs, api, protection, protection_note)
    rows = engine.replay(items)
    report = engine.summarize(rows, marks)
    text = engine.render_report(report, tree_sha=tree, repo_label=label)
    if protection is not None:
        text += ("note: C3 compared the CURRENT branch protection against itself for every PR (historical protection is "
                 "not in git); only workflow-file folds between base and head can differ per PR.\n")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(f"# tree {tree} corund replay over {len(rows)} merged PR(s); branch {args.branch}; "
                 f"redacted={'yes' if args.redact else 'no'}; C1 {'ran' if args.run_tests else 'not run (no --run-tests)'}; "
                 f"C3/C4 {'ran read-only' if api is not None else 'not run (no token)'}\n")
        if args.header_comment:
            fh.write(f"# {args.header_comment}\n")
        for row in rows:
            if args.redact:
                row = engine.ReplayRow(row.pr_id, row.base_sha, row.head_sha,
                                       tuple(engine.redact_verdict(v) for v in row.verdicts), row.meta)
            fh.write(json.dumps(next(iter(engine.rows_to_dicts([row]))), sort_keys=True) + "\n")
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
