# Corund Action

**Require the PR's new tests to fail on the old code.**

Corund reverts your pull request's non-test diff onto the base, runs the tests the pull request
added or changed, and requires at least one of them to EXECUTE and FAIL BY ASSERTION on that
reverted tree — then to pass again with the change restored. Both runs go on the receipt.

No model decides anything. Every verdict is a deterministic comparison you can re-run.

## Quickstart

One file, `.github/workflows/corund.yml`, in a repository whose tests run under pytest:

```yaml
name: corund
on:
  pull_request:
permissions:
  contents: read          # actions/checkout reads the repository
  checks: write           # Corund posts its own check run
  pull-requests: write    # Corund upserts one receipt comment on the PR
jobs:
  corund:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0             # load-bearing: red-on-revert needs the merge-base
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pytest          # test dependencies ONLY -- not the package itself
      - uses: Grade-Inc/corund-action@v0.1.3
        with:
          runner: pytest
          test-command: python -m pytest
```

That is the whole install. Nothing blocks: every check-run conclusion is `neutral` until you opt in.

**Why `runner:` is named.** This release detects the runner from a config file (`pytest.ini`, `pyproject.toml`, `package.json`). A repository with tests and none of those has nothing to detect, so naming it is what keeps the paste block working on a bare repository; the next release reads the family from `test-command` and the line becomes optional.

**Why each line is there.** `fetch-depth: 0` is not optional — with a shallow clone the base commit
is not in the checkout, and Corund reports `NOT_RUN` saying so rather than guessing. The three
permissions are the three things the Action does: read the repository, post one check run
(`POST`/`PATCH /repos/{owner}/{repo}/check-runs`), and upsert one comment
(`GET`/`POST`/`PATCH /repos/{owner}/{repo}/issues/{number}/comments`). It asks for nothing else.
`test-command` defaults to `python -m pytest`; the jest and vitest families are supported too, and
every other runner reports `NOT_RUN` naming the runner rather than passing.

**Importing the code under test.** Corund takes your non-test diff back out in a *fresh worktree*,
so the tests have to import your code **from that tree**. An editable install (`pip install -e .`)
points at the original checkout, which is never reverted -- so every test that reads through one
passes on the reverted tree and the verdict is `UNPROVEN`, however good the test is. A non-editable
`pip install .` is the same trap: it copies your code into `site-packages` at head. Install your test
*dependencies*, not the package under test. A flat layout is imported from the rootdir automatically;
a `src/` layout needs `pythonpath = ["src"]` under `[tool.pytest.ini_options]`.

**The pin.** `@v0.1.3` is a released tag, not a moving one. `@v0` is a floating major that is
re-pointed on each release and `@main` is not a release at all; pin the exact version so the
Action that runs tomorrow is the one you read today.

## What the check does, and what it will not claim

| verdict | what it means |
|---|---|
| `PROVEN` | a new/changed test executed on the reverted tree and failed **in its own call, by assertion**, and passes with the change restored. The witness is named on the receipt. |
| `UNPROVEN-green-on-revert` | the tests still pass with the change reverted — the canonical fake green. Not an accusation: a stated, named absence of proof. |
| `UNPROVEN-compile` / `UNPROVEN-collection` | the red was an ImportError, a collection failure, a fixture error or a signature TypeError — a red for the wrong reason, so it is not a witness. |
| `UNPROVEN-flaky` | the red did not reproduce on a third, freshly reverted tree. |
| `UNPROVEN-contaminated` | the pull request itself adds or modifies test infrastructure that runs during the reverted phase (a conftest hook, an autouse fixture, `pytest_plugins`). That code is never reverted, so every witness is refused, naming the file. A safety refusal, not a finding of intent. |
| `CRASHED` | the Action could not complete the comparison — a missing runner, a timeout, an unhandled exception. The exception's own text is on the receipt. |
| `NOT_RUN` | there was nothing to compare: no changed test files, only test and test-infrastructure files, a reverse patch with no executable line, an unsupported runner, or a shallow clone. |

**Corund never reports a check it did not run.** `CRASHED` and `NOT_RUN` are their own verdicts and
are never folded into a pass. If the Action itself crashes, the check it owns is concluded
`CRASHED` with the exception's text. There is no path to green by omission.

**The limit we publish rather than hide.** Corund proves a test ran and fails on revert; it does not
verify the test asserts the CORRECT value, so a pull request whose code and test are wrong in
agreement is NOT DECIDED. Nor does a `PROVEN` verdict mean the test targets the bug you had in mind:
a test whose assertion genuinely fails once the diff is reverted is `PROVEN` even if what it asserts
on is an unrelated constant the same diff touched. The receipt names the witness so you can read
what it asserted.

**What this release does not check.** This release checks that new/changed tests fail on the old
code. It does not flag a PR that only turns off an existing test.

## Posture: observe first

Every check-run conclusion is `neutral` by default and the receipt is posted as one pull-request
comment, updated in place on each push. Nothing blocks. To let the check block a merge, opt it in:

```yaml
with:
  block: c1
```

In BLOCK mode: `PROVEN` → success; `CRASHED` / `NOT_RUN` → `action_required` (it blocks, and it is
distinct from a failure — a crash is never green); every `UNPROVEN` → `neutral`, never a block by
itself. Read the replay report below before opting in.

## `corund replay` — read-only, before any enforcement

Replay runs the checks over the repository's own last N merged pull requests, derived from
first-parent history (`Merge pull request #N` merges, or squash commits whose subject ends in
`(#N)`). It posts nothing to GitHub.

```
python3 corund_cli.py replay --repo-dir . --branch main --last 50 --run-tests \
    --test-command "python -m pytest" \
    --out corund-replay.jsonl --report corund-replay.txt
```

`--run-tests` is what makes the check run: without it the revert-run is skipped and it reports `NOT_RUN`.
Expect it to be slow — it runs your changed tests two or three times per pull request.

What it prints, from a real run over a one-pull-request repository whose new test does fail on the
old code. The report writes one row per check in the package; the red-on-revert row is the one this
release runs, and the rest of the report is elided here rather than retyped:

```
tree da4c69aaca091116351f4ed6afc36aeb0cef1903 — corund replay over 1 merged PR(s) of demo-repo
[...]
red-on-revert: ran 1/1; would have flagged 0; PROVEN 1; UNPROVEN 0; NOT_RUN 0; CRASHED 0; marked true positive 0, false positive 0, unmarked 0; false positive rate n/a (nothing marked)
[...]
```

`--out` writes one JSON line per pull request with every verdict and its evidence lines; `--report`
writes the summary above. Read the `UNPROVEN` and `CRASHED` counts on your own history before you
set `block:` — that is the point of running it first.

## Inputs

| input | default | meaning |
|---|---|---|
| `test-command` | `python -m pytest` | your runner; Corund appends the report flag and the changed test files |
| `test-globs` | pytest + jest/vitest patterns | which changed files are tests; everything else is the non-test diff the check reverts |
| `runner` | `auto` | `pytest`, `jest`, `vitest`, or auto-detect; any other runner reports `NOT_RUN` |
| `block` | (empty) | comma list of check ids allowed to conclude failure |
| `timeout-minutes` | `20` | per test run (at most three: with, without, rerun) |
| `token` | `github.token` | needs `checks: write` and `pull-requests: write`; read from the environment, never passed as an argument |

### Outputs

`receipt-path` — the receipt JSON: the verdict with its evidence lines, the runner, each test run's
command and exit code, and `internal_error` if the Action crashed. It is also uploaded as the
`corund-receipt` artifact and rendered into the job summary and the pull-request comment.

## How red-on-revert actually works

1. `git diff --name-status --no-renames -z <merge-base> <head>`, split three ways by `test-globs`:
   test files, **test-infrastructure files** (conftest.py, pytest.ini, pyproject.toml, setup.cfg,
   tox.ini, jest/vitest/vite config, package.json, `.github/workflows/**`, `__init__.py` under test
   dirs — never reverted), and the non-test diff.
2. The reverse patch of the non-test files is read for what it contains (text lines, mode-only,
   binary, symlink). A patch that changes no text line is `NOT_RUN`, "nothing executable to revert".
3. Phase **with-change**: a fresh `git worktree` at the head with its `.git` link removed, a fresh
   `TMPDIR`, a `--collect-only` pre-pass, then your test command on the changed test files with a
   machine-readable report (pytest `--junitxml`; jest `--json`; vitest `--reporter=json`) written
   outside the tree. The tree is deleted.
4. Phase **without-change**: a second fresh worktree, the reverse patch applied, `.git` removed,
   the same pre-pass and run.
5. If anything is red without the change, phase **rerun**: a third fresh reverted tree. A red that
   does not reproduce is flaky.
6. Every phase is reconciled: the runner's exit code, the report's counts and the ids the same tree
   collected must agree, and a witness must be collected in both trees. Any disagreement is
   `CRASHED` naming it; a forged report never yields `PROVEN`.
7. Every red is attributed: the raising frame is read from the report (`--tb=auto` is appended so it
   is there); an assertion raised from a `conftest.py` or a plugin is not the test's own and is never
   a witness.
8. Compare. No `git status`, marker file or temp-dir oracle can tell one phase from another: no phase
   runs in a tree another phase has touched, and no tree carries `.git`. State a test persists outside
   the tree and its TMPDIR (HOME, the original checkout, the network) is NOT isolated, and that is
   named on every receipt.

Your test command must import the code under test from the execution tree (rootdir, `pythonpath`, a
src layout). An editable install of the original checkout is not reverted, and every test that reads
through one is green on the reverted tree.

### Runner support

| runner family | status | how results are read |
|---|---|---|
| pytest family (pytest, `python -m pytest`, tox/nox wrappers that pass args through) | supported | `--junitxml`; `fail` = a `<failure>` in the call phase whose message head is `assert` / `AssertionError` / `Failed:`; `<error>` = any other phase; a `--collect-only -q` pre-pass in the same tree gives the collected ids, and exit code / report counts / collected ids / witness ids are reconciled per run |
| jest / vitest family (`jest --json`, `vitest --reporter=json`) | supported | JSON reporter; `assertionResults` status and `failureMessages`; a suite-level `testExecError` is a collection error; collected-id reconciliation NOT AVAILABLE (no test-level listing without running) |
| go test, cargo test, rspec, minitest, phpunit, dotnet test, gradle/maven, mocha (without a junit reporter) | not supported | `NOT_RUN`, naming the runner |

### Execution isolation

| layer | what the Action does |
|---|---|
| execution tree | a FRESH `git worktree` per phase (with-change, without-change, rerun) with its `.git` link REMOVED before anything runs — no `git status` / `git log` / `rev-parse` oracle exists in any phase; the tree is deleted after |
| the reverted tree | head checkout + `git apply` of the reverse patch of the NON-TEST, NON-INFRASTRUCTURE files, then `.git` removed; the rerun is a third fresh tree built the same way |
| test infrastructure | conftest.py, pytest.ini, pyproject.toml, setup.cfg, tox.ini, jest/vitest/vite config, package.json, `.github/workflows/**`, `__init__.py` under test dirs are NEVER reverted (revert-protected surface) |
| temp dir | TMPDIR/TEMP/TMP point at a fresh directory per run, deleted after; the report is written OUTSIDE the tree and outside that TMPDIR at an unpredictable path; `PYTEST_ADDOPTS` is dropped from the environment |
| reconciliation | per run: the runner's exit code must agree with the report (pytest 0 = no failures and ≥ 1 test; 1 = ≥ 1 failure/error; 5 = nothing collected; 2/3/4 = did not complete), the `<testsuite>` counts must equal the counted `<testcase>`s, and the set of reported ids must equal the set collected by the pre-pass; every witness must be collected in BOTH trees. Any disagreement → `CRASHED` naming it; never `PROVEN` |
| reverted diff composition | named on the receipt (files, +/- lines, mode-only / binary / symlink) |
| witness attribution | the raising frame of every red is read from the junit body; an AssertionError raised from a conftest.py, a site-packages plugin or pytest's own internals is NOT the test's own assertion and never a witness |
| contamination | if the pull request adds or modifies test infrastructure that can affect outcomes, that code is never reverted and RUNS during the reverted phase: every witness is REFUSED — `UNPROVEN-contaminated` naming the file. A safety refusal, not a finding of intent |
| not closed | state persisted outside the tree and TMPDIR (HOME, XDG dirs, the original checkout path, the network) can still carry between phases — NAMED on every receipt's isolation line; a token-free self-consistent forgery; an outcome-rewriting hook in a conftest that predates the pull request. A sandbox is the next layer |

Corund speaks when your PR changes tests; on PRs that don't, it stays quiet.

## Licence

**This Action is MIT-licensed and free.** Public repositories and private ones, observe mode or
blocking -- `block:` is an input in your own workflow file and nothing here checks a licence. It
runs on your runners, in your CI, and reports to your repository. There is no account, no key and
no paid tier: this repository is the product.

A managed, organisation-wide service is being built. It is not part of this release and is not for
sale today; if you want it for your team, write to support@corund.dev.

## Layout

```
action.yml          the composite action
entrypoint.py       python3 entrypoint.py run ...
corund_cli.py       corund run | corund replay
corund_action/      gathering: git, the revert-run, runner adapters, the GitHub API client, the receipt
corund_checks/      the pure checks and the replay engine; stdlib only; run_check never raises
```

MIT License, Grade-Inc. This repository is a generated mirror; see NOTICE.
