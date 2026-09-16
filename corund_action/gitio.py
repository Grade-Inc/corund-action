"""git helpers for the Action and the replay CLI — every call is a local `git` subprocess against
the caller's checkout; nothing here writes to a remote. Errors carry git's own stderr.

Paths: `changed_files` reads `--name-status -z` (NUL-separated, never quoted) and every diff is
taken with `core.quotePath=false`, so a non-ASCII or space-bearing path is the same string
everywhere. Renames are OFF by default (`--no-renames`): a test file renamed away must appear as a
deletion plus an addition, not as a hunk-less rename header. NEW module."""
from __future__ import annotations

import os
import re
import shutil
import subprocess  # nosec B404 — the Action shells out to git by design; every call passes an argv list, never a shell string

_PR_SUBJECT_RE = re.compile(r"\(#(\d+)\)\s*$")
_MERGE_SUBJECT_RE = re.compile(r"^Merge pull request #(\d+)\b")


class GitError(Exception):
    pass


# Every git command carries an explicit identity and a neutral config, and never depends on the host:
# GitHub-hosted runners have NO global identity (PR 49's CI: `git merge ... exited 128: Committer
# identity unknown`), may sign nothing, may have autocrlf set, and may own the checkout as another user.
GIT_IDENTITY = {"GIT_AUTHOR_NAME": "corund-check", "GIT_AUTHOR_EMAIL": "check@corund.dev",
                "GIT_COMMITTER_NAME": "corund-check", "GIT_COMMITTER_EMAIL": "check@corund.dev"}
_NEUTRAL_CONFIG = ("commit.gpgsign=false", "tag.gpgsign=false", "core.autocrlf=false", "safe.directory=*",
                   "core.quotePath=false", "advice.detachedHead=false")


def git_env() -> dict[str, str]:
    """The environment for every git subprocess: the host's PATH etc., an explicit identity (the host's
    GIT_* identity, if any, is NOT trusted — the Action's commits and merges are its own), no prompts."""
    env = {**os.environ, **GIT_IDENTITY, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    return env


def git_command(repo_dir: str, *args: str) -> list[str]:
    """`git -C <repo> -c <neutral config>... <args>` — the ONE builder every call below uses."""
    cmd = ["git", "-C", repo_dir]
    for kv in _NEUTRAL_CONFIG:
        cmd += ["-c", kv]
    return cmd + list(args)


def _run_git(repo_dir: str, *args: str, input_bytes: bytes | None = None) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(git_command(repo_dir, *args), capture_output=True, env=git_env(), input=input_bytes)  # nosec B603 — argv list, shell=False; every argument is a git flag or a SHA this module built


def git_bytes(repo_dir: str, *args: str, check: bool = True, input_bytes: bytes | None = None) -> bytes:
    r = _run_git(repo_dir, *args, input_bytes=input_bytes)
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args[:4])}... exited {r.returncode}: {r.stderr.decode('utf-8', errors='replace').strip()[:600]}")
    return r.stdout


def git(repo_dir: str, *args: str, check: bool = True, input_text: str | None = None) -> str:
    """Text form: decoded with replacement — a Latin-1 source in a diff must never be a traceback. A patch
    that must apply byte-for-byte goes through `git_bytes` / `diff(..., as_bytes=True)` instead."""
    out = git_bytes(repo_dir, *args, check=check, input_bytes=input_text.encode("utf-8", errors="surrogateescape") if input_text is not None else None)
    return out.decode("utf-8", errors="replace")


def rev_parse(repo_dir: str, ref: str) -> str:
    return git(repo_dir, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def has_commit(repo_dir: str, sha: str) -> bool:
    return _run_git(repo_dir, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def merge_base(repo_dir: str, a: str, b: str) -> str:
    return git(repo_dir, "merge-base", a, b).strip()


def diff(repo_dir: str, a: str, b: str, paths: list[str] | None = None, binary: bool = True, renames: bool = False,
         as_bytes: bool = False):
    args = ["diff", "--no-color", "--no-ext-diff"]
    if not renames:
        args.append("--no-renames")
    if binary:
        args.append("--binary")
    args += [a, b]
    if paths:
        args += ["--", *paths]
    return git_bytes(repo_dir, *args) if as_bytes else git(repo_dir, *args)


def changed_files(repo_dir: str, a: str, b: str) -> list[tuple[str, str]]:
    """[(status, path)] between two commits, renames OFF (a rename is D + A); paths are raw (-z)."""
    out = git(repo_dir, "diff", "--name-status", "--no-renames", "-z", a, b)
    parts = out.split("\0")
    rows: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(parts):
        status, path = parts[i], parts[i + 1]
        if status:
            rows.append((status[:1], path))
        i += 2
    return rows


def show_file(repo_dir: str, ref: str, path: str, max_bytes: int = 2_000_000) -> str | None:
    r = _run_git(repo_dir, "show", f"{ref}:{path}")
    if r.returncode != 0 or len(r.stdout) > max_bytes:
        return None
    return r.stdout.decode("utf-8", errors="replace")


def ls_tree_paths(repo_dir: str, ref: str, prefix: str) -> list[str]:
    r = _run_git(repo_dir, "ls-tree", "-r", "--name-only", "-z", ref, "--", prefix)
    return [l for l in r.stdout.decode("utf-8", errors="replace").split("\0") if l] if r.returncode == 0 else []


def tracked_paths(repo_dir: str, *refs: str) -> list[str]:
    """Every path git TRACKS at any of `refs`, as one set.

    SEVENTH CYCLE (verifier 6, F1). C1's own-frame allowlist refuses to affirm when an origin string
    could name more than one tracked file, and it needs the repository's tracked set to ask that.
    `c1_runner` was passing the CHANGED TEST FILES instead, so the ambiguity test was asking about
    the wrong set and could only ever see one candidate.

    Both refs are unioned because the reverted tree is a MIX — the merge-base's non-test files beside
    head's test files — so a name tracked at either ref is a name that can be in play. A larger set
    can only make the allowlist refuse MORE often, never less, which is the safe direction for a
    guard whose failure mode is affirming the wrong file."""
    out: set[str] = set()
    for ref in refs:
        r = _run_git(repo_dir, "ls-tree", "-r", "--name-only", "-z", ref)
        if r.returncode == 0:
            out.update(l for l in r.stdout.decode("utf-8", errors="replace").split("\0") if l)
    return sorted(out)


def worktree_add(repo_dir: str, dest: str, ref: str) -> None:
    git(repo_dir, "worktree", "add", "--detach", "-q", dest, ref)


def strip_git_link(tree_dir: str) -> str:
    """Remove the `.git` link (or dir) from an execution tree so no git oracle exists inside it.
    Returns what was removed, for the receipt."""
    link = os.path.join(tree_dir, ".git")
    if os.path.isfile(link):
        os.remove(link)
        return ".git link removed"
    if os.path.isdir(link):
        shutil.rmtree(link, ignore_errors=True)
        return ".git directory removed"
    return "no .git present"


def worktree_remove(repo_dir: str, dest: str) -> None:
    _run_git(repo_dir, "worktree", "remove", "--force", dest)
    if os.path.isdir(dest):
        shutil.rmtree(dest, ignore_errors=True)
    _run_git(repo_dir, "worktree", "prune")


def worktree_prune(repo_dir: str) -> None:
    _run_git(repo_dir, "worktree", "prune")


def apply_patch(worktree_dir: str, patch) -> None:
    """Apply a patch byte-for-byte (bytes preferred; a str is encoded with surrogateescape)."""
    data = patch if isinstance(patch, bytes) else patch.encode("utf-8", errors="surrogateescape")
    r = _run_git(worktree_dir, "apply", "--whitespace=nowarn", "-", input_bytes=data)
    if r.returncode != 0:
        raise GitError(f"git apply exited {r.returncode}: {r.stderr.decode('utf-8', errors='replace').strip()[:600]}")


def merged_prs(repo_dir: str, branch: str, n: int, scan_limit: int = 5000) -> list[dict]:
    """The last `n` merged PRs on `branch`, newest first, derived from first-parent history: a merge
    commit whose subject is `Merge pull request #N` (base = parent 1, head = parent 2) or a
    squash/rebase commit whose subject ends with `(#N)` (base = parent 1, head = the commit)."""
    out = git(repo_dir, "log", "--first-parent", f"--max-count={scan_limit}", "--format=%H%x00%P%x00%s", branch)
    prs: list[dict] = []
    for line in out.splitlines():
        parts = line.split("\x00")
        if len(parts) != 3:
            continue
        sha, parents_s, subject = parts
        parents = parents_s.split()
        if len(parents) >= 2:
            m = _MERGE_SUBJECT_RE.match(subject)
            prs.append({"sha": sha, "base_sha": parents[0], "head_sha": parents[1], "kind": "merge",
                        "pr": m.group(1) if m else None})
        elif len(parents) == 1:
            m = _PR_SUBJECT_RE.search(subject)
            if m:
                prs.append({"sha": sha, "base_sha": parents[0], "head_sha": sha, "kind": "squash", "pr": m.group(1)})
        if len(prs) >= n:
            break
    return prs
