"""A stdlib (urllib) GitHub REST client for exactly what the Action needs — create/complete check
runs, upsert one PR comment, and two read-only GETs the replay uses (branch protection, PR
reviews) — plus an in-memory Fake that every test uses. No test ever performs a network call.

The token arrives via the environment (CORUND_GITHUB_TOKEN), never argv (a secret never
passes as a CLI argument), and never appears in any error text this module raises. NEW module."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

API_VERSION = "2022-11-28"


class GitHubApiError(Exception):
    def __init__(self, method: str, path: str, status: int | None, body: str):
        self.status = status
        super().__init__(f"{method} {path} -> {status}: {body[:300]}")


class GitHubApi:
    def __init__(self, *, token: str, repo: str, api_base: str = "https://api.github.com", timeout_s: int = 30):
        if not token:
            raise ValueError("a token is required (env CORUND_GITHUB_TOKEN)")
        if not repo or "/" not in repo:
            raise ValueError("repo must be owner/name")
        self._token, self.repo, self.api_base, self.timeout_s = token, repo, api_base.rstrip("/"), timeout_s

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict | list | None]:
        url = f"{self.api_base}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self._token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION, "User-Agent": "corund-action",
            **({"Content-Type": "application/json"} if data is not None else {}),
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 — https to the configured API base  # nosec B310 — https to the configured API base; no caller supplies a scheme
                raw = resp.read().decode("utf-8")
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise GitHubApiError(method, path, exc.code, raw.replace(self._token, "<token>")) from None
        except urllib.error.URLError as exc:
            raise GitHubApiError(method, path, None, str(exc.reason)) from None

    # --- checks -------------------------------------------------------------------------------
    def create_check_run(self, *, name: str, head_sha: str, status: str, output: dict | None = None,
                         conclusion: str | None = None, details_url: str | None = None) -> int:
        body = {"name": name, "head_sha": head_sha, "status": status}
        if output:
            body["output"] = output
        if conclusion:
            body["conclusion"] = conclusion
        if details_url:
            body["details_url"] = details_url
        _, data = self._request("POST", f"/repos/{self.repo}/check-runs", body)
        return int(data["id"])  # type: ignore[index]

    def update_check_run(self, *, check_run_id: int, status: str, conclusion: str | None, output: dict,
                         details_url: str | None = None) -> None:
        body = {"status": status, "output": output}
        if conclusion:
            body["conclusion"] = conclusion
        if details_url:
            body["details_url"] = details_url
        self._request("PATCH", f"/repos/{self.repo}/check-runs/{check_run_id}", body)

    # --- PR comment --------------------------------------------------------------------------
    def upsert_pr_comment(self, *, pr_number: int, marker: str, body: str) -> str:
        _, comments = self._request("GET", f"/repos/{self.repo}/issues/{pr_number}/comments?per_page=100")
        for c in comments or []:  # type: ignore[union-attr]
            if isinstance(c, dict) and marker in (c.get("body") or ""):
                self._request("PATCH", f"/repos/{self.repo}/issues/comments/{c['id']}", {"body": body})
                return "updated"
        self._request("POST", f"/repos/{self.repo}/issues/{pr_number}/comments", {"body": body})
        return "created"

    # --- read-only GETs (replay) ------------------------------------------------------------
    def get_branch_protection(self, branch: str) -> dict | None:
        try:
            _, data = self._request("GET", f"/repos/{self.repo}/branches/{branch}/protection")
        except GitHubApiError as exc:
            if exc.status == 404:
                return None
            raise
        return data if isinstance(data, dict) else None

    def get_pr_reviews(self, pr_number: int) -> list[dict]:
        _, data = self._request("GET", f"/repos/{self.repo}/pulls/{pr_number}/reviews?per_page=100")
        out = []
        for r in data or []:  # type: ignore[union-attr]
            if isinstance(r, dict):
                out.append({"reviewer_id": (r.get("user") or {}).get("id"), "commit_sha": r.get("commit_id"),
                            "state": r.get("state"), "submitted_at": r.get("submitted_at")})
        return out

    def get_pr(self, pr_number: int) -> dict:
        _, data = self._request("GET", f"/repos/{self.repo}/pulls/{pr_number}")
        return data if isinstance(data, dict) else {}


class FakeGitHubApi:
    """In-memory stand-in. Records every call; `fail_on` makes a named method raise, so the
    Action's own crash path can be exercised without a network."""

    def __init__(self, *, fail_on: set[str] | None = None, protection: dict | None = None,
                 reviews: dict[int, list] | None = None):
        self.calls: list[tuple] = []
        self.check_runs: dict[int, dict] = {}
        self.comments: dict[int, list[dict]] = {}
        self.fail_on = fail_on or set()
        self.protection = protection
        self.reviews = reviews or {}
        self._next = 100
        self.repo = "o/r"

    def _maybe_fail(self, name: str):
        if name in self.fail_on:
            raise GitHubApiError("X", name, 403, "forbidden by the fake")

    def create_check_run(self, *, name, head_sha, status, output=None, conclusion=None, details_url=None) -> int:
        self._maybe_fail("create_check_run")
        self._next += 1
        self.check_runs[self._next] = {"name": name, "head_sha": head_sha, "status": status, "output": output,
                                       "conclusion": conclusion, "details_url": details_url}
        self.calls.append(("create_check_run", self._next, name, status))
        return self._next

    def update_check_run(self, *, check_run_id, status, conclusion, output, details_url=None) -> None:
        self._maybe_fail("update_check_run")
        self.check_runs[check_run_id].update(status=status, conclusion=conclusion, output=output)
        self.calls.append(("update_check_run", check_run_id, status, conclusion))

    def upsert_pr_comment(self, *, pr_number, marker, body) -> str:
        self._maybe_fail("upsert_pr_comment")
        lst = self.comments.setdefault(pr_number, [])
        for c in lst:
            if marker in c["body"]:
                c["body"] = body
                self.calls.append(("comment", pr_number, "updated"))
                return "updated"
        lst.append({"body": body})
        self.calls.append(("comment", pr_number, "created"))
        return "created"

    def get_branch_protection(self, branch):
        self._maybe_fail("get_branch_protection")
        self.calls.append(("get_branch_protection", branch))
        return self.protection

    def get_pr_reviews(self, pr_number):
        self._maybe_fail("get_pr_reviews")
        self.calls.append(("get_pr_reviews", pr_number))
        return list(self.reviews.get(pr_number, []))

    def get_pr(self, pr_number):
        return {}
