"""corund_action — the GitHub Action / CLI around `corund_checks`. It gathers the inputs the pure
checks need (git diffs, a revert-run of the PR's tests, branch protection, reviews), calls
`run_check`, and posts the receipt. It is honest about its own crash: any internal exception
concludes the check runs it owns as CRASHED with the text. NEW package."""
from __future__ import annotations

from . import paths as _paths

_paths.ensure_checks_importable()

__version__ = "0.1.0"
