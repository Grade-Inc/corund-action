#!/usr/bin/env python3
"""`corund` — the CLI. `corund run ...` (what the Action runs) and `corund replay ...` (read-only
over the last N merged PRs; the onboarding step before any BLOCK opt-in)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corund_action import entrypoint, replay_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "replay":
        return replay_cli.main(argv[1:])
    if argv and argv[0] == "run":
        return entrypoint.main(argv)
    sys.stderr.write("usage: corund run [...] | corund replay [...]\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
