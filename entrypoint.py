#!/usr/bin/env python3
"""Corund Action entrypoint. `python3 entrypoint.py run ...` — see corund_action/entrypoint.py."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corund_action.entrypoint import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
