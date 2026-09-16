"""Locate `corund_checks` in both trees: the monorepo (../checks next to action/) and the public
mirror (corund_checks vendored beside action.yml). Fails loudly if neither is present — a caller
never degrades to a fake."""
from __future__ import annotations

import sys
from pathlib import Path

ACTION_DIR = Path(__file__).resolve().parents[1]


def ensure_checks_importable() -> None:
    try:
        import corund_checks  # noqa: F401
        return
    except ImportError:
        pass
    for cand in (ACTION_DIR.parent / "checks", ACTION_DIR):
        if (cand / "corund_checks" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            break
    import corund_checks  # noqa: F401  — raises ImportError loudly if still absent
