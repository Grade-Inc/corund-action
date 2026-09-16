"""corund_checks — the four deterministic checks Corund runs on a pull request, as pure functions
over input snapshots. No network, no git, no model judgment inside this package: callers build the
snapshots (see checks/README.md for the exact keys) and `run_check` never raises.

    C1 red-on-revert      C2 skip-audit      C3 gate-fold      C4 approval-SHA binding

Absolute rule: Corund never reports a check it did not run. CRASHED carries the
exception's own text; NOT_RUN names the missing input; neither is ever PROVEN.

NAMING, fixed 2026-09-09 after a production crash on app.corund.dev (`replay CRASHED` on the
dashboard and onboarding, `TypeError: 'module' object is not callable`): `replay` below is the
SUBMODULE (`checks/corund_checks/replay.py`), not the engine callable. `from . import replay`
binds the package attribute `replay` to the imported module object; that module's own top-level
`def replay(items)` function lives at `replay.replay`, one dotted hop further in, exactly as
`action/corund_action/replay_cli.py` (`from corund_checks import replay as engine`, then
`engine.replay(...)`, `engine.summarize(...)`, `engine.render_report(...)`) and
`checks/tests/test_replay.py` already call it — real, working callers that a same-named callable
export at the package level would have broken by shadowing the module they depend on.
`corund_checks.replay(...)`, called as if `replay` were the function, calls the MODULE instead and
raises the TypeError above. `run_replay`, `ReplayItem`, and `ReplayRow` are the engine's callable
and its input/output dataclasses, exported under names that never collide with the submodule, so a
caller that wants the replay CONTRACT -- build a `ReplayItem` per PR, call `run_replay`, read a
`ReplayRow` back -- never needs to resolve the module-vs-attribute question, or reach through the
submodule at all, to get it. Each is `replay.<name>` itself, never a reimplementation.
"""
from __future__ import annotations

from .verdict import (
    ALLOWED_STATES, CHECK_IDS, CHECK_NAMES, FLAGGING_STATES, STATES, UNPROVEN_REASONS, CheckId,
    MissingInput, State, UnprovenReason, Verdict,
)
from .runner import INPUT_SCHEMA, OPTIONAL_INPUTS, run_check
from . import replay

# Unambiguous callable/dataclass aliases -- see the NAMING note above. A caller that wants the
# replay contract never has to know `replay` is a submodule to reach it.
run_replay = replay.replay
ReplayItem = replay.ReplayItem
ReplayRow = replay.ReplayRow

__version__ = "0.1.0"

__all__ = [
    "ALLOWED_STATES", "CHECK_IDS", "CHECK_NAMES", "FLAGGING_STATES", "INPUT_SCHEMA",
    "OPTIONAL_INPUTS", "STATES", "UNPROVEN_REASONS", "CheckId", "MissingInput", "State",
    "UnprovenReason", "Verdict", "run_check", "replay", "run_replay", "ReplayItem", "ReplayRow",
    "__version__",
]
