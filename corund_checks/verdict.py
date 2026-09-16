"""The verdict vocabulary of the four checks, and the invariants that make "a crash is never green"
true by construction rather than by convention.

Per the check contract (design rulings of 2026-09-05):

  State    = PROVEN | FAILED | UNPROVEN | GAMED_SUSPECT | CRASHED | NOT_RUN
  C1 speaks PROVEN / UNPROVEN(reason) / CRASHED / NOT_RUN — never FAILED, and (since the owner's
     decision of 2026-09-15) never GAMED_SUSPECT: see ALLOWED_STATES below.
  C2-C4 speak PROVEN / FAILED / GAMED_SUSPECT / CRASHED / NOT_RUN — never UNPROVEN.
  reason   in {"compile","collection","flaky","green-on-revert","contaminated"}, REQUIRED iff state == UNPROVEN.
           ("contaminated" — owner ruling v2, 2026-09-05: PR-authored test infrastructure ran during the
           reverted phase, so a witness was refused for safety; the receipt names the file.)
  error    the exception's own text, REQUIRED iff state == CRASHED.
  evidence human-readable, SHA-pinned lines; may be empty only for NOT_RUN.

Prose forms are hyphenated ("UNPROVEN-collection", "GAMED-SUSPECT"); code forms use underscores.
NEW module, no ported antecedent.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

CheckId = Literal["C1", "C2", "C3", "C4"]
State = Literal["PROVEN", "FAILED", "UNPROVEN", "GAMED_SUSPECT", "CRASHED", "NOT_RUN"]
UnprovenReason = Literal["compile", "collection", "flaky", "green-on-revert", "contaminated"]

CHECK_IDS: tuple[CheckId, ...] = ("C1", "C2", "C3", "C4")
CHECK_NAMES: dict[str, str] = {
    "C1": "red-on-revert", "C2": "skip-audit", "C3": "gate-fold", "C4": "approval-SHA binding",
}
STATES: tuple[State, ...] = ("PROVEN", "FAILED", "UNPROVEN", "GAMED_SUSPECT", "CRASHED", "NOT_RUN")
UNPROVEN_REASONS: tuple[UnprovenReason, ...] = ("compile", "collection", "flaky", "green-on-revert", "contaminated")

# The states each check is allowed to speak. Enforced in Verdict.__post_init__ and re-checked by
# the runner, so a check cannot drift into a vocabulary its consumers do not expect.
#
# C1 LOST GAMED_SUSPECT on 2026-09-15 (the owner's decision, measured, not stylistic). The field
# sweep of 1,079 merged PRs across 27 repositories put C2's false-accusation rate at 6.58% and its
# accusatory precision at 12.3%, against gates of <3% and >=85%, across nine systematic idiom classes. Each of those classes reached a user in one
# of exactly two ways: as C2's own verdict, or as a C2 gaming marker that turned C1's verdict into
# GAMED_SUSPECT -- C1's ONLY accusatory verdict. Corund now ships C1 alone, and C1 does not accuse:
# where it cannot PROVE, it withholds with a named reason.
#
# This row is the ENFORCEMENT of that, not a description of it. The Action passes `gaming_markers={}`
# so no gamed path is reachable; that is good behaviour by one caller. The owner's ruling
# of 2026-09-07: "Guards are enforced by the core, never left to the caller ... a guard that only
# holds when the caller passes an optional field is not enforcement." With this row, the App, the
# replay CLI and any future caller are refused too -- in Verdict.__post_init__, and again in the
# runner, where the refusal becomes CRASHED and never a green.
#
# CRASHED STAYS. The absolute rule -- "Corund never reports a check it did not run. CRASHED
# is its own verdict ... never folded into PROVEN or FAILED" -- is untouched by the pivot.
# C2/C3/C4 are FROZEN, not removed: their rows do not move.
ALLOWED_STATES: dict[str, frozenset[str]] = {
    "C1": frozenset({"PROVEN", "UNPROVEN", "CRASHED", "NOT_RUN"}),
    "C2": frozenset({"PROVEN", "FAILED", "GAMED_SUSPECT", "CRASHED", "NOT_RUN"}),
    "C3": frozenset({"PROVEN", "FAILED", "GAMED_SUSPECT", "CRASHED", "NOT_RUN"}),
    "C4": frozenset({"PROVEN", "FAILED", "GAMED_SUSPECT", "CRASHED", "NOT_RUN"}),
}

# States that mean "this check would have flagged the PR" (replay accounting, block posture).
FLAGGING_STATES: frozenset[str] = frozenset({"FAILED", "GAMED_SUSPECT"})


@dataclasses.dataclass(frozen=True)
class Verdict:
    check: CheckId
    state: State
    base_sha: str | None
    head_sha: str | None
    approval_sha: str | None
    evidence: tuple[str, ...]
    error: str | None = None
    reason: UnprovenReason | None = None

    def __post_init__(self) -> None:
        if self.check not in ALLOWED_STATES:
            raise ValueError(f"unknown check id {self.check!r}; expected one of {CHECK_IDS}")
        if self.state not in STATES:
            raise ValueError(f"unknown state {self.state!r}; expected one of {STATES}")
        if self.state not in ALLOWED_STATES[self.check]:
            raise ValueError(f"{self.check} may not speak {self.state}; allowed: "
                             f"{sorted(ALLOWED_STATES[self.check])}")
        if self.state == "UNPROVEN":
            if self.reason not in UNPROVEN_REASONS:
                raise ValueError(f"UNPROVEN requires reason in {UNPROVEN_REASONS}, got {self.reason!r}")
        elif self.reason is not None:
            raise ValueError(f"reason is only allowed for UNPROVEN, got {self.reason!r} on {self.state}")
        if self.state == "CRASHED":
            if not self.error:
                raise ValueError("CRASHED requires the exception's own text in `error`")
        elif self.error is not None:
            raise ValueError(f"error is only allowed for CRASHED, got {self.error!r} on {self.state}")
        if not isinstance(self.evidence, tuple):
            object.__setattr__(self, "evidence", tuple(self.evidence))
        if not self.evidence and self.state != "NOT_RUN":
            raise ValueError(f"{self.state} requires at least one evidence line")
        if any(not isinstance(line, str) for line in self.evidence):
            raise ValueError("every evidence line must be a str")

    @property
    def display_state(self) -> str:
        """The prose form: UNPROVEN-<reason>, GAMED-SUSPECT, else the state itself."""
        if self.state == "UNPROVEN":
            return f"UNPROVEN-{self.reason}"
        return self.state.replace("_", "-")

    @property
    def flagged(self) -> bool:
        return self.state in FLAGGING_STATES

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["evidence"] = list(self.evidence)
        d["display_state"] = self.display_state
        d["check_name"] = CHECK_NAMES[self.check]
        return d


class MissingInput(Exception):
    """Raised by a check when a required input is absent or unusable — the runner turns it into
    NOT_RUN with the missing names in the evidence. Never a pass."""

    def __init__(self, *names: str):
        self.names = tuple(names)
        super().__init__(", ".join(names))
