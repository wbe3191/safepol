"""Curriculum learning schedule (Appendix A): stage -> environment settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

STAGES = [
    # (fraction of training budget, scenarios, drift, chaos)
    # every stage exercises both the network-level (S2) and the host-level (S1) response class, so that
    # no action family is introduced only after the policy has already committed to another one
    (0.15, ["S1", "S2"], False, False),
    (0.20, ["S1", "S2", "S3", "S4", "S6", "S7"], False, False),
    (0.25, ["S1", "S2", "S3", "S4", "S5", "S6", "S7"], False, False),
    (0.20, ["S1", "S2", "S3", "S4", "S5", "S6", "S7"], True, False),
    (0.20, ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"], True, True),
]


@dataclass
class Curriculum:
    stages: List = field(default_factory=lambda: list(STAGES))
    enabled: bool = True
    current: int = -1

    def apply(self, env, progress: float) -> None:
        if not self.enabled:
            if self.current < 0:
                env.cfg.scenarios = list(self.stages[-1][1]); env.cfg.drift = self.stages[-1][2]
                env.cfg.chaos = self.stages[-1][3]; self.current = 0
            return
        acc, idx = 0.0, 0
        for i, (frac, *_rest) in enumerate(self.stages):
            acc += frac
            if progress < acc:
                idx = i
                break
        else:
            idx = len(self.stages) - 1
        if idx != self.current:
            _, sc, drift, chaos = self.stages[idx]
            env.cfg.scenarios = list(sc); env.cfg.drift = drift; env.cfg.chaos = chaos
            self.current = idx
