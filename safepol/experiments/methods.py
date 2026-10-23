"""Method registry: every compared system is an (env config, agent factory) pair."""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Optional

from safepol.sim.replay_env import EnvConfig
from safepol.sim.curriculum import Curriculum, STAGES


@dataclasses.dataclass
class MethodSpec:
    name: str
    kind: str                       # ppo | dqn | static
    env: Dict                       # EnvConfig overrides
    agent: Dict = dataclasses.field(default_factory=dict)   # PPOConfig / DQNConfig overrides
    curriculum: bool = True
    controller: Optional[str] = None   # for static baselines
    group: str = "main"


FULL_SCENARIOS = list(STAGES[-1][1])

METHODS: Dict[str, MethodSpec] = {
    # ---- learned systems -------------------------------------------------------
    "SAFEPOL": MethodSpec("SAFEPOL", "ppo", dict(shield=True, repair="maxsat", masks=True, graph_features=True),
                          dict(lagrangian=True, warm_start_steps=20_000)),
    "B4-DQN-PSO": MethodSpec("B4-DQN-PSO", "dqn", dict(shield=False, masks=False, conflict_resolver="pso"),
                             {}, curriculum=False, group="baseline"),
    "B5-PPO": MethodSpec("B5-PPO", "ppo", dict(shield=False, masks=False), dict(lagrangian=False, warm_start_steps=20_000),
                         curriculum=False, group="baseline"),
    # ---- ablations ------------------------------------------------------------------
    "A-noWarmStart": MethodSpec("A-noWarmStart", "ppo", dict(shield=True, masks=True), dict(lagrangian=True), group="ablation"),
    "A-noShield": MethodSpec("A-noShield", "ppo", dict(shield=False, masks=True), dict(lagrangian=True, warm_start_steps=20_000), group="ablation"),
    "A-noCost": MethodSpec("A-noCost", "ppo", dict(shield=True, masks=True), dict(lagrangian=False, warm_start_steps=20_000), group="ablation"),
    "A-noMask": MethodSpec("A-noMask", "ppo", dict(shield=True, masks=False), dict(lagrangian=True, warm_start_steps=20_000), group="ablation"),
    "A-flat": MethodSpec("A-flat", "ppo", dict(shield=True, masks=True), dict(lagrangian=True, flat=True, warm_start_steps=20_000), group="ablation"),
    "A-noGraph": MethodSpec("A-noGraph", "ppo", dict(shield=True, masks=True, graph_features=False), dict(lagrangian=True, warm_start_steps=20_000), group="ablation"),
    "A-noCurriculum": MethodSpec("A-noCurriculum", "ppo", dict(shield=True, masks=True), dict(lagrangian=True, warm_start_steps=20_000), curriculum=False, group="ablation"),
    "A-PSOrepair": MethodSpec("A-PSOrepair", "ppo", dict(shield=True, repair="pso", masks=True), dict(lagrangian=True, warm_start_steps=20_000), group="ablation"),
    "A-greedyRepair": MethodSpec("A-greedyRepair", "ppo", dict(shield=True, repair="greedy", masks=True), dict(lagrangian=True, warm_start_steps=20_000), group="ablation"),
    "A-detector": MethodSpec("A-detector", "ppo", dict(shield=True, masks=True, reward_source="detector"), dict(lagrangian=True, warm_start_steps=20_000), group="ablation"),
    # ---- non-learning baselines -------------------------------------------------------
    "B0-none": MethodSpec("B0-none", "static", dict(shield=False, masks=False), controller="B0NoPolicy", group="baseline"),
    "B1-static": MethodSpec("B1-static", "static", dict(shield=False, masks=False), controller="B1Static", group="baseline"),
    "B2-observed": MethodSpec("B2-observed", "static", dict(shield=False, masks=False), controller="B2Observed", group="baseline"),
    "B3-responder": MethodSpec("B3-responder", "static", dict(shield=False, masks=False), controller="B3RuleResponder", group="baseline"),
    "B6-shield-only": MethodSpec("B6-shield-only", "static", dict(shield=True, masks=False), controller="B6ShieldOnly", group="baseline"),
    "B7-hardened": MethodSpec("B7-hardened", "static", dict(shield=True, masks=False), controller="B7Hardened", group="baseline"),
    "B8-shielded-runbook": MethodSpec("B8-shielded-runbook", "static", dict(shield=True, masks=False), controller="B3RuleResponder", group="baseline"),
    # hybrid: shielded runbook responds to alerts, the constrained learner acts on all other steps
    "SAFEPOL-H": MethodSpec("SAFEPOL-H", "ppo", dict(shield=True, repair="maxsat", masks=True, graph_features=True, runbook_prior=True),
                            dict(lagrangian=True, warm_start_steps=20_000)),
}


def make_env_config(spec: MethodSpec, seed: int, **extra) -> EnvConfig:
    d = dict(scenarios=list(FULL_SCENARIOS), seed=seed)
    d.update(spec.env); d.update(extra)
    return EnvConfig(**d)
