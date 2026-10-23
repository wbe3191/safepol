"""Non-learning baselines B0-B3 and B6, sharing a simple controller interface:
    reset(env) -> None      (may replace env.P at episode start)
    act(env, obs) -> Action
"""
from __future__ import annotations

from typing import Optional, Set

import numpy as np

from safepol.bench.policies import allowlist_policy
from safepol.model.policy_model import EndpointPolicy, Policy
from safepol.sim.action_space import Action, K_IDX
from safepol.sim.replay_env import FEATURES, ReplayEnv

NOOP = K_IDX["NOOP"]


class Controller:
    name = "base"

    def reset(self, env: ReplayEnv) -> None:
        pass

    def act(self, env: ReplayEnv, obs: np.ndarray) -> Action:
        return Action(NOOP, 0, env.topo.V, 0)


class B0NoPolicy(Controller):
    """Default-allow cluster (no network policy, unconfined seccomp)."""
    name = "B0-none"

    def reset(self, env):
        env.P = Policy(env.topo)          # no rules at all
        env.P0_AS = max(1, env.P.attack_surface())


class B1Static(Controller):
    """Hand-written allow-list from the operator's dependency graph + RuntimeDefault seccomp."""
    name = "B1-static"

    def reset(self, env):
        env.P = allowlist_policy(env.topo, env.inv, set(env.inv.dep_edges()), "runtime_default")
        env.P0_AS = max(1, env.P.attack_surface())


class B2Observed(Controller):
    """Observation-driven generator (Confine/sysfilter/Hubble-style): allow-list from the flows
    observed in the baseline window, seccomp from observed syscalls; regenerated every
    `refresh` periods from the (unlabeled, alert-filtered) observation window."""
    name = "B2-observed"

    def __init__(self, refresh: int = 20):
        self.refresh = refresh

    def _generate(self, env) -> Policy:
        flows: Set = set(env.inv.dep_edges())
        for s in env.observed_window:
            flows |= s
        return allowlist_policy(env.topo, env.inv, flows, "observed")

    def reset(self, env):
        env.P = self._generate(env)
        env.P0_AS = max(1, env.P.attack_surface())

    def act(self, env, obs):
        if env.t > 0 and env.t % self.refresh == 0:
            env.P = self._generate(env)          # direct regeneration (no agent action)
        return Action(NOOP, 0, env.topo.V, 0)


class B3RuleResponder(Controller):
    """Runtime detection + fixed response playbook (Falco/Tetragon style):
    on a network-related alert quarantine the workload's non-dependency egress and ingress
    (two rules), on a host alert tighten seccomp and drop capabilities, on a frontend
    overload alert rate-limit. No verification: the playbook uses the operator's (possibly
    stale) dependency list, and it never re-opens anything."""
    name = "B3-responder"

    def __init__(self):
        self.queue = []
        self.done: Set[int] = set()

    def reset(self, env):
        self.queue, self.done = [], set()

    def act(self, env, obs):
        if self.queue:
            return self.queue.pop(0)
        X = env.X
        ai, di = FEATURES.index("alerts"), FEATURES.index("danger_sc")
        for w in range(env.topo.n):
            if X[w, ai] > 0 and w not in self.done:
                self.done.add(w)
                if X[w, di] > 0.1:
                    self.queue = [Action(K_IDX["DROP_CAP"], w, env.topo.V, 0)]
                    return Action(K_IDX["TIGHTEN_SC"], w, env.topo.V, 0)
                if w == env.frontend and X[w, FEATURES.index("conns")] > 0.9:
                    return Action(K_IDX["RATE_LIMIT"], w, env.topo.V, 0)
                self.queue = [Action(K_IDX["DENY_IN"], w, env.topo.V, 0)]
                return Action(K_IDX["DENY_OUT"], w, env.topo.V, 0)
        return Action(NOOP, 0, env.topo.V, 0)


class B6ShieldOnly(B2Observed):
    """Observation-driven policy whose every regeneration passes through the Shield
    (no learning). Requires env.cfg.shield = True."""
    name = "B6-shield-only"

    def reset(self, env):
        P = self._generate(env)
        ok, P_safe, _ = env.shield.verify_or_repair(P, env.P)
        env.P = P_safe
        env.P0_AS = max(1, env.P.attack_surface())

    def act(self, env, obs):
        if env.t > 0 and env.t % self.refresh == 0:
            ok, P_safe, _ = env.shield.verify_or_repair(self._generate(env), env.P)
            env.P = P_safe
        return Action(NOOP, 0, env.topo.V, 0)


class B7Hardened(B6ShieldOnly):
    """Strongest static configuration: B6 plus maximal endpoint hardening applied once at
    episode start to every workload (tight seccomp, dropped capabilities, rate limit), each
    step passing through the Shield. Represents 'harden everything the invariants allow'
    without any learning or reaction; isolates the value of adaptivity in SAFEPOL."""
    name = "B7-hardened"

    def reset(self, env):
        super().reset(env)
        self.queue = []
        for w in range(env.topo.n):
            for k in ("TIGHTEN_SC", "DROP_CAP", "RATE_LIMIT"):
                self.queue.append(Action(K_IDX[k], w, env.topo.V, 0))

    def act(self, env, obs):
        if self.queue:
            return self.queue.pop(0)
        if env.t > 0 and env.t % self.refresh == 0:
            # allow-list refresh keeps the endpoint hardening (seccomp / caps / rate limits)
            P = self._generate(env)
            P.endpoint = {w: EndpointPolicy(e.syscalls, e.caps, e.rate_limit) for w, e in env.P.endpoint.items()}
            ok, P_safe, _ = env.shield.verify_or_repair(P, env.P)
            env.P = P_safe
        return Action(NOOP, 0, env.topo.V, 0)
