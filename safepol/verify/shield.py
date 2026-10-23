"""The Shield: monotone action masks (Lemma 2) and verify-or-repair (Algorithm 2)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from safepol.model.policy_model import IN, OUT, Policy
from safepol.sim.action_space import ACTION_TYPES, K_IDX, Action, ActionSpace, DROPPABLE_CAPS, SYSCALL_GROUPS
from safepol.verify.conflicts import detect_conflicts, harmful
from safepol.verify.invariants import Invariants
from safepol.verify.repair import MaxSATRepair, RepairResult, greedy_repair, pso_repair


@dataclass
class ShieldStats:
    decisions: int = 0
    verified_ok: int = 0
    repaired: int = 0
    rejected: int = 0           # fell back to P_t
    conflicts_fixed: int = 0
    by_method: Dict[str, int] = field(default_factory=dict)
    distances: List[int] = field(default_factory=list)
    verify_latency: List[float] = field(default_factory=list)
    repair_latency: List[float] = field(default_factory=list)
    violations_blocked: Dict[str, int] = field(default_factory=dict)

    def intervention_rate(self) -> float:
        return (self.repaired + self.rejected) / max(1, self.decisions)


class Shield:
    def __init__(self, inv: Invariants, aspace: ActionSpace, repair: str = "maxsat",
                 timeout_ms: int = 200, enabled: bool = True, conflict_check: bool = True,
                 prefer_additions: bool = False):
        self.inv = inv
        self.aspace = aspace
        self.repair_method = repair
        self.enabled = enabled
        self.conflict_check = conflict_check
        self.maxsat = MaxSATRepair(inv, timeout_ms, prefer_additions) if repair == "maxsat" else None
        self.stats = ShieldStats()

    # -------------------------------------------------------------------------------
    # Lemma 2: cheap monotone masks.  Returns per-head masks as a dict that the policy
    # network can consult: mask_k[k], mask_w[k][w], mask_p[k][w][p] (lazy), mask_pi.
    # -------------------------------------------------------------------------------
    def mask(self, P: Policy, history_len: int = 0) -> "ActionMask":
        return ActionMask(self, P, history_len)

    def action_feasible(self, P: Policy, a: Action, history_len: int = 0) -> bool:
        """Cheap structural pre-check used for masking.  Conservative: True = not obviously bad."""
        topo, inv = self.inv.topo, self.inv
        kind = a.kind
        w = a.w
        if kind == "NOOP":
            return True
        if kind == "ROLLBACK":
            return history_len > 0
        if kind == "REMOVE":
            return P.endpoint[w].rate_limit is not None or \
                any(r.origin in ("agent", "repair") and w in r.subject for r in P.rules)
        if kind == "TIGHTEN_SC":
            obs = inv.observed_syscalls.get(w, set())
            return bool(P.endpoint[w].syscalls & SYSCALL_GROUPS["danger"]) and not (obs & SYSCALL_GROUPS["danger"])
        if kind == "RELAX_SC":
            return w not in inv.untrusted and not (P.endpoint[w].syscalls >= SYSCALL_GROUPS["danger"])
        if kind == "DROP_CAP":
            return bool(P.endpoint[w].caps & DROPPABLE_CAPS)
        if kind == "RATE_LIMIT":
            return P.endpoint[w].rate_limit is None
        if kind == "ISOLATE":
            return True
        # network actions
        if a.p >= topo.V:
            return True
        p = a.p
        ports = self.aspace.port_mask(a)
        if kind == "DENY_OUT":
            # would cut a dependency / legit flow of w -> p
            for (s, d, m, _) in self.inv.must_allow_flows():
                if s == w and d == p and (m & ports):
                    return False
            return True
        if kind == "DENY_IN":
            for (s, d, m, _) in self.inv.must_allow_flows():
                if s == p and d == w and (m & ports):
                    return False
            return True
        if kind == "ALLOW_OUT":
            return not any(u == w and v == p for (u, v, _) in self.inv.must_deny_pairs())
        if kind == "ALLOW_IN":
            return not any(u == p and v == w for (u, v, _) in self.inv.must_deny_pairs())
        return True

    # -------------------------------------------------------------------------------
    # Algorithm 2
    # -------------------------------------------------------------------------------
    def verify_or_repair(self, P_cand: Policy, P_t: Policy, changed: Optional[set] = None) -> Tuple[bool, Policy, Dict]:
        """Returns (candidate_was_safe, P_safe, info).  ``changed``: workloads whose rules differ
        between P_t and P_cand (the action's subjects); when given, conflict detection and the
        MaxSAT encoding are localised to them plus the workloads named in the violations."""
        st = self.stats
        st.decisions += 1
        info: Dict = {"violations": [], "conflicts": 0, "method": None, "distance": 0}
        t0 = time.perf_counter()
        viol = self.inv.check(P_cand)
        confs = harmful(detect_conflicts(P_cand, "fast", changed)) if self.conflict_check else []
        st.verify_latency.append(time.perf_counter() - t0)
        info["violations"] = [v.inv for v in viol]
        info["conflicts"] = len(confs)
        for v in viol:
            st.violations_blocked[v.inv] = st.violations_blocked.get(v.inv, 0) + 1
        if not viol and not confs:
            st.verified_ok += 1
            return True, P_cand, info
        if not self.enabled:
            # Shield disabled (ablation / unconstrained baselines): pass through unsafe policy.
            return False, P_cand, info
        affected = None
        if changed is not None:
            affected = set(changed)
            for v in viol:
                for w in (v.src, v.dst, v.workload):
                    if w is not None and self.inv.topo.is_workload(w):
                        affected.add(w)
        res = self._repair(P_cand, affected)
        st.repair_latency.append(res.latency)
        if res.policy is not None:
            st.repaired += 1
            st.by_method[res.method] = st.by_method.get(res.method, 0) + 1
            st.distances.append(res.distance)
            if confs and not viol:
                st.conflicts_fixed += 1
            info["method"], info["distance"] = res.method, res.distance
            return False, res.policy, info
        st.rejected += 1
        info["method"] = "reject"
        return False, P_t.copy(), info

    def _repair(self, P_cand: Policy, affected: Optional[set] = None) -> RepairResult:
        if self.repair_method == "maxsat":
            res = self.maxsat.repair(P_cand, affected)
            if res.policy is None and affected is not None and res.status != "timeout":
                res = self.maxsat.repair(P_cand, None)      # widen to the whole policy once
            if res.policy is not None:
                return res
            g = greedy_repair(P_cand, self.inv)
            g.method = "greedy(fallback:%s)" % res.status
            return g
        if self.repair_method == "pso":
            return pso_repair(P_cand, self.inv)
        return greedy_repair(P_cand, self.inv)


class ActionMask:
    """Lazy per-head masks for the factorized policy."""

    def __init__(self, shield: Shield, P: Policy, history_len: int):
        self.shield, self.P, self.hist = shield, P, history_len
        asp = shield.aspace
        self.nK, self.nW, self.nP, self.nPi = asp.nK, asp.nW, asp.nP, asp.nPi

    def k_mask(self) -> np.ndarray:
        """An action type is selectable only if at least one subject is feasible for it.
        Without this, a type whose effect is exhausted (e.g. TIGHTEN_SC once every seccomp
        profile is tight) degenerates into a penalised pseudo-NOOP that the policy can collapse on."""
        m = np.ones(self.nK, dtype=bool)
        m[K_IDX["ROLLBACK"]] = self.hist > 0
        m[K_IDX["REMOVE"]] = any(r.origin in ("agent", "repair") for r in self.P.rules) or \
            any(e.rate_limit is not None for e in self.P.endpoint.values())
        for k in range(self.nK):
            if m[k] and ACTION_TYPES[k] not in ("NOOP", "ROLLBACK"):
                m[k] = self._w_feasible(k).any()
        m[K_IDX["NOOP"]] = True
        return m

    def _w_feasible(self, k: int) -> np.ndarray:
        if not hasattr(self, "_wcache"):
            self._wcache = {}
        if k not in self._wcache:
            m = np.zeros(self.nW, dtype=bool)
            for w in range(self.nW):
                m[w] = self.shield.action_feasible(self.P, Action(k, w, self.shield.aspace.WILD, 0), self.hist)
            self._wcache[k] = m
        return self._wcache[k]

    def w_mask(self, k: int) -> np.ndarray:
        if ACTION_TYPES[k] in ("NOOP", "ROLLBACK"):
            m = np.zeros(self.nW, dtype=bool); m[0] = True   # subject irrelevant -> single choice
            return m
        m = self._w_feasible(k).copy()
        if not m.any():
            m[:] = True   # only reachable when masks are bypassed (k chosen without k_mask)
        return m

    def p_mask(self, k: int, w: int) -> np.ndarray:
        m = np.ones(self.nP, dtype=bool)
        kind = ACTION_TYPES[k]
        if kind in ("DENY_IN", "DENY_OUT", "ALLOW_IN", "ALLOW_OUT", "REMOVE"):
            for p in range(self.nP - 1):
                if p == w:
                    m[p] = False
                    continue
                m[p] = self.shield.action_feasible(self.P, Action(k, w, p, 0), self.hist)
        else:
            m[:-1] = False  # peer irrelevant -> force wildcard
        if not m.any():
            m[-1] = True
        return m

    def pi_mask(self, k: int, w: int, p: int) -> np.ndarray:
        m = np.ones(self.nPi, dtype=bool)
        kind = ACTION_TYPES[k]
        if kind not in ("DENY_IN", "DENY_OUT", "ALLOW_IN", "ALLOW_OUT") or p >= self.nP - 1:
            m[1:] = False
        else:
            for pi in range(self.nPi):
                m[pi] = self.shield.action_feasible(self.P, Action(k, w, p, pi), self.hist)
            if not m.any():
                m[0] = True
        return m
