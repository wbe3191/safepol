"""Security and availability invariants Phi over a policy state P (Sec. 1.3 of the design).

Phi_sec  : phi1 Internet !-> db,  phi2 non-sys !-> MetadataService, phi3 no CAP_SYS_ADMIN/PTRACE,
           phi4 untrusted workloads have no dangerous syscalls,  phi5 BR(w) <= kappa
Phi_avail: phi6 observed legitimate flows stay allowed, phi7 observed syscalls stay allowed,
           phi8 service dependency edges stay reachable
All predicates are decidable on the finite domain; reachability-based ones use the cached
incremental matrix of Policy.reach().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from safepol.model.policy_model import (Policy, Topology, DANGEROUS_SYSCALLS)

Flow = Tuple[int, int, int]  # (src, dst, port_mask)


@dataclass
class Violation:
    inv: str
    detail: str
    src: Optional[int] = None
    dst: Optional[int] = None
    port_mask: int = 0
    workload: Optional[int] = None


@dataclass
class Invariants:
    topo: Topology
    db: Set[int] = field(default_factory=set)
    sys: Set[int] = field(default_factory=set)
    untrusted: Set[int] = field(default_factory=set)
    kappa: Optional[int] = None
    forbidden_caps: FrozenSet[str] = frozenset({"CAP_SYS_ADMIN", "CAP_SYS_PTRACE"})
    forbidden_syscalls: FrozenSet[str] = frozenset({"ptrace", "mount", "unshare", "keyctl", "bpf"})
    legit_flows: Set[Flow] = field(default_factory=set)        # phi6 window
    observed_syscalls: Dict[int, Set[str]] = field(default_factory=dict)  # phi7
    use_dep_edges: bool = True                                  # phi8
    dep_edges_override: Optional[List[Flow]] = None             # stale operator graph (drift)
    enabled: FrozenSet[str] = frozenset({"phi1", "phi2", "phi3", "phi4", "phi5", "phi6", "phi7", "phi8"})

    def dep_edges(self) -> List[Flow]:
        return self.dep_edges_override if self.dep_edges_override is not None else list(self.topo.dep_edges)

    # ---- derived sets --------------------------------------------------------------
    def must_deny_pairs(self) -> List[Tuple[int, int, str]]:
        out = []
        t = self.topo
        if "phi1" in self.enabled:
            try:
                inet = t.by_kind("Internet")
                for w in self.db:
                    out.append((inet, w, "phi1"))
            except KeyError:
                pass
        if "phi2" in self.enabled:
            try:
                meta = t.by_kind("MetadataService")
                for w in range(t.n):
                    if w not in self.sys:
                        out.append((w, meta, "phi2"))
            except KeyError:
                pass
        return out

    def must_allow_flows(self) -> List[Tuple[int, int, int, str]]:
        out = []
        if "phi6" in self.enabled:
            for (s, d, m) in self.legit_flows:
                out.append((s, d, m, "phi6"))
        if "phi8" in self.enabled and self.use_dep_edges:
            for (s, d, m) in self.dep_edges():
                out.append((s, d, m, "phi8"))
        return out

    # ---- checking --------------------------------------------------------------------
    def check(self, P: Policy, only: Optional[Set[str]] = None) -> List[Violation]:
        which = self.enabled if only is None else (self.enabled & only)
        viol: List[Violation] = []
        M = P.reach()
        for (u, v, name) in self.must_deny_pairs():
            if name in which and M[u, v] != 0:
                viol.append(Violation(name, f"{self.topo.names[u]} -> {self.topo.names[v]} reachable", u, v, int(M[u, v])))
        if "phi3" in which:
            for w, ep in P.endpoint.items():
                bad = ep.caps & self.forbidden_caps
                if bad:
                    viol.append(Violation("phi3", f"{self.topo.names[w]} holds {sorted(bad)}", workload=w))
        if "phi4" in which:
            for w in self.untrusted:
                bad = P.endpoint[w].syscalls & self.forbidden_syscalls
                if bad:
                    viol.append(Violation("phi4", f"{self.topo.names[w]} allows {sorted(bad)}", workload=w))
        if "phi5" in which and self.kappa is not None:
            br = P.blast_radius()
            for w in range(self.topo.n):
                if br[w] > self.kappa:
                    viol.append(Violation("phi5", f"BR({self.topo.names[w]})={br[w]}>{self.kappa}", workload=w))
        for (u, v, m, name) in self.must_allow_flows():
            if name in which and (M[u, v] & m) != m:
                viol.append(Violation(name, f"{self.topo.names[u]} -> {self.topo.names[v]} blocked", u, v, m))
        if "phi7" in which:
            for w, obs in self.observed_syscalls.items():
                missing = obs - P.endpoint[w].syscalls
                if missing:
                    viol.append(Violation("phi7", f"{self.topo.names[w]} blocks observed {sorted(missing)[:3]}", workload=w))
        return viol

    def satisfied(self, P: Policy) -> bool:
        return not self.check(P)

    def affected_by(self, diff_subjects: Set[int], direction_changed: Set[str]) -> Set[str]:
        """Dependency index rule -> invariants: network rules affect phi1,2,5,6,8 only."""
        return {"phi1", "phi2", "phi5", "phi6", "phi8"} & self.enabled
