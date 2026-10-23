"""Initial policy P0, invariants Phi, and static baseline policies for the benchmark."""
from __future__ import annotations

from typing import Dict, Set

from safepol.model.policy_model import (ALLOW, DENY, IN, OUT, DOCKER_DEFAULT_CAPS, SYSCALL_GROUPS,
                                        SYSCALL_UNIVERSE, EndpointPolicy, Policy, Rule, Topology)
from safepol.verify.invariants import Invariants

RUNTIME_DEFAULT_SECCOMP = SYSCALL_UNIVERSE - (SYSCALL_GROUPS["danger"] - {"ptrace", "personality"})
BASELINE_OBSERVED = SYSCALL_GROUPS["base"] | SYSCALL_GROUPS["net"] | SYSCALL_GROUPS["proc"] | SYSCALL_GROUPS["fs"]


def make_invariants(topo: Topology, kappa=None) -> Invariants:
    db = set(topo.workload_idx(lambda w: w.tier == "db"))
    sys = set()  # no workload may touch the metadata service in the boutique
    untrusted = set(topo.workload_idx(lambda w: w.name in ("adservice",)))
    inv = Invariants(topo, db=db, sys=sys, untrusted=untrusted, kappa=kappa)
    inv.legit_flows = set(topo.dep_edges)
    inv.observed_syscalls = {i: set(BASELINE_OBSERVED) for i in range(topo.n)}
    return inv


def initial_policy(topo: Topology, inv: Invariants) -> Policy:
    """Hand-written minimal P0 |= Phi: lock the metadata service and the database from the
    Internet (Cilium 'lockdown' idiom), leave everything else open."""
    P = Policy(topo)
    allw = frozenset(range(topo.n))
    everything = frozenset(range(topo.V))
    meta = topo.by_kind("MetadataService")
    inet = topo.by_kind("Internet")
    P.add_rule(Rule(OUT, allw, everything, topo.ps.ALL, ALLOW, 0, "init"))
    P.add_rule(Rule(OUT, allw, frozenset({meta}), topo.ps.ALL, DENY, 10, "init"))
    for d in inv.db:
        P.add_rule(Rule(IN, frozenset({d}), everything, topo.ps.ALL, ALLOW, 0, "init"))
        P.add_rule(Rule(IN, frozenset({d}), frozenset({inet}), topo.ps.ALL, DENY, 10, "init"))
    for w in range(topo.n):
        sc = SYSCALL_UNIVERSE - inv.forbidden_syscalls if w in inv.untrusted else SYSCALL_UNIVERSE
        P.endpoint[w] = EndpointPolicy(sc, DOCKER_DEFAULT_CAPS)
    assert inv.satisfied(P), inv.check(P)
    return P


def allowlist_policy(topo: Topology, inv: Invariants, flows: Set, seccomp: str = "runtime_default",
                     origin: str = "baseline") -> Policy:
    """Default-deny allow-list derived from a flow set (B1: dependency graph; B2: observations)."""
    P = Policy(topo)
    by_out: Dict[int, Dict[int, int]] = {}
    by_in: Dict[int, Dict[int, int]] = {}
    for (s, d, m) in flows:
        if topo.is_workload(s):
            by_out.setdefault(s, {}).setdefault(d, 0)
            by_out[s][d] |= m
        if topo.is_workload(d):
            by_in.setdefault(d, {}).setdefault(s, 0)
            by_in[d][s] |= m
    # every workload is selected in both directions (default deny), even with no peers
    for w in range(topo.n):
        for d, m in by_out.get(w, {}).items():
            P.add_rule(Rule(OUT, frozenset({w}), frozenset({d}), m, ALLOW, 0, origin))
        if w not in by_out:
            P.add_rule(Rule(OUT, frozenset({w}), frozenset(), 0, ALLOW, 0, origin))
        for s, m in by_in.get(w, {}).items():
            P.add_rule(Rule(IN, frozenset({w}), frozenset({s}), m, ALLOW, 0, origin))
        if w not in by_in:
            P.add_rule(Rule(IN, frozenset({w}), frozenset(), 0, ALLOW, 0, origin))
    for w in range(topo.n):
        if seccomp == "runtime_default":
            sc = RUNTIME_DEFAULT_SECCOMP
        elif seccomp == "observed":
            sc = frozenset(inv.observed_syscalls.get(w, BASELINE_OBSERVED))
        else:
            sc = SYSCALL_UNIVERSE
        if w in inv.untrusted:
            sc = sc - inv.forbidden_syscalls
        P.endpoint[w] = EndpointPolicy(frozenset(sc), DOCKER_DEFAULT_CAPS)
    return P
