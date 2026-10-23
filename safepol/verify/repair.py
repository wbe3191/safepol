"""Minimal policy repair: Partial MaxSAT (Z3 Optimize), greedy fallback and PSO (prior work).

Given a candidate policy P_cand and invariants Phi, find P' with P' |= Phi, Conf(P') = {} and
minimal edit distance d(P_cand, P').  Edit operations:
  * drop a rule                           (cost 1)
  * exempt a peer from a deny rule        (cost 1, "narrowing" a deny; Cilium deny-wins means an
                                           allow exception cannot override a deny)
  * add an allow rule for a blocked flow  (cost 1)
  * add a deny rule for a must-deny pair  (cost 1)
Endpoint invariants (phi3, phi4, phi7) are repaired directly (drop caps / syscalls).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from safepol.model.policy_model import ALLOW, DENY, IN, OUT, Policy, Rule
from safepol.verify.conflicts import detect_conflicts, harmful, SHADOW, CORREL
from safepol.verify.invariants import Invariants


@dataclass
class RepairResult:
    policy: Optional[Policy]
    distance: int
    method: str
    latency: float
    status: str = "ok"          # ok | timeout | infeasible
    stats: Dict = field(default_factory=dict)


# -------------------------------------------------------------------------------------
# endpoint (non-network) repair
# -------------------------------------------------------------------------------------
def repair_endpoint(P: Policy, inv: Invariants) -> Tuple[Policy, int, bool]:
    """Fix phi3/phi4/phi7 directly. Returns (P', edits, feasible)."""
    P = P.copy()
    edits = 0
    feasible = True
    for w, ep in P.endpoint.items():
        caps = ep.caps
        if "phi3" in inv.enabled and caps & inv.forbidden_caps:
            caps = caps - inv.forbidden_caps
            edits += 1
        sc = ep.syscalls
        obs = inv.observed_syscalls.get(w, set())
        if "phi7" in inv.enabled and not obs <= sc:
            sc = sc | obs
            edits += 1
        if "phi4" in inv.enabled and w in inv.untrusted and sc & inv.forbidden_syscalls:
            if "phi7" in inv.enabled and obs & inv.forbidden_syscalls:
                feasible = False  # phi4 and phi7 contradict: cannot repair
            sc = sc - inv.forbidden_syscalls
            edits += 1
        P.endpoint[w] = type(ep)(sc, caps, ep.rate_limit)
    return P, edits, feasible


def _companion_allow_needed(P: Policy, w: int, direction: str) -> bool:
    return not P._selected(w, direction)


# -------------------------------------------------------------------------------------
# MaxSAT repair
# -------------------------------------------------------------------------------------
class MaxSATRepair:
    def __init__(self, inv: Invariants, timeout_ms: int = 200, prefer_additions: bool = False):
        import z3
        self.z3 = z3
        self.inv = inv
        self.timeout_ms = timeout_ms
        self.prefer_additions = prefer_additions

    def repair(self, P_cand: Policy, affected: Optional[Set[int]] = None) -> RepairResult:
        """MaxSAT-minimal repair.  ``affected``: optional set of workloads whose rules may be edited
        (the subjects changed by the candidate action plus the workloads named in its violations).
        Rules of other workloads are constants of the encoding and only flows / pairs touching an
        affected workload are encoded (Sec. 2.2 dependency index); with affected=None the whole
        policy is encoded.  Soundness: a flow not touching `affected` has the same verdict as in
        P_t |= Phi, and the result is post-checked against the full invariant set anyway."""
        z3, inv, topo = self.z3, self.inv, self.inv.topo
        t0 = time.perf_counter()
        P, ep_edits, feasible = repair_endpoint(P_cand, inv)
        if not feasible:
            return RepairResult(None, 0, "maxsat", time.perf_counter() - t0, "infeasible")
        all_rules = P.rules
        K = topo.ps.K
        if affected is None:
            rules = all_rules
            fixed_rules: List[Rule] = []
            must_allow = inv.must_allow_flows()
            must_deny = inv.must_deny_pairs()
            in_scope = lambda w: True
        else:
            A = set(affected)
            rules = [r for r in all_rules if r.subject & A]
            fixed_rules = [r for r in all_rules if not (r.subject & A)]
            must_allow = [f for f in inv.must_allow_flows() if f[0] in A or f[1] in A]
            must_deny = [p for p in inv.must_deny_pairs() if p[0] in A or p[1] in A]
            in_scope = lambda w: w in A
        # relevant peers for exemptions: peers appearing in must-allow flows
        rel_peers: Set[int] = set()
        for (u, v, m, _) in must_allow:
            rel_peers.update((u, v))

        opt = z3.Optimize()
        opt.set("timeout", self.timeout_ms)
        keep = {r.rid: z3.Bool(f"k{r.rid}") for r in rules}
        ex: Dict[Tuple[int, int], object] = {}
        for r in rules:
            if r.verdict == DENY:
                for p in r.peer & rel_peers:
                    ex[(r.rid, p)] = z3.Bool(f"x{r.rid}_{p}")
        add_out: Dict[Tuple[int, int, int], object] = {}
        add_in: Dict[Tuple[int, int, int], object] = {}
        for (u, v, m, _) in must_allow:
            if topo.is_workload(u) and in_scope(u):
                add_out[(u, v, m)] = z3.Bool(f"ao{u}_{v}_{m}")
            if topo.is_workload(v) and in_scope(v):
                add_in[(u, v, m)] = z3.Bool(f"ai{u}_{v}_{m}")
        add_deny: Dict[Tuple[int, int], object] = {}
        for (u, v, _) in must_deny:
            if (topo.is_workload(u) and in_scope(u)) or (topo.is_workload(v) and in_scope(v)):
                add_deny[(u, v)] = z3.Bool(f"ad{u}_{v}")

        # soft clauses (edit distance)
        for r in rules:
            opt.add_soft(keep[r.rid], 1, "dist")
        for var in ex.values():
            opt.add_soft(z3.Not(var), 1, "dist")
        for var in list(add_out.values()) + list(add_in.values()) + list(add_deny.values()):
            opt.add_soft(z3.Not(var), 1, "dist")
        if self.prefer_additions:
            # optional lexicographically lower objective that breaks edit-distance ties towards
            # additions: removing a rule can *unselect* an endpoint (Cilium semantics), which opens it
            # to everything, whereas adding a single-pair rule only narrows or opens one pair.  Off in
            # the simulator experiments (plain edit distance); on in the cluster adapter.
            for r in rules:
                opt.add_soft(keep[r.rid], 1, "removals")

        # per-workload rule index over the encoded rules
        by_w: Dict[Tuple[str, int], List[Rule]] = {}
        for r in rules:
            for w in r.subject:
                by_w.setdefault((r.direction, w), []).append(r)

        def deny_on_egress(u, v):
            return (u, v) in add_deny and topo.is_workload(u) and in_scope(u)

        def deny_on_ingress(u, v):
            return (u, v) in add_deny and not deny_on_egress(u, v)

        def sel(w: int, direction: str):
            terms = [keep[r.rid] for r in by_w.get((direction, w), [])]
            if direction == OUT:
                terms += [v_ for (u, v, m), v_ in add_out.items() if u == w]
                terms += [v_ for (u, v), v_ in add_deny.items() if u == w and deny_on_egress(u, v)]
            else:
                terms += [v_ for (u, v, m), v_ in add_in.items() if v == w]
                terms += [v_ for (u, v), v_ in add_deny.items() if v == w and deny_on_ingress(u, v)]
            return z3.Or(terms) if terms else z3.BoolVal(False)

        def allowed(u: int, v: int, c: int):
            """Is port class c allowed from u to v under the decision variables?"""
            bit = 1 << c
            parts = []
            # egress side of u
            if topo.is_workload(u):
                if in_scope(u):
                    eg_allow = [keep[r.rid] for r in by_w.get((OUT, u), []) if r.verdict == ALLOW
                                and v in r.peer and r.ports & bit]
                    eg_allow += [var for (uu, vv, m), var in add_out.items() if uu == u and vv == v and m & bit]
                    eg_deny = []
                    for r in by_w.get((OUT, u), []):
                        if r.verdict == DENY and v in r.peer and r.ports & bit:
                            x = ex.get((r.rid, v))
                            eg_deny.append(z3.And(keep[r.rid], z3.Not(x)) if x is not None else keep[r.rid])
                    if deny_on_egress(u, v):
                        eg_deny.append(add_deny[(u, v)])
                    parts.append(z3.Not(z3.Or(eg_deny)) if eg_deny else z3.BoolVal(True))
                    parts.append(z3.Or(z3.Not(sel(u, OUT)), z3.Or(eg_allow) if eg_allow else z3.BoolVal(False)))
                else:
                    parts.append(z3.BoolVal(bool(P.egress_ports(u, v) & bit)))
            if topo.is_workload(v):
                if in_scope(v):
                    in_allow = [keep[r.rid] for r in by_w.get((IN, v), []) if r.verdict == ALLOW
                                and u in r.peer and r.ports & bit]
                    in_allow += [var for (uu, vv, m), var in add_in.items() if uu == u and vv == v and m & bit]
                    in_deny = []
                    for r in by_w.get((IN, v), []):
                        if r.verdict == DENY and u in r.peer and r.ports & bit:
                            x = ex.get((r.rid, u))
                            in_deny.append(z3.And(keep[r.rid], z3.Not(x)) if x is not None else keep[r.rid])
                    if deny_on_ingress(u, v):
                        in_deny.append(add_deny[(u, v)])
                    parts.append(z3.Not(z3.Or(in_deny)) if in_deny else z3.BoolVal(True))
                    parts.append(z3.Or(z3.Not(sel(v, IN)), z3.Or(in_allow) if in_allow else z3.BoolVal(False)))
                else:
                    parts.append(z3.BoolVal(bool(P.ingress_ports(u, v) & bit)))
            return z3.And(parts) if parts else z3.BoolVal(True)

        def class_groups(u: int, v: int, mask: int) -> List[int]:
            """Representative port classes: classes with identical rule coverage for (u,v) are
            equivalent in the encoding, so one representative per signature suffices."""
            rel = [r for r in by_w.get((OUT, u), []) if v in r.peer] + [r for r in by_w.get((IN, v), []) if u in r.peer]
            fixed_eg = P.egress_ports(u, v) if (topo.is_workload(u) and not in_scope(u)) else None
            fixed_in = P.ingress_ports(u, v) if (topo.is_workload(v) and not in_scope(v)) else None
            sigs: Dict[Tuple, int] = {}
            for c in range(K):
                if not (mask >> c & 1):
                    continue
                bit = 1 << c
                sig = tuple(bool(r.ports & bit) for r in rel)
                sig = sig + tuple(bool(m & bit) for (uu, vv, m) in add_out if uu == u and vv == v) \
                          + tuple(bool(m & bit) for (uu, vv, m) in add_in if uu == u and vv == v)
                sig = sig + (None if fixed_eg is None else bool(fixed_eg & bit),
                             None if fixed_in is None else bool(fixed_in & bit))
                sigs.setdefault(sig, c)
            return list(sigs.values())

        # hard: must-allow flows
        for (u, v, m, _) in must_allow:
            for c in class_groups(u, v, m):
                opt.add(allowed(u, v, c))
        # hard: must-deny pairs (all classes)
        for (u, v, _) in must_deny:
            for c in class_groups(u, v, topo.ps.ALL):
                opt.add(z3.Not(allowed(u, v, c)))
        # hard: no harmful conflicts among kept rules
        by_id = {r.rid: r for r in rules}
        for cf in harmful(detect_conflicts(P, "fast", None if affected is None else set(affected))):
            if cf.hi not in by_id or cf.lo not in by_id:
                continue
            hi, lo = by_id[cf.hi], by_id[cf.lo]
            overlap = hi.peer & lo.peer
            exvars = [ex.get((hi.rid, p)) for p in overlap] if hi.verdict == DENY else [None]
            if all(x is not None for x in exvars) and exvars:
                opt.add(z3.Implies(z3.And(keep[hi.rid], keep[lo.rid]), z3.And(exvars)))
            else:
                opt.add(z3.Or(z3.Not(keep[hi.rid]), z3.Not(keep[lo.rid])))

        res = opt.check()
        lat = time.perf_counter() - t0
        if res != z3.sat:
            return RepairResult(None, 0, "maxsat", lat, "timeout" if res == z3.unknown else "infeasible")
        mdl = opt.model()
        tv = lambda b: z3.is_true(mdl.eval(b, model_completion=True))
        P2 = Policy(topo, [r.copy() for r in fixed_rules], {k: v for k, v in P.endpoint.items()})
        d = ep_edits
        for r in rules:
            if not tv(keep[r.rid]):
                d += 1
                continue
            r2 = r.copy()
            if r.verdict == DENY:
                exempted = {p for p in r.peer if (r.rid, p) in ex and tv(ex[(r.rid, p)])}
                if exempted:
                    r2.peer = frozenset(r.peer - exempted)
                    r2.origin = "repair"
                    d += len(exempted)
                    if not r2.peer:
                        continue
            P2.rules.append(r2)
        for (u, v, m), var in add_out.items():
            if tv(var):
                P2.rules.append(Rule(OUT, frozenset({u}), frozenset({v}), m, ALLOW, 0, "repair")); d += 1
        for (u, v, m), var in add_in.items():
            if tv(var):
                P2.rules.append(Rule(IN, frozenset({v}), frozenset({u}), m, ALLOW, 0, "repair")); d += 1
        for (u, v), var in add_deny.items():
            if tv(var):
                if deny_on_egress(u, v):
                    P2.rules.append(Rule(OUT, frozenset({u}), frozenset({v}), topo.ps.ALL, DENY, 10, "repair"))
                else:
                    P2.rules.append(Rule(IN, frozenset({v}), frozenset({u}), topo.ps.ALL, DENY, 10, "repair"))
                d += 1
        P2.version = P_cand.version
        lat = time.perf_counter() - t0
        ok = inv.satisfied(P2) and not harmful(detect_conflicts(P2, "fast"))
        return RepairResult(P2 if ok else None, d, "maxsat", lat, "ok" if ok else "postcheck_failed",
                            {"vars": len(keep) + len(ex) + len(add_out) + len(add_in) + len(add_deny)})


# -------------------------------------------------------------------------------------
# Greedy repair (fallback, upper bound on d)
# -------------------------------------------------------------------------------------
def greedy_repair(P_cand: Policy, inv: Invariants, max_iter: int = 50) -> RepairResult:
    t0 = time.perf_counter()
    topo = inv.topo
    P, d, feasible = repair_endpoint(P_cand, inv)
    if not feasible:
        return RepairResult(None, 0, "greedy", time.perf_counter() - t0, "infeasible")
    for _ in range(max_iter):
        viol = inv.check(P)
        confs = harmful(detect_conflicts(P, "fast"))
        if not viol and not confs:
            P.version = P_cand.version
            return RepairResult(P, d, "greedy", time.perf_counter() - t0)
        if viol:
            v = viol[0]
            u, w = v.src, v.dst
            if v.inv in ("phi1", "phi2"):
                if topo.is_workload(u):
                    if _companion_allow_needed(P, u, OUT):
                        P.add_rule(Rule(OUT, frozenset({u}), frozenset(range(topo.V)), topo.ps.ALL, ALLOW, 0, "repair")); d += 1
                    P.add_rule(Rule(OUT, frozenset({u}), frozenset({w}), topo.ps.ALL, DENY, 10, "repair"))
                else:
                    if _companion_allow_needed(P, w, IN):
                        P.add_rule(Rule(IN, frozenset({w}), frozenset(range(topo.V)), topo.ps.ALL, ALLOW, 0, "repair")); d += 1
                    P.add_rule(Rule(IN, frozenset({w}), frozenset({u}), topo.ps.ALL, DENY, 10, "repair"))
                d += 1
            elif v.inv in ("phi6", "phi8"):
                m = v.port_mask
                fixed = False
                # 1) narrow a matching deny
                for r in list(P.rules):
                    if r.verdict != DENY or not (r.ports & m):
                        continue
                    if r.direction == OUT and u in r.subject and w in r.peer:
                        P.remove_rule(r.rid); r2 = r.copy(); r2.peer = frozenset(r.peer - {w}); r2.origin = "repair"
                        if r2.peer: P.add_rule(r2)
                        d += 1; fixed = True
                    elif r.direction == IN and w in r.subject and u in r.peer:
                        P.remove_rule(r.rid); r2 = r.copy(); r2.peer = frozenset(r.peer - {u}); r2.origin = "repair"
                        if r2.peer: P.add_rule(r2)
                        d += 1; fixed = True
                if not fixed:
                    # 2) default-deny: add allow on the selected side(s)
                    if topo.is_workload(u) and P._selected(u, OUT) and not (P.egress_ports(u, w) & m) == m:
                        P.add_rule(Rule(OUT, frozenset({u}), frozenset({w}), m, ALLOW, 0, "repair")); d += 1
                    if topo.is_workload(w) and P._selected(w, IN) and not (P.ingress_ports(u, w) & m) == m:
                        P.add_rule(Rule(IN, frozenset({w}), frozenset({u}), m, ALLOW, 0, "repair")); d += 1
            elif v.inv == "phi5":
                break
        elif confs:
            c = confs[0]
            P.remove_rule(c.lo); d += 1
    return RepairResult(None, d, "greedy", time.perf_counter() - t0, "infeasible")


def pso_conflict_resolve(P_cand: Policy, n_particles: int = 20, iters: int = 20, seed: int = 0) -> Tuple[Policy, float]:
    """Prior-work style conflict resolution: PSO over rule subsets minimising Boolean-matrix
    conflicts (no invariants).  Returns (policy, latency)."""
    t0 = time.perf_counter()
    confs = harmful(detect_conflicts(P_cand, "fast"))
    if not confs:
        return P_cand, time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    R = len(P_cand.rules)

    def build(mask):
        return Policy(P_cand.topo, [r.copy() for r, k in zip(P_cand.rules, mask) if k], dict(P_cand.endpoint))

    def fitness(mask):
        return 100 * len(harmful(detect_conflicts(build(mask), "fast"))) + (R - mask.sum())

    X = (rng.random((n_particles, R)) < 0.9).astype(float); X[0] = 1.0
    Vel = rng.normal(0, 0.5, (n_particles, R))
    pbest = X.copy(); pfit = np.array([fitness(x > 0.5) for x in X])
    g = int(pfit.argmin()); gbest, gfit = pbest[g].copy(), pfit[g]
    for _ in range(iters):
        r1, r2 = rng.random((n_particles, R)), rng.random((n_particles, R))
        Vel = 0.7 * Vel + 1.5 * r1 * (pbest - X) + 1.5 * r2 * (gbest - X)
        X = (rng.random((n_particles, R)) < 1 / (1 + np.exp(-Vel))).astype(float)
        fit = np.array([fitness(x > 0.5) for x in X])
        better = fit < pfit; pbest[better] = X[better]; pfit[better] = fit[better]
        g = int(pfit.argmin())
        if pfit[g] < gfit:
            gbest, gfit = pbest[g].copy(), pfit[g]
        if gfit < 100:
            break
    P = build(gbest > 0.5); P.version = P_cand.version
    return P, time.perf_counter() - t0


# -------------------------------------------------------------------------------------
# PSO repair (re-implementation of the heuristic used in the prior ASEJ'26 system)
# -------------------------------------------------------------------------------------
def pso_repair(P_cand: Policy, inv: Invariants, n_particles: int = 20, iters: int = 30,
               seed: int = 0) -> RepairResult:
    """Binary PSO over keep/drop decisions for each rule (plus endpoint repair).
    Fitness = 1000*#violations + 100*#harmful conflicts + #drops.  No exemptions / additions:
    the search space is the one the original work used (rule subsets)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    P0, d0, feasible = repair_endpoint(P_cand, inv)
    if not feasible:
        return RepairResult(None, 0, "pso", time.perf_counter() - t0, "infeasible")
    R = len(P0.rules)
    if R == 0:
        ok = inv.satisfied(P0)
        return RepairResult(P0 if ok else None, d0, "pso", time.perf_counter() - t0, "ok" if ok else "infeasible")

    def build(mask: np.ndarray) -> Policy:
        return Policy(inv.topo, [r.copy() for r, k in zip(P0.rules, mask) if k], dict(P0.endpoint))

    def fitness(mask: np.ndarray) -> float:
        P = build(mask)
        return 1000 * len(inv.check(P)) + 100 * len(harmful(detect_conflicts(P, "fast"))) + (R - mask.sum())

    X = (rng.random((n_particles, R)) < 0.8).astype(float)
    X[0] = 1.0
    Vel = rng.normal(0, 0.5, (n_particles, R))
    pbest = X.copy(); pfit = np.array([fitness(x > 0.5) for x in X])
    g = int(pfit.argmin()); gbest = pbest[g].copy(); gfit = pfit[g]
    w, c1, c2 = 0.7, 1.5, 1.5
    for _ in range(iters):
        r1, r2 = rng.random((n_particles, R)), rng.random((n_particles, R))
        Vel = w * Vel + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        prob = 1 / (1 + np.exp(-Vel))
        X = (rng.random((n_particles, R)) < prob).astype(float)
        fit = np.array([fitness(x > 0.5) for x in X])
        better = fit < pfit
        pbest[better] = X[better]; pfit[better] = fit[better]
        g = int(pfit.argmin())
        if pfit[g] < gfit:
            gbest, gfit = pbest[g].copy(), pfit[g]
        if gfit < 100:
            break
    P = build(gbest > 0.5)
    P.version = P_cand.version
    ok = gfit < 100
    return RepairResult(P if ok else None, d0 + int(R - (gbest > 0.5).sum()), "pso",
                        time.perf_counter() - t0, "ok" if ok else "infeasible")
