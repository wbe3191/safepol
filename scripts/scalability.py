"""RQ5 scalability: verification / repair latency vs. topology size on synthetic call graphs.
usage: PYTHONPATH=. python scripts/scalability.py 100 500 1000 5000
Writes results/scalability.csv (one row per (n, repeat, component))."""
import os
import random
import sys
import time

import numpy as np
import pandas as pd

from safepol.bench.policies import allowlist_policy, make_invariants
from safepol.bench.topology import synthetic_topology
from safepol.model.policy_model import ALLOW, DENY, OUT, Rule
from safepol.sim.action_space import Action, ActionSpace, K_IDX
from safepol.verify.conflicts import detect_conflicts, harmful
from safepol.verify.invariants import Invariants
from safepol.verify.repair import MaxSATRepair, greedy_repair, pso_repair

sizes = [int(x) for x in sys.argv[1:]] or [100, 500, 1000, 5000]
REPEATS = 5
rows = []


def rec(n, comp, rep, lat, **kw):
    rows.append(dict(n=n, component=comp, repeat=rep, latency_ms=lat * 1000, **kw))
    print(f"n={n:5d} {comp:28s} rep={rep} {lat*1000:9.2f} ms {kw}", flush=True)


for n in sizes:
    topo = synthetic_topology(n, seed=0)
    inv = make_invariants(topo)
    asp = ActionSpace(topo)
    P = allowlist_policy(topo, inv, set(topo.dep_edges))
    rng = random.Random(n)
    deps = list(topo.dep_edges)
    for rep in range(REPEATS):
        # (1) full reachability recomputation (oracle)
        t0 = time.perf_counter(); P._reach = None; P._mats = None; P.reach(); rec(n, "reach_full", rep, time.perf_counter() - t0, rules=len(P.rules))
        # (2) incremental update after one rule
        w = rng.randrange(topo.n)
        r = Rule(OUT, frozenset({w}), frozenset({rng.randrange(topo.V)}), topo.ps.ALL, DENY, 10, "bench")
        t0 = time.perf_counter(); P.add_rule(r); P.reach(); rec(n, "reach_incremental", rep, time.perf_counter() - t0)
        P.remove_rule(r.rid)
        # (3) invariant check (phi1,2,3,4,6,7,8) on the whole policy
        t0 = time.perf_counter(); v = inv.check(P); rec(n, "invariant_check", rep, time.perf_counter() - t0, violations=len(v))
        # (4) conflict detection: global vs localised to one workload
        t0 = time.perf_counter(); c = detect_conflicts(P, "fast"); rec(n, "conflicts_global", rep, time.perf_counter() - t0, found=len(c))
        t0 = time.perf_counter(); c = detect_conflicts(P, "fast", {w}); rec(n, "conflicts_local", rep, time.perf_counter() - t0, found=len(c))
        # (5) shield decision on a violating action: DENY_OUT cutting a dependency edge (phi8)
        (s, d, m) = rng.choice([e for e in deps if topo.is_workload(e[0])])
        a = Action(K_IDX["DENY_OUT"], s, d, 0)
        Pc = asp.apply(P, a, None, inv)
        t0 = time.perf_counter(); viol = inv.check(Pc); t_v = time.perf_counter() - t0
        affected = {s} | {x for vv in viol for x in (vv.src, vv.dst) if x is not None and topo.is_workload(x)}
        rec(n, "verify_candidate", rep, t_v, violations=len(viol))
        ms = MaxSATRepair(inv, 5000)
        res = ms.repair(Pc, affected); rec(n, "repair_maxsat_local", rep, res.latency, status=res.status, distance=res.distance, vars=res.stats.get("vars"))
        if n <= 1000:
            res = ms.repair(Pc, None); rec(n, "repair_maxsat_global", rep, res.latency, status=res.status, distance=res.distance, vars=res.stats.get("vars"))
        res = greedy_repair(Pc, inv); rec(n, "repair_greedy", rep, res.latency, status=res.status, distance=res.distance)
        if n <= 1000:
            res = pso_repair(Pc, inv); rec(n, "repair_pso", rep, res.latency, status=res.status, distance=res.distance)
        # (6) policy copy (per decision, with copy-on-write caches)
        t0 = time.perf_counter(); Q = P.copy(); rec(n, "policy_copy", rep, time.perf_counter() - t0)
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/scalability.csv", index=False)
print(pd.DataFrame(rows).groupby(["n", "component"]).latency_ms.median().unstack(0).round(2).to_string())
