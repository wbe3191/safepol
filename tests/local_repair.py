"""Localised MaxSAT repair must produce a policy that is (i) safe and (ii) no farther from the
candidate than the global repair, over random single-workload edits of safe policies."""
import random
import time

from safepol.bench.policies import allowlist_policy, initial_policy, make_invariants
from safepol.bench.topology import make_port_space, online_boutique
from safepol.sim.action_space import ACTION_TYPES, Action, ActionSpace, K_IDX
from safepol.verify.conflicts import detect_conflicts, harmful
from safepol.verify.repair import MaxSATRepair

ps = make_port_space()
topo = online_boutique(ps)
inv = make_invariants(topo)
asp = ActionSpace(topo)
rep = MaxSATRepair(inv, 2000)
rng = random.Random(7)
starts = [initial_policy(topo, inv), allowlist_policy(topo, inv, set(topo.dep_edges))]
n_viol = n_same = 0
t_loc = t_glob = 0.0
for i in range(150):
    P = rng.choice(starts).copy()
    for _ in range(rng.randint(0, 4)):          # a few prior safe/unsafe edits
        a = Action(rng.choice([K_IDX[k] for k in ("DENY_IN", "DENY_OUT", "ALLOW_IN", "ALLOW_OUT", "ISOLATE")]),
                   rng.randrange(topo.n), rng.randrange(topo.V + 1), rng.randrange(3))
        P = asp.apply(P, a, None, inv)
    a = Action(rng.choice([K_IDX[k] for k in ("DENY_IN", "DENY_OUT", "ISOLATE")]),
               rng.randrange(topo.n), rng.randrange(topo.V + 1), rng.randrange(3))
    Pc = asp.apply(P, a, None, inv)
    viol = inv.check(Pc)
    if not viol and not harmful(detect_conflicts(Pc, "fast")):
        continue
    n_viol += 1
    affected = {a.w} | {w for v in viol for w in (v.src, v.dst, v.workload) if w is not None and topo.is_workload(w)}
    t0 = time.perf_counter(); rl = rep.repair(Pc, affected); t_loc += time.perf_counter() - t0
    t0 = time.perf_counter(); rg = rep.repair(Pc, None); t_glob += time.perf_counter() - t0
    assert rg.policy is not None, rg.status
    if rl.policy is None:
        print("local infeasible -> would widen:", rl.status, a.describe(topo)); continue
    assert inv.satisfied(rl.policy) and not harmful(detect_conflicts(rl.policy, "fast"))
    assert rl.distance <= rg.distance + 1e-9, (rl.distance, rg.distance, a.describe(topo))
    n_same += rl.distance == rg.distance
print(f"unsafe candidates={n_viol} same_distance={n_same} local={t_loc/n_viol*1000:.1f}ms global={t_glob/n_viol*1000:.1f}ms")
print("localised repair ok")
