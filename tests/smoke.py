"""Smoke tests for the SAFEPOL core: semantics, invariants, conflicts, repair, shield, env."""
import random
import time

import numpy as np

from safepol.bench.policies import allowlist_policy, initial_policy, make_invariants
from safepol.bench.topology import make_port_space, online_boutique
from safepol.model.policy_model import ALLOW, DENY, IN, OUT, Policy, Rule
from safepol.sim.action_space import Action, ActionSpace, K_IDX
from safepol.sim.replay_env import EnvConfig, ReplayEnv
from safepol.verify.conflicts import cross_validate, detect_conflicts, harmful
from safepol.verify.repair import MaxSATRepair, greedy_repair, pso_repair
from safepol.verify.shield import Shield


def test_semantics():
    ps = make_port_space()
    topo = online_boutique(ps)
    inv = make_invariants(topo)
    P0 = initial_policy(topo, inv)
    assert inv.satisfied(P0)
    # full vs incremental reach agree
    assert (P0.full_reach() == P0.reach()).all()
    inet, redis = topo.by_kind("Internet"), topo.index["redis-cart"]
    assert P0.reach()[inet, redis] == 0
    assert P0.reach()[topo.index["frontend"], topo.index["cartservice"]] != 0
    meta = topo.by_kind("MetadataService")
    assert all(P0.reach()[w, meta] == 0 for w in range(topo.n))
    # deny-only selection => default deny (Cilium semantics)
    P = Policy(topo)
    fe, cart = topo.index["frontend"], topo.index["cartservice"]
    P.add_rule(Rule(OUT, frozenset({fe}), frozenset({cart}), ps.ALL, DENY, 10))
    assert P.reach()[fe, topo.index["adservice"]] == 0, "deny-only policy must switch endpoint to default-deny"
    assert (P.full_reach() == P.reach()).all()
    # allow-list baseline satisfies Phi and has small attack surface
    B1 = allowlist_policy(topo, inv, set(topo.dep_edges))
    assert inv.satisfied(B1), inv.check(B1)
    assert B1.attack_surface() < P0.attack_surface()
    print("semantics ok; AS(P0)=%d AS(B1)=%d BR(P0)=%s" % (P0.attack_surface(), B1.attack_surface(), P0.blast_radius()))


def test_conflicts():
    ps = make_port_space()
    topo = online_boutique(ps)
    inv = make_invariants(topo)
    P = initial_policy(topo, inv)
    fe, cart, ad = topo.index["frontend"], topo.index["cartservice"], topo.index["adservice"]
    # allow fe->cart:7070 shadowed by deny fe->cart all ports
    P.add_rule(Rule(OUT, frozenset({fe}), frozenset({cart}), ps.mask_of_name("cart-7070"), ALLOW, 0))
    P.add_rule(Rule(OUT, frozenset({fe}), frozenset({cart, ad}), ps.ALL, DENY, 10))
    c = detect_conflicts(P, "fast")
    kinds = sorted(x.kind for x in c)
    print("conflicts:", kinds)
    assert "shadowing" in kinds
    agree, nf, ns = cross_validate(P)
    assert agree, (nf, ns)
    print("SMT cross-validation ok (%d anomalies)" % nf)


def test_repair():
    ps = make_port_space()
    topo = online_boutique(ps)
    inv = make_invariants(topo)
    P0 = initial_policy(topo, inv)
    asp = ActionSpace(topo)
    fe = topo.index["frontend"]
    # over-aggressive: deny frontend egress to everything -> breaks phi8
    a = Action(K_IDX["DENY_OUT"], fe, asp.WILD, 0)
    Pc = asp.apply(P0, a)
    assert not inv.satisfied(Pc)
    for name, fn in [("maxsat", lambda: MaxSATRepair(inv, 500).repair(Pc)),
                     ("greedy", lambda: greedy_repair(Pc, inv)),
                     ("pso", lambda: pso_repair(Pc, inv))]:
        res = fn()
        ok = res.policy is not None and inv.satisfied(res.policy)
        print(f"repair[{name}] status={res.status} ok={ok} d={res.distance} lat={res.latency*1000:.1f}ms")
        assert name == "pso" or ok
        if ok:
            assert not harmful(detect_conflicts(res.policy, "fast"))
    # ALLOW to metadata -> phi2 violation -> repair should drop/deny it
    a = Action(K_IDX["ALLOW_OUT"], fe, topo.by_kind("MetadataService"), 0)
    Pc = asp.apply(P0, a)
    res = MaxSATRepair(inv, 500).repair(Pc)
    assert res.policy is not None and inv.satisfied(res.policy), res.status
    print("repair phi2 ok d=%d" % res.distance)


def test_env(steps=100):
    cfg = EnvConfig(seed=1)
    env = ReplayEnv(cfg)
    rng = random.Random(0)
    obs, info = env.reset(options={"scenario": "S2"})
    assert obs.shape == (env.obs_dim,)
    t0 = time.perf_counter()
    for t in range(steps):
        m = env.masks()
        km = m.k_mask(); k = rng.choice(np.flatnonzero(km))
        wm = m.w_mask(k); w = rng.choice(np.flatnonzero(wm))
        pm = m.p_mask(k, w); p = rng.choice(np.flatnonzero(pm))
        pim = m.pi_mask(k, w, p); pi = rng.choice(np.flatnonzero(pim))
        obs, r, done, trunc, info = env.step(Action(int(k), int(w), int(p), int(pi)))
        assert np.isfinite(r)
        if done:
            break
    dt = time.perf_counter() - t0
    s = env.summary()
    print("env random policy: %.1f ms/step; summary=%s" % (dt / (t + 1) * 1000, {k: s[k] for k in ("FBR", "MBR", "inv_violations", "cand_violations", "interventions", "ASR", "TTM", "rules_final")}))
    assert s["inv_violations"] == 0
    # no-shield env must be able to violate
    env2 = ReplayEnv(EnvConfig(seed=2, shield=False, masks=False))
    env2.reset(options={"scenario": "S4"})
    ad = env2.topo.index["adservice"]          # untrusted: relaxing seccomp violates phi4
    env2.step(Action(K_IDX["RELAX_SC"], ad, env2.topo.V, 0))
    assert env2.summary()["inv_violations"] == 1, env2.summary()
    print("no-shield violation recorded ok")


if __name__ == "__main__":
    test_semantics(); test_conflicts(); test_repair(); test_env()
    print("ALL OK")
