"""Randomised equivalence test: incremental reachability maintenance == full recomputation."""
import random

from safepol.bench.policies import initial_policy, make_invariants
from safepol.bench.topology import make_port_space, online_boutique, synthetic_topology
from safepol.model.policy_model import ALLOW, DENY, IN, OUT, Policy, Rule


def check(topo, n_ops=200, seed=0):
    rng = random.Random(seed)
    inv = make_invariants(topo) if topo.n < 50 else None
    P = initial_policy(topo, inv) if inv else Policy(topo)
    P.reach()
    for i in range(n_ops):
        op = rng.random()
        if op < 0.6 or not P.rules:
            w = rng.randrange(topo.n)
            k = rng.randint(1, 4)
            peers = frozenset(rng.sample(range(topo.V), k)) - {w}
            ports = rng.choice([topo.ps.ALL, 1 << rng.randrange(topo.ps.K), rng.getrandbits(topo.ps.K)])
            r = Rule(rng.choice([IN, OUT]), frozenset({w}), peers, ports, rng.choice([ALLOW, DENY]), rng.choice([0, 10]), "t")
            P.add_rule(r)
        else:
            P.remove_rule(rng.choice(P.rules).rid)
        if i % 7 == 0:
            assert (P.reach() == P.full_reach()).all(), f"mismatch after op {i}"
    assert (P.reach() == P.full_reach()).all()


if __name__ == "__main__":
    ps = make_port_space()
    check(online_boutique(ps), 300, 0)
    check(online_boutique(ps, True), 300, 1)
    check(synthetic_topology(60, 2), 200, 2)
    print("incremental reach ok")
