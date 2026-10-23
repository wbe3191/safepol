"""Factorized action space a = (k, w, p, pi) and its application P ⊕ a (Sec. 1.5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from safepol.model.policy_model import (ALLOW, DENY, IN, OUT, DANGEROUS_CAPS, SYSCALL_GROUPS,
                                        EndpointPolicy, Policy, Rule, Topology)

ACTION_TYPES = ["NOOP", "DENY_IN", "DENY_OUT", "ALLOW_IN", "ALLOW_OUT", "REMOVE",
                "TIGHTEN_SC", "RELAX_SC", "DROP_CAP", "RATE_LIMIT", "ISOLATE", "ROLLBACK"]
K_IDX = {k: i for i, k in enumerate(ACTION_TYPES)}
PORT_TEMPLATES = ["ALL", "SVC", "NONSVC"]   # pi: all ports | peer's service ports | complement
NETWORK_TYPES = {"DENY_IN", "DENY_OUT", "ALLOW_IN", "ALLOW_OUT"}
DROPPABLE_CAPS = frozenset({"CAP_NET_RAW", "CAP_DAC_OVERRIDE", "CAP_SETUID", "CAP_SETGID",
                            "CAP_SYS_CHROOT", "CAP_MKNOD", "CAP_NET_BIND_SERVICE", "CAP_FOWNER"}) | DANGEROUS_CAPS


@dataclass(frozen=True)
class Action:
    k: int
    w: int
    p: int      # endpoint index or V (= wildcard '*')
    pi: int

    @property
    def kind(self) -> str:
        return ACTION_TYPES[self.k]

    def describe(self, topo: Topology) -> str:
        p = "*" if self.p >= topo.V else topo.names[self.p]
        return f"{self.kind}({topo.names[self.w]}, {p}, {PORT_TEMPLATES[self.pi]})"


class ActionSpace:
    def __init__(self, topo: Topology):
        self.topo = topo
        self.nK = len(ACTION_TYPES)
        self.nW = topo.n
        self.nP = topo.V + 1
        self.nPi = len(PORT_TEMPLATES)
        self.WILD = topo.V
        # service port mask per endpoint (externals: all ports of class "other"/named)
        self.svc_mask = np.zeros(topo.V, dtype=np.int64)
        for i, w in enumerate(topo.workloads):
            m = 0
            for pn in w.ports:
                proto, port = topo.ps.named_ports[pn]
                m |= topo.ps.mask_of_port(proto, port)
            self.svc_mask[i] = m
        for i in range(topo.n, topo.V):
            # externals: "service" ports = everything that is not ephemeral
            self.svc_mask[i] = topo.ps.ALL & ~topo.ps.mask_of_range(0, 32768, 65535) & ~topo.ps.mask_of_range(1, 32768, 65535)

    def flat_size(self) -> int:
        return self.nK * self.nW * self.nP * self.nPi

    def port_mask(self, a: Action) -> int:
        ps = self.topo.ps
        if a.pi == 0 or a.p >= self.topo.V:
            return ps.ALL
        svc = int(self.svc_mask[a.p]) or ps.ALL
        return svc if a.pi == 1 else (ps.ALL & ~svc)

    def peers(self, a: Action, inv=None) -> frozenset:
        if a.p >= self.topo.V:
            peers = set(i for i in range(self.topo.V) if i != a.w)
            if inv is not None and a.kind.startswith("DENY"):
                # wildcard template for deny = "everything except known dependencies"
                for (s, d, m, _) in inv.must_allow_flows():
                    if a.kind == "DENY_OUT" and s == a.w:
                        peers.discard(d)
                    elif a.kind == "DENY_IN" and d == a.w:
                        peers.discard(s)
            return frozenset(peers)
        return frozenset({a.p})

    # ----------------------------------------------------------------------------
    def apply(self, P: Policy, a: Action, history: Optional[List[Policy]] = None, inv=None) -> Policy:
        """Return P ⊕ a (a fresh Policy; P is not modified)."""
        topo = self.topo
        kind = a.kind
        if kind == "ROLLBACK":
            if history:
                return history[-1].copy()
            return P.copy()
        Q = P.copy()
        w = a.w
        if kind == "NOOP":
            return Q
        if kind in NETWORK_TYPES:
            direction = IN if kind.endswith("_IN") else OUT
            verdict = DENY if kind.startswith("DENY") else ALLOW
            peers = self.peers(a, inv)
            if not peers:
                return Q
            ports = self.port_mask(a)
            if verdict == DENY and not Q._selected(w, direction):
                # Cilium "lockdown" idiom: deny X + allow all, so that the deny does not
                # implicitly switch the endpoint into default-deny for every other peer.
                Q.add_rule(Rule(direction, frozenset({w}), frozenset(i for i in range(topo.V) if i != w),
                                topo.ps.ALL, ALLOW, 0, "agent"))
            Q.add_rule(Rule(direction, frozenset({w}), peers, ports, verdict,
                            10 if verdict == DENY else 0, "agent"))
            return Q
        if kind == "REMOVE":
            # undo the agent's own changes on w: lift a rate limit first, else drop the latest agent rule
            if a.p >= topo.V and P.endpoint[w].rate_limit is not None:
                ep = Q.endpoint[w]
                Q.endpoint[w] = EndpointPolicy(ep.syscalls, ep.caps, None)
                return Q
            cands = [r for r in Q.rules if r.origin in ("agent", "repair") and w in r.subject
                     and (a.p >= topo.V or a.p in r.peer)]
            if cands:
                Q.remove_rule(cands[-1].rid)
            return Q
        if kind == "ISOLATE":
            Q.rules = [r for r in Q.rules if w not in r.subject]
            Q._reach = None
            others = frozenset(i for i in range(topo.V) if i != w)
            Q.add_rule(Rule(IN, frozenset({w}), others, topo.ps.ALL, DENY, 20, "agent"))
            Q.add_rule(Rule(OUT, frozenset({w}), others, topo.ps.ALL, DENY, 20, "agent"))
            return Q
        ep = Q.endpoint[w]
        if kind == "TIGHTEN_SC":
            Q.endpoint[w] = EndpointPolicy(ep.syscalls - SYSCALL_GROUPS["danger"], ep.caps, ep.rate_limit)
        elif kind == "RELAX_SC":
            Q.endpoint[w] = EndpointPolicy(ep.syscalls | SYSCALL_GROUPS["danger"], ep.caps, ep.rate_limit)
        elif kind == "DROP_CAP":
            Q.endpoint[w] = EndpointPolicy(ep.syscalls, ep.caps - DROPPABLE_CAPS, ep.rate_limit)
        elif kind == "RATE_LIMIT":
            # install a per-source limit (q connections/s); idempotent, lifted via REMOVE(w, *)
            Q.endpoint[w] = EndpointPolicy(ep.syscalls, ep.caps, 50.0 if ep.rate_limit is None else ep.rate_limit)
        return Q
