"""Attack scenarios S1-S8 mapped to MITRE ATT&CK for Containers.

Each scenario is a small state machine driven by the simulator.  At every decision period it
emits labelled events:
  * network flows   (src, dst, port_mask, count, label="mal")
  * syscall events  (workload, syscall name)
  * capability uses (workload, cap)
and exposes `goal_reached(P, ...)` – whether the attacker achieved its objective under the
current policy.  Stage 1 (recon / staging) runs for `stage1_steps` periods before the goal is
attempted, which makes time-to-mitigate meaningful: a defender that reacts within stage 1
prevents the attack altogether."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from safepol.model.policy_model import Policy, Topology

ATTACK_IDS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
ATTACK_INFO = {
    "S1": ("Container escape attempt", "T1611"),
    "S2": ("Lateral movement / scan", "T1046,T1021"),
    "S3": ("Credential theft / API server", "T1552.005,T1528"),
    "S4": ("Cryptomining", "T1496"),
    "S5": ("Application-layer DoS", "T1499"),
    "S6": ("Malicious image / reverse shell", "T1525,T1195"),
    "S7": ("Privilege escalation", "T1068,T1548"),
    "S8": ("Adaptive attacker", "evasion"),
}


@dataclass
class FlowEvent:
    src: int
    dst: int
    port_mask: int
    count: int
    label: str  # legit | mal


@dataclass
class HostEvent:
    workload: int
    syscalls: Set[str] = field(default_factory=set)
    caps: Set[str] = field(default_factory=set)
    label: str = "mal"


class Attack:
    sid: str = "S0"
    stage1_steps: int = 3
    detect_prob: float = 0.9      # probability that kernel telemetry raises an alert per period

    def __init__(self, topo: Topology, victim: int, rng: random.Random, start: int, duration: int):
        self.topo, self.victim, self.rng = topo, victim, rng
        self.start, self.duration = start, duration
        self.ps = topo.ps
        self.first_event_step: Optional[int] = None
        self.first_blocked_step: Optional[int] = None
        self.success_steps = 0
        self.goal_attempts = 0
        self.blocked_now = False

    # helpers ---------------------------------------------------------------------
    def active(self, t: int) -> bool:
        return self.start <= t < self.start + self.duration

    def stage(self, t: int) -> int:
        return 1 if t - self.start < self.stage1_steps else 2

    def _ext(self, kind: str) -> int:
        return self.topo.by_kind(kind)

    def _pm(self, name: str) -> int:
        return self.ps.mask_of_name(name)

    # interface -------------------------------------------------------------------
    def events(self, t: int, P: Policy) -> Tuple[List[FlowEvent], List[HostEvent]]:
        raise NotImplementedError

    def goal_flows(self, t: int) -> List[Tuple[int, int, int]]:
        """Flows that constitute the objective at stage 2 (empty for host-only goals)."""
        return []

    def objective_open(self, P: Policy) -> bool:
        """Would the attacker's objective succeed under P (evaluated with stage-2 semantics)?"""
        gf = self.goal_flows(self.start + self.stage1_steps)
        if not gf:
            return False
        return all(P.allows(s, d, m) for (s, d, m) in gf)

    def goal_reached(self, t: int, P: Policy) -> bool:
        return self.stage(t) == 2 and self.objective_open(P)

    def observe_outcome(self, t: int, P: Policy, dropped_mal: int, total_mal: int) -> None:
        """TTM bookkeeping: mitigated = objective closed under the enforced policy."""
        if self.first_event_step is None:
            self.first_event_step = t
        blocked = not self.objective_open(P)
        if blocked and self.first_blocked_step is None:
            self.first_blocked_step = t
        self.blocked_now = blocked

    def feature_signature(self) -> Dict[str, float]:
        """Observable per-workload telemetry deltas on the victim (kernel-side sources)."""
        return {"cpu": 0.1, "conns": 0.2, "new_peers": 0.2, "ext_egress": 0.0, "danger_sc": 0.0}


# ----------------------------------------------------------------------------------
class S1Escape(Attack):
    sid, stage1_steps, detect_prob = "S1", 2, 0.9
    RECON = {"readlink", "getdents64", "stat"}
    EXPLOIT = {"unshare", "mount", "setns", "keyctl"}

    def events(self, t, P):
        sc = self.RECON if self.stage(t) == 1 else self.EXPLOIT
        return [], [HostEvent(self.victim, set(sc), {"CAP_SYS_ADMIN"} if self.stage(t) == 2 else set())]

    def objective_open(self, P):
        return bool(self.EXPLOIT & P.endpoint[self.victim].syscalls)

    def feature_signature(self):
        return {"cpu": 0.05, "conns": 0.0, "new_peers": 0.0, "ext_egress": 0.0, "danger_sc": 0.8}


class S2Lateral(Attack):
    sid, stage1_steps, detect_prob = "S2", 3, 0.75

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        topo = self.topo
        sensitive = [i for i, w in enumerate(topo.workloads) if w.tier == "db" or w.name == "paymentservice"]
        self.targets = [i for i in sensitive if (self.victim, i) not in topo.dep_set and i != self.victim]
        if not self.targets:
            self.targets = [i for i in range(topo.n) if i != self.victim and (self.victim, i) not in topo.dep_set][:2]

    def events(self, t, P):
        fl = []
        if self.stage(t) == 1:  # scan: every workload on its service port + ssh
            for v in range(self.topo.n):
                if v == self.victim:
                    continue
                m = 0
                for pn in self.topo.workloads[v].ports:
                    proto, port = self.ps.named_ports[pn]
                    m |= self.ps.mask_of_port(proto, port)
                m |= self._pm("ssh-22")
                fl.append(FlowEvent(self.victim, v, m, 2, "mal"))
        else:
            for tg in self.targets:
                m = 0
                for pn in self.topo.workloads[tg].ports:
                    proto, port = self.ps.named_ports[pn]
                    m |= self.ps.mask_of_port(proto, port)
                fl.append(FlowEvent(self.victim, tg, m, 10, "mal"))
        return fl, [HostEvent(self.victim, {"socket", "connect"}, set())]

    def goal_flows(self, t):
        out = []
        for tg in self.targets:
            m = 0
            for pn in self.topo.workloads[tg].ports:
                proto, port = self.ps.named_ports[pn]
                m |= self.ps.mask_of_port(proto, port)
            out.append((self.victim, tg, m))
        return out

    def objective_open(self, P):
        return any(P.allows(s, d, m) for (s, d, m) in self.goal_flows(0))

    def feature_signature(self):
        return {"cpu": 0.05, "conns": 0.3, "new_peers": 0.9, "ext_egress": 0.0, "danger_sc": 0.0}


class S3Credential(Attack):
    sid, stage1_steps, detect_prob = "S3", 2, 0.8

    def events(self, t, P):
        meta, api = self._ext("MetadataService"), self._ext("APIServer")
        if self.stage(t) == 1:
            fl = [FlowEvent(self.victim, meta, self._pm("meta-80"), 5, "mal")]
        else:
            fl = [FlowEvent(self.victim, api, self._pm("https-443"), 8, "mal")]
        return fl, [HostEvent(self.victim, {"openat", "read"}, set())]

    def goal_flows(self, t):
        return [(self.victim, self._ext("APIServer"), self._pm("https-443"))]

    def feature_signature(self):
        return {"cpu": 0.02, "conns": 0.1, "new_peers": 0.3, "ext_egress": 0.5, "danger_sc": 0.0}


class S4Mining(Attack):
    sid, stage1_steps, detect_prob = "S4", 2, 0.85

    def events(self, t, P):
        inet = self._ext("Internet")
        if self.stage(t) == 1:
            fl = [FlowEvent(self.victim, inet, self._pm("https-443"), 3, "mal"),   # download miner
                  FlowEvent(self.victim, inet, self._pm("dns-53"), 2, "mal")]
        else:
            fl = [FlowEvent(self.victim, inet, self._pm("mining-3333"), 20, "mal")]
        return fl, [HostEvent(self.victim, {"execve", "clone", "prctl"}, set())]

    def goal_flows(self, t):
        return [(self.victim, self._ext("Internet"), self._pm("mining-3333"))]

    def feature_signature(self):
        return {"cpu": 0.9, "conns": 0.1, "new_peers": 0.1, "ext_egress": 0.8, "danger_sc": 0.0}


class S5DoS(Attack):
    """Internet -> frontend flood.  Victim is the frontend; the goal is overload (SLO violation).
    Cannot be solved by deny rules (Internet->frontend must stay reachable, phi8); RATE_LIMIT is
    the expected response."""
    sid, stage1_steps, detect_prob = "S5", 2, 0.9
    FLOOD = 1500

    def __init__(self, topo, victim, rng, start, duration):
        fe = topo.workload_idx(lambda w: w.tier == "ingress")[0]
        super().__init__(topo, fe, rng, start, duration)

    def events(self, t, P):
        inet = self._ext("Internet")
        n = 200 if self.stage(t) == 1 else self.FLOOD
        return [FlowEvent(inet, self.victim, self._pm("http-8080"), n, "mal")], []

    def objective_open(self, P):
        q = P.endpoint[self.victim].rate_limit
        return q is None or q * 5 > 400   # unlimited (or very loose) => overload

    def feature_signature(self):
        return {"cpu": 0.8, "conns": 1.0, "new_peers": 0.2, "ext_egress": 0.0, "danger_sc": 0.0}


class S6MaliciousImage(Attack):
    sid, stage1_steps, detect_prob = "S6", 1, 0.7

    def events(self, t, P):
        inet = self._ext("Internet")
        fl = [FlowEvent(self.victim, inet, self._pm("dns-53"), 1, "mal")]
        if self.stage(t) == 2:
            fl.append(FlowEvent(self.victim, inet, self._pm("revshell-4444"), 4, "mal"))
        return fl, [HostEvent(self.victim, {"execve", "dup", "socket", "connect"}, set())]

    def goal_flows(self, t):
        return [(self.victim, self._ext("Internet"), self._pm("revshell-4444"))]

    def feature_signature(self):
        return {"cpu": 0.05, "conns": 0.05, "new_peers": 0.1, "ext_egress": 0.4, "danger_sc": 0.1}


class S7PrivEsc(Attack):
    sid, stage1_steps, detect_prob = "S7", 2, 0.85
    SC = {"ptrace", "process_vm_readv", "personality"}

    def events(self, t, P):
        caps = {"CAP_NET_RAW"} if self.stage(t) == 1 else {"CAP_NET_RAW", "CAP_DAC_OVERRIDE"}
        sc = {"socket", "sendto"} if self.stage(t) == 1 else set(self.SC)
        return [], [HostEvent(self.victim, sc, caps)]

    def objective_open(self, P):
        ep = P.endpoint[self.victim]
        return bool(self.SC & ep.syscalls) and "CAP_NET_RAW" in ep.caps

    def feature_signature(self):
        return {"cpu": 0.05, "conns": 0.05, "new_peers": 0.0, "ext_egress": 0.0, "danger_sc": 0.7}


class S8Adaptive(Attack):
    """Port-hopping, low-rate exfiltration/mining with spoofed pod-level metrics.
    Switches Internet port whenever the current one is dropped; reduces per-period volume to
    stay below naive thresholds; kernel telemetry still shows connect() to new peers."""
    sid, stage1_steps, detect_prob = "S8", 2, 0.55
    PORTS = ["mining-3333", "https-443", "mining-14444", "ssh-22"]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pi = 0
        self.hops = 0

    def events(self, t, P):
        inet = self._ext("Internet")
        for _ in range(len(self.PORTS)):   # hop to the first port that is still open
            if P.allows(self.victim, inet, self._pm(self.PORTS[self.pi])):
                break
            self.pi = (self.pi + 1) % len(self.PORTS)
            self.hops += 1
        n = 1 if self.stage(t) == 1 else 3
        return [FlowEvent(self.victim, inet, self._pm(self.PORTS[self.pi]), n, "mal")], \
               [HostEvent(self.victim, {"connect", "socket"}, set())]

    def goal_flows(self, t):
        return [(self.victim, self._ext("Internet"), self._pm(self.PORTS[self.pi]))]

    def objective_open(self, P):
        # the adaptive attacker hops ports: it is only contained if *every* fallback port is closed
        inet = self._ext("Internet")
        return any(P.allows(self.victim, inet, self._pm(p)) for p in self.PORTS)

    def feature_signature(self):
        # spoofed: cpu/conns look normal; only kernel-side ext_egress/new_peers leak a weak signal
        return {"cpu": 0.0, "conns": 0.02, "new_peers": 0.15, "ext_egress": 0.2, "danger_sc": 0.0}


SCENARIOS = {"S1": S1Escape, "S2": S2Lateral, "S3": S3Credential, "S4": S4Mining,
             "S5": S5DoS, "S6": S6MaliciousImage, "S7": S7PrivEsc, "S8": S8Adaptive}


def make_attack(sid: str, topo: Topology, rng: random.Random, start: int, duration: int,
                victim: Optional[int] = None) -> Attack:
    cls = SCENARIOS[sid]
    if victim is None:
        cands = [i for i, w in enumerate(topo.workloads) if w.tier in ("app", "ingress") and w.name != "loadgenerator"]
        victim = rng.choice(cands)
    return cls(topo, victim, rng, start, duration)
