"""Policy model: workloads, endpoints, rules, policy state and reachability semantics.

Semantics follow Cilium's CiliumNetworkPolicy (CNP):
  * deny rules take precedence over allow rules regardless of order;
  * an endpoint that is *selected* by at least one policy in a direction falls into
    default-deny for that direction (only explicitly allowed traffic passes);
  * an endpoint that is not selected by any policy in a direction allows everything;
  * external endpoints (CIDRs) carry no policy and always "allow" on their side.

Ports are abstracted into a fixed, finite set of *port classes* (an elementary
partition of {TCP,UDP} x [0,65535] induced by the benchmark's port universe).  Rule
port sets and flow ports therefore become bit masks, which keeps reachability a
vectorised bit operation and keeps the SMT/MaxSAT encodings exact over the
benchmark's packet space.  The conflict analyser (verify/conflicts.py) additionally
cross-checks the class abstraction with a 16-bit bit-vector encoding of raw ports.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

TCP, UDP = 0, 1
PROTO_NAME = {TCP: "TCP", UDP: "UDP"}
ALLOW, DENY = "allow", "deny"
IN, OUT = "in", "out"

# Linux capabilities (subset used by the benchmark) and syscall vocabulary.
CAPABILITIES = [
    "CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_FSETID", "CAP_KILL", "CAP_SETGID",
    "CAP_SETUID", "CAP_SETPCAP", "CAP_NET_BIND_SERVICE", "CAP_NET_RAW", "CAP_SYS_CHROOT",
    "CAP_MKNOD", "CAP_AUDIT_WRITE", "CAP_SETFCAP", "CAP_SYS_ADMIN", "CAP_SYS_PTRACE",
    "CAP_NET_ADMIN", "CAP_SYS_MODULE", "CAP_DAC_READ_SEARCH", "CAP_SYS_RAWIO",
]
DOCKER_DEFAULT_CAPS = frozenset(CAPABILITIES[:14])
DANGEROUS_CAPS = frozenset({"CAP_SYS_ADMIN", "CAP_SYS_PTRACE", "CAP_SYS_MODULE", "CAP_SYS_RAWIO",
                            "CAP_DAC_READ_SEARCH", "CAP_NET_ADMIN"})

# Syscall groups (seccomp templates); names follow x86-64.
SYSCALL_GROUPS: Dict[str, FrozenSet[str]] = {
    "base": frozenset({"read", "write", "openat", "close", "fstat", "mmap", "mprotect", "munmap",
                       "brk", "rt_sigaction", "rt_sigprocmask", "ioctl", "pread64", "pwrite64",
                       "futex", "nanosleep", "getpid", "exit_group", "epoll_wait", "epoll_ctl",
                       "clock_gettime", "getrandom", "lseek", "fcntl", "dup", "pipe2"}),
    "net": frozenset({"socket", "connect", "accept4", "bind", "listen", "sendto", "recvfrom",
                      "sendmsg", "recvmsg", "getsockopt", "setsockopt", "shutdown", "getpeername"}),
    "proc": frozenset({"clone", "clone3", "execve", "wait4", "kill", "prctl", "setuid", "setgid"}),
    "fs": frozenset({"mkdir", "unlink", "rename", "chmod", "chown", "stat", "readlink", "getdents64"}),
    "danger": frozenset({"ptrace", "mount", "umount2", "unshare", "setns", "keyctl", "bpf",
                         "init_module", "finit_module", "reboot", "kexec_load", "pivot_root",
                         "personality", "process_vm_readv", "add_key"}),
}
SYSCALL_UNIVERSE: FrozenSet[str] = frozenset(itertools.chain.from_iterable(SYSCALL_GROUPS.values()))
DANGEROUS_SYSCALLS = SYSCALL_GROUPS["danger"]


# --------------------------------------------------------------------------------------
# Port classes
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PortClass:
    proto: int
    lo: int
    hi: int
    name: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{PROTO_NAME[self.proto]}:{self.lo}-{self.hi}({self.name})"


class PortSpace:
    """Elementary partition of the port universe.  Each class gets a bit; sets are ints."""

    def __init__(self, named_ports: Dict[str, Tuple[int, int]]):
        # named_ports: name -> (proto, port). Build partition of [0,65535] per proto.
        cuts: Dict[int, Set[int]] = {TCP: {0, 65536}, UDP: {0, 65536}}
        for _, (proto, port) in named_ports.items():
            cuts[proto].update({port, port + 1})
        # ephemeral range boundary
        cuts[TCP].update({32768}); cuts[UDP].update({32768})
        self.classes: List[PortClass] = []
        for proto in (TCP, UDP):
            pts = sorted(cuts[proto])
            for lo, hi in zip(pts[:-1], pts[1:]):
                name = next((n for n, (p, pt) in named_ports.items() if p == proto and pt == lo and hi == lo + 1), None)
                if name is None:
                    name = "ephemeral" if lo >= 32768 else "other"
                self.classes.append(PortClass(proto, lo, hi - 1, name))
        if len(self.classes) > 63:
            raise ValueError("too many port classes for a 64-bit mask")
        self.K = len(self.classes)
        self.ALL = (1 << self.K) - 1
        self._by_name: Dict[str, int] = {}
        for i, c in enumerate(self.classes):
            if c.name not in ("ephemeral", "other"):
                self._by_name[c.name] = 1 << i
        self.named_ports = dict(named_ports)

    def mask_of_port(self, proto: int, port: int) -> int:
        for i, c in enumerate(self.classes):
            if c.proto == proto and c.lo <= port <= c.hi:
                return 1 << i
        raise ValueError(port)

    def mask_of_range(self, proto: int, lo: int, hi: int) -> int:
        m = 0
        for i, c in enumerate(self.classes):
            if c.proto == proto and c.hi >= lo and c.lo <= hi:
                m |= 1 << i
        return m

    def mask_of_name(self, name: str) -> int:
        return self._by_name[name]

    def mask_all_proto(self, proto: int) -> int:
        return self.mask_of_range(proto, 0, 65535)

    def describe(self, mask: int) -> str:
        if mask == self.ALL:
            return "*"
        return ",".join(str(c) for i, c in enumerate(self.classes) if mask >> i & 1)

    def intervals(self, mask: int) -> List[Tuple[int, int, int]]:
        """Return (proto, lo, hi) intervals covered by the mask (merged)."""
        out: List[Tuple[int, int, int]] = []
        for i, c in enumerate(self.classes):
            if mask >> i & 1:
                if out and out[-1][0] == c.proto and out[-1][2] + 1 == c.lo:
                    out[-1] = (c.proto, out[-1][1], c.hi)
                else:
                    out.append((c.proto, c.lo, c.hi))
        return out


# --------------------------------------------------------------------------------------
# Endpoints and workloads
# --------------------------------------------------------------------------------------
@dataclass
class Workload:
    name: str
    namespace: str = "default"
    labels: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # served ports
    tier: str = "app"            # app | db | sys | ingress
    trusted: bool = True
    image_vuln_score: float = 0.0


@dataclass
class ExternalEndpoint:
    name: str
    cidr: str
    kind: str = "external"  # Internet | NodeCIDR | MetadataService | APIServer


class Topology:
    """Endpoint index space V = W ∪ E with a service dependency graph G_dep."""

    def __init__(self, workloads: Sequence[Workload], externals: Sequence[ExternalEndpoint],
                 dep_edges: Iterable[Tuple[str, str, str]], port_space: PortSpace):
        self.workloads = list(workloads)
        self.externals = list(externals)
        self.n = len(self.workloads)
        self.V = self.n + len(self.externals)
        self.names = [w.name for w in self.workloads] + [e.name for e in self.externals]
        self.index = {nm: i for i, nm in enumerate(self.names)}
        self.ps = port_space
        # dependency edges: (src, dst, port_name)
        self.dep_edges: List[Tuple[int, int, int]] = []
        for s, d, pn in dep_edges:
            proto, port = port_space.named_ports[pn]
            self.dep_edges.append((self.index[s], self.index[d], port_space.mask_of_port(proto, port)))
        self.dep_set = {(s, d) for s, d, _ in self.dep_edges}
        self.ext_kind = {self.index[e.name]: e.kind for e in self.externals}

    def is_workload(self, i: int) -> bool:
        return i < self.n

    def by_kind(self, kind: str) -> int:
        for i, k in self.ext_kind.items():
            if k == kind:
                return i
        raise KeyError(kind)

    def workload_idx(self, pred) -> List[int]:
        return [i for i, w in enumerate(self.workloads) if pred(w)]

    def clone_with_new_workload(self, w: Workload, edges: Iterable[Tuple[str, str, str]]) -> "Topology":
        wl = self.workloads + [w]
        old_edges = [(self.names[s], self.names[d], self._port_name(m)) for s, d, m in self.dep_edges]
        return Topology(wl, self.externals, old_edges + list(edges), self.ps)

    def _port_name(self, mask: int) -> str:
        for i, c in enumerate(self.ps.classes):
            if mask >> i & 1:
                return c.name
        raise KeyError(mask)


# --------------------------------------------------------------------------------------
# Rules and policy state
# --------------------------------------------------------------------------------------
_rule_counter = itertools.count(1)


@dataclass
class Rule:
    """Network rule rho = (dir, subject, peer, ports, verdict, priority)."""
    direction: str                 # IN | OUT
    subject: FrozenSet[int]        # workload indices selected by sigma_s
    peer: FrozenSet[int]           # endpoint indices selected by sigma_p (may include externals)
    ports: int                     # port-class bit mask
    verdict: str                   # ALLOW | DENY
    priority: int = 0              # higher matched first (within same verdict class)
    origin: str = "init"           # init | agent | repair | baseline
    rid: int = field(default_factory=lambda: next(_rule_counter))

    def key(self) -> Tuple:
        return (self.direction, self.subject, self.peer, self.ports, self.verdict)

    def copy(self) -> "Rule":
        return Rule(self.direction, self.subject, self.peer, self.ports, self.verdict,
                    self.priority, self.origin, self.rid)

    def size(self) -> int:
        return len(self.subject) * len(self.peer)


@dataclass
class EndpointPolicy:
    syscalls: FrozenSet[str]
    caps: FrozenSet[str]
    rate_limit: Optional[float] = None   # connections / s, None = unlimited


class Policy:
    """Policy state P = (R, {Sigma_w}, {C_w}, {q_w})."""

    def __init__(self, topo: Topology, rules: Optional[List[Rule]] = None,
                 endpoint: Optional[Dict[int, EndpointPolicy]] = None):
        self.topo = topo
        self.rules: List[Rule] = list(rules or [])
        self.endpoint: Dict[int, EndpointPolicy] = endpoint or {
            i: EndpointPolicy(SYSCALL_UNIVERSE, DOCKER_DEFAULT_CAPS) for i in range(topo.n)}
        self.version = 0
        self._reach: Optional[np.ndarray] = None  # cached allowed-port-mask matrix
        self._mats: Optional[Dict[str, np.ndarray]] = None  # cached allow/deny matrices for incremental updates
        self._cow = False

    # ---- copy / diff ---------------------------------------------------------------
    def copy(self) -> "Policy":
        p = Policy(self.topo, [r.copy() for r in self.rules],
                   {k: EndpointPolicy(v.syscalls, v.caps, v.rate_limit) for k, v in self.endpoint.items()})
        p.version = self.version
        # caches are shared copy-on-write: whichever side mutates first takes a private copy
        p._reach, p._mats = self._reach, self._mats
        p._cow = self._cow = self._reach is not None
        return p

    def rule_ids(self) -> Set[int]:
        return {r.rid for r in self.rules}

    def edit_distance(self, other: "Policy") -> int:
        """d(P,P') = #rules added + #rules removed (+ syscall/cap symmetric differences)."""
        a = {r.key(): r for r in self.rules}
        b = {r.key(): r for r in other.rules}
        d = len(set(a) ^ set(b))
        for w in range(self.topo.n):
            if w in self.endpoint and w in other.endpoint:
                d += len(self.endpoint[w].syscalls ^ other.endpoint[w].syscalls) > 0
                d += len(self.endpoint[w].caps ^ other.endpoint[w].caps) > 0
                d += (self.endpoint[w].rate_limit != other.endpoint[w].rate_limit)
        return d

    # ---- mutation ------------------------------------------------------------------
    def add_rule(self, rule: Rule) -> None:
        if any(r.key() == rule.key() for r in self.rules):
            return
        self.rules.append(rule)
        self._invalidate(rule)

    def remove_rule(self, rid: int) -> Optional[Rule]:
        for i, r in enumerate(self.rules):
            if r.rid == rid:
                self.rules.pop(i)
                self._invalidate(r)
                return r
        return None

    def _invalidate(self, rule: Optional[Rule] = None) -> None:
        """Incremental cache maintenance (Sec. 2.2): a rule with subject S and direction OUT can only
        change rows S of M_P (its egress side); direction IN only columns S.  Those rows/columns are
        rebuilt from the per-subject rule index and the cached ingress/egress matrices, so the cost
        is O(|S| * (V + |R_S|)) instead of O(V^2 * |R|) for a full recomputation."""
        if self._reach is None or rule is None or self._mats is None:
            self._reach = None; self._mats = None; self._cow = False
            return
        if self._cow:
            self._reach = self._reach.copy()
            self._mats = {k: v.copy() for k, v in self._mats.items()}
            self._cow = False
        m = self._mats
        subj = np.fromiter(rule.subject, dtype=np.int64)
        if rule.direction == OUT:
            m["eg_allow"][subj, :] = 0; m["eg_deny"][subj, :] = 0; m["sel_out"][subj] = False
            for u in rule.subject:
                for r in self.rules:
                    if r.direction == OUT and u in r.subject:
                        m["sel_out"][u] = True
                        p = np.fromiter(r.peer, dtype=np.int64)
                        (m["eg_allow"] if r.verdict == ALLOW else m["eg_deny"])[u, p] |= r.ports
            eg = np.where(m["sel_out"][subj, None], m["eg_allow"][subj] & ~m["eg_deny"][subj], self.topo.ps.ALL)
            ing = np.where(m["sel_in"][None, :], m["in_allow"][subj] & ~m["in_deny"][subj], self.topo.ps.ALL)
            self._reach[subj, :] = eg & ing
        else:
            m["in_allow"][:, subj] = 0; m["in_deny"][:, subj] = 0; m["sel_in"][subj] = False
            for v in rule.subject:
                for r in self.rules:
                    if r.direction == IN and v in r.subject:
                        m["sel_in"][v] = True
                        p = np.fromiter(r.peer, dtype=np.int64)
                        (m["in_allow"] if r.verdict == ALLOW else m["in_deny"])[p, v] |= r.ports
            ing = np.where(m["sel_in"][None, subj], m["in_allow"][:, subj] & ~m["in_deny"][:, subj], self.topo.ps.ALL)
            eg = np.where(m["sel_out"][:, None], m["eg_allow"][:, subj] & ~m["eg_deny"][:, subj], self.topo.ps.ALL)
            self._reach[:, subj] = eg & ing
        n = self.topo.n
        for u in rule.subject:
            self._reach[u, u] = 0
        self._reach[n:, n:] = 0

    # ---- reachability --------------------------------------------------------------
    def _selected(self, w: int, direction: str) -> bool:
        return any(r.direction == direction and w in r.subject for r in self.rules)

    def egress_ports(self, u: int, v: int) -> int:
        """Port-class mask allowed by u's egress side towards v."""
        if not self.topo.is_workload(u):
            return self.topo.ps.ALL
        selected, allow, deny = False, 0, 0
        for r in self.rules:
            if r.direction != OUT or u not in r.subject:
                continue
            selected = True
            if v in r.peer:
                if r.verdict == ALLOW:
                    allow |= r.ports
                else:
                    deny |= r.ports
        if not selected:
            return self.topo.ps.ALL
        return allow & ~deny

    def ingress_ports(self, u: int, v: int) -> int:
        if not self.topo.is_workload(v):
            return self.topo.ps.ALL
        selected, allow, deny = False, 0, 0
        for r in self.rules:
            if r.direction != IN or v not in r.subject:
                continue
            selected = True
            if u in r.peer:
                if r.verdict == ALLOW:
                    allow |= r.ports
                else:
                    deny |= r.ports
        if not selected:
            return self.topo.ps.ALL
        return allow & ~deny

    def verdict_mask(self, u: int, v: int) -> int:
        if u == v:
            return 0
        return self.egress_ports(u, v) & self.ingress_ports(u, v)

    def allows(self, u: int, v: int, port_mask: int) -> bool:
        return bool(self.reach()[u, v] & port_mask)

    def _row_out(self, u: int) -> np.ndarray:
        V = self.topo.V
        row = np.zeros(V, dtype=np.int64)
        for v in range(V):
            row[v] = self.verdict_mask(u, v)
        return row

    def _col_in(self, v: int) -> np.ndarray:
        V = self.topo.V
        col = np.zeros(V, dtype=np.int64)
        for u in range(V):
            col[u] = self.verdict_mask(u, v)
        return col

    def reach(self) -> np.ndarray:
        """Matrix of allowed port-class masks; M_P[u,v] = (reach()[u,v] != 0)."""
        if self._reach is None:
            self._reach = self.full_reach(keep=True)
        return self._reach

    def full_reach(self, keep: bool = False) -> np.ndarray:
        """Vectorised full recomputation (used for scale experiments and as oracle).
        With keep=True the intermediate allow/deny matrices are cached for incremental updates."""
        topo, V, n = self.topo, self.topo.V, self.topo.n
        ALL = topo.ps.ALL
        eg_allow = np.zeros((V, V), dtype=np.int64)
        eg_deny = np.zeros((V, V), dtype=np.int64)
        in_allow = np.zeros((V, V), dtype=np.int64)
        in_deny = np.zeros((V, V), dtype=np.int64)
        sel_out = np.zeros(V, dtype=bool)
        sel_in = np.zeros(V, dtype=bool)
        for r in self.rules:
            s = np.fromiter(r.subject, dtype=np.int64)
            p = np.fromiter(r.peer, dtype=np.int64)
            if r.direction == OUT:
                sel_out[s] = True
                tgt = eg_allow if r.verdict == ALLOW else eg_deny
                tgt[np.ix_(s, p)] |= r.ports
            else:
                sel_in[s] = True
                tgt = in_allow if r.verdict == ALLOW else in_deny
                tgt[np.ix_(p, s)] |= r.ports
        eg = np.where(sel_out[:, None], eg_allow & ~eg_deny, ALL)
        ing = np.where(sel_in[None, :], in_allow & ~in_deny, ALL)
        M = eg & ing
        np.fill_diagonal(M, 0)
        # external->external pairs are not part of the cluster
        M[n:, n:] = 0
        self._mats = dict(eg_allow=eg_allow, eg_deny=eg_deny, in_allow=in_allow, in_deny=in_deny,
                          sel_out=sel_out, sel_in=sel_in) if keep else None
        return M

    def bool_reach(self) -> np.ndarray:
        return self.reach() != 0

    def attack_surface(self, weights: Optional[np.ndarray] = None) -> int:
        M = self.bool_reach()
        if weights is None:
            return int(M.sum())
        return int((M * weights).sum())

    def blast_radius(self) -> np.ndarray:
        """BR(u) = #workloads reachable from u via transitive closure over workloads."""
        n = self.topo.n
        M = self.bool_reach()[:n, :n].copy()
        R = M.copy()
        # Floyd–Warshall style closure (boolean), n small; for large n use repeated squaring
        for k in range(n):
            R |= R[:, k:k + 1] & R[k:k + 1, :]
        np.fill_diagonal(R, False)
        return R.sum(axis=1)

    # ---- embedding -------------------------------------------------------------------
    def embedding(self) -> np.ndarray:
        """h(P): [#rules, #allow, #deny, #in, #out, AS/AS_max, mean BR/n, per-w in/out degree]."""
        n, V = self.topo.n, self.topo.V
        M = self.bool_reach()
        AS = M.sum() / max(1, n * (V - 1))
        br = self.blast_radius() / max(1, n - 1)
        in_deg = M[:, :n].sum(axis=0) / max(1, V - 1)
        out_deg = M[:n, :].sum(axis=1) / max(1, V - 1)
        allow = sum(r.verdict == ALLOW for r in self.rules)
        deny = len(self.rules) - allow
        ins = sum(r.direction == IN for r in self.rules)
        glob = np.array([len(self.rules) / 50.0, allow / 50.0, deny / 50.0, ins / 50.0,
                         (len(self.rules) - ins) / 50.0, AS, br.mean()], dtype=np.float32)
        return np.concatenate([glob, in_deg.astype(np.float32), out_deg.astype(np.float32),
                               br.astype(np.float32)])

    def describe(self) -> str:
        t = self.topo
        lines = []
        for r in sorted(self.rules, key=lambda r: (-r.priority, r.rid)):
            subj = ",".join(t.names[i] for i in sorted(r.subject))
            peer = "*" if len(r.peer) == t.V else ",".join(t.names[i] for i in sorted(r.peer))
            lines.append(f"[{r.rid:3d} p{r.priority}] {r.verdict:5s} {r.direction:3s} {subj} <-> {peer} ports={t.ps.describe(r.ports)} ({r.origin})")
        return "\n".join(lines)
