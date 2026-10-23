"""Translation between the SAFEPOL policy model and Kubernetes/Cilium objects.

Policy  -> CiliumNetworkPolicy (one per workload)  +  pod securityContext (seccomp/capabilities)
NetworkPolicy (networking.k8s.io/v1, e.g. the upstream Online Boutique set) -> Policy

Mapping of model endpoints to Cilium peer selectors:
  workload w                      endpointSelector {app: <name>}            (same namespace)
  Internet        (0.0.0.0/0)     entities [world]
  NodeCIDR                        entities [host, remote-node]
  MetadataService 169.254.169.254 CIDR 169.254.169.254/32
  APIServer                       entities [kube-apiserver]
Port-class masks become port ranges (Cilium >= 1.16 supports endPort); the full mask omits toPorts.
Model semantics (deny precedence, selected => default-deny) are exactly Cilium's, so a rule set is
translated rule by rule.  Rate limits are not translated (Cilium has no per-connection rate limit
primitive; a deployment would use an L7 proxy), which the paper states explicitly.
"""
from __future__ import annotations

import json
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from safepol.model.policy_model import (ALLOW, DENY, IN, OUT, TCP, UDP, PROTO_NAME, CAPABILITIES,
                                        DOCKER_DEFAULT_CAPS, SYSCALL_UNIVERSE, EndpointPolicy, Policy,
                                        Rule, Topology)

ENTITY_OF_KIND = {"Internet": ["world"], "NodeCIDR": ["host", "remote-node"], "APIServer": ["kube-apiserver"]}
CIDR_OF_KIND = {"MetadataService": "169.254.169.254/32"}


# ---------------------------------------------------------------------------------------
# Policy -> CiliumNetworkPolicy
# ---------------------------------------------------------------------------------------
def _ports(topo: Topology, mask: int) -> Optional[List[dict]]:
    if mask == topo.ps.ALL:
        return None
    out = []
    for proto, lo, hi in topo.ps.intervals(mask):
        lo = max(lo, 1)
        if hi < lo:
            continue
        p = {"port": str(lo), "protocol": PROTO_NAME[proto]}
        if hi > lo:
            p["endPort"] = hi
        out.append(p)
    return [{"ports": out}] if out else None


def _peer_blocks(topo: Topology, peers: FrozenSet[int], direction: str, subject: int,
                 skip_kinds: FrozenSet[str] = frozenset()) -> List[dict]:
    """Split a peer set into Cilium peer-selector blocks (endpoints / entities / CIDR)."""
    key_ep = "fromEndpoints" if direction == IN else "toEndpoints"
    key_ent = "fromEntities" if direction == IN else "toEntities"
    key_cidr = "fromCIDR" if direction == IN else "toCIDR"
    others = set(range(topo.V)) - {subject}
    if peers >= others:
        return [{key_ent: ["all"]}]
    blocks: List[dict] = []
    wl = [i for i in sorted(peers) if topo.is_workload(i)]
    if wl:
        blocks.append({key_ep: [{"matchLabels": {"app": topo.names[i]}} for i in wl]})
    ents: List[str] = []
    cidrs: List[str] = []
    for i in sorted(peers):
        if topo.is_workload(i):
            continue
        kind = topo.ext_kind[i]
        if kind in skip_kinds:
            continue
        if kind in ENTITY_OF_KIND:
            ents += ENTITY_OF_KIND[kind]
        elif kind in CIDR_OF_KIND:
            cidrs.append(CIDR_OF_KIND[kind])
    if ents:
        blocks.append({key_ent: ents})
    if cidrs:
        blocks.append({key_cidr: cidrs})
    return blocks


def policy_to_cnps(P: Policy, namespace: str = "boutique", prefix: str = "safepol", dns_carveout: bool = True,
                   apiserver_is_host: bool = False, notes: Optional[list] = None) -> List[dict]:
    """``apiserver_is_host``: on clusters whose node also runs the API server, Cilium gives the host
    identity the ``reserved:kube-apiserver`` label as well, so a deny on the kube-apiserver entity
    would also deny the kubelet (health probes).  The model keeps NodeCIDR and APIServer apart, so a
    deny rule that names APIServer but not NodeCIDR cannot be realised faithfully; it is skipped and
    reported through ``notes`` instead of silently over-denying."""
    topo = P.topo
    cnps = []
    api_kinds = {i for i, k in topo.ext_kind.items() if k == "APIServer"}
    node_kinds = {i for i, k in topo.ext_kind.items() if k == "NodeCIDR"}
    for w in range(topo.n):
        spec: Dict[str, list] = {}
        for direction, allow_key, deny_key in ((IN, "ingress", "ingressDeny"), (OUT, "egress", "egressDeny")):
            rules = [r for r in P.rules if r.direction == direction and w in r.subject]
            if not rules:
                continue
            allows, denies = [], []
            for r in rules:
                if not r.peer:
                    continue  # empty-peer allow = selection only; handled below
                skip: FrozenSet[str] = frozenset()
                if r.verdict == DENY and apiserver_is_host and (r.peer & api_kinds) and not (r.peer & node_kinds):
                    skip = frozenset({"APIServer"})
                    if notes is not None:
                        notes.append(f"{topo.names[w]} {direction}: deny of APIServer not realisable (identity overlaps host)")
                for blk in _peer_blocks(topo, r.peer, direction, w, skip):
                    entry = dict(blk)
                    tp = _ports(topo, r.ports)
                    if tp:
                        entry["toPorts"] = tp
                    (allows if r.verdict == ALLOW else denies).append(entry)
            if not allows:
                # selected in this direction with nothing allowed: explicit deny-all, which is what the
                # model's "selected => default-deny" means and what Cilium would do with an empty section.
                key_ent = "fromEntities" if direction == IN else "toEntities"
                denies.append({key_ent: ["all"]})
            if allows and direction == OUT and dns_carveout:
                # cluster DNS is infrastructure outside the modelled endpoint space; a selected egress
                # section without it would break name resolution for the legitimate flows Phi protects
                allows.append({"toEndpoints": [{"matchLabels": {"k8s:io.kubernetes.pod.namespace": "kube-system", "k8s:k8s-app": "kube-dns"}}],
                               "toPorts": [{"ports": [{"port": "53", "protocol": "ANY"}]}]})
            if allows:
                spec[allow_key] = allows
            if denies:
                spec[deny_key] = denies
        if not spec:
            continue
        cnps.append({
            "apiVersion": "cilium.io/v2", "kind": "CiliumNetworkPolicy",
            "metadata": {"name": f"{prefix}-{topo.names[w]}", "namespace": namespace,
                         "labels": {"safepol.io/managed": "true"}},
            "spec": {"endpointSelector": {"matchLabels": {"app": topo.names[w]}}, **spec},
        })
    return cnps


# ---------------------------------------------------------------------------------------
# EndpointPolicy -> seccomp profile + capabilities
# ---------------------------------------------------------------------------------------
def seccomp_profile(ep: EndpointPolicy) -> dict:
    """Default-allow profile that denies exactly the syscalls removed from the model's universe,
    i.e. the profile is the complement of the model's allowed set over the benchmark vocabulary
    (an allow-list over the full x86-64 table would need the image's real syscall footprint)."""
    denied = sorted(SYSCALL_UNIVERSE - ep.syscalls)
    prof = {"defaultAction": "SCMP_ACT_ALLOW", "architectures": ["SCMP_ARCH_X86_64"], "syscalls": []}
    if denied:
        prof["syscalls"].append({"names": denied, "action": "SCMP_ACT_ERRNO", "errnoRet": 1})
    return prof


def security_context(ep: EndpointPolicy, profile_file: Optional[str]) -> dict:
    drop = sorted(DOCKER_DEFAULT_CAPS - ep.caps)
    sc: dict = {"allowPrivilegeEscalation": False,
                "capabilities": {"drop": [c.replace("CAP_", "") for c in drop]}}
    if profile_file:
        sc["seccompProfile"] = {"type": "Localhost", "localhostProfile": profile_file}
    return sc


# ---------------------------------------------------------------------------------------
# networking.k8s.io/v1 NetworkPolicy -> Policy
# ---------------------------------------------------------------------------------------
def _match_workloads(topo: Topology, selector: Optional[dict]) -> List[int]:
    if selector is None:
        return []
    ml = (selector or {}).get("matchLabels", {}) or {}
    out = []
    for i, w in enumerate(topo.workloads):
        if all(w.labels.get(k) == v for k, v in ml.items()):
            out.append(i)
    return out


def _cidr_peers(topo: Topology, cidr: str, excepts: Iterable[str]) -> List[int]:
    import ipaddress
    net = ipaddress.ip_network(cidr, strict=False)
    exc = [ipaddress.ip_network(e, strict=False) for e in excepts]
    out = []
    for i in range(topo.n, topo.V):
        e = topo.externals[i - topo.n]
        enet = ipaddress.ip_network(e.cidr, strict=False)
        if enet.subnet_of(net) or net.subnet_of(enet):
            if not any(enet.subnet_of(x) for x in exc):
                out.append(i)
    return out


def _np_ports(topo: Topology, ports: Optional[list]) -> int:
    if not ports:
        return topo.ps.ALL
    m = 0
    for p in ports:
        proto = UDP if str(p.get("protocol", "TCP")).upper() == "UDP" else TCP
        port = p.get("port")
        if port is None:
            m |= topo.ps.mask_all_proto(proto)
            continue
        if isinstance(port, str) and not port.isdigit():
            continue  # named container port: not resolvable from the model
        lo = int(port)
        hi = int(p.get("endPort", lo))
        m |= topo.ps.mask_of_range(proto, lo, hi)
    return m


def netpols_to_policy(docs: Iterable[dict], topo: Topology, origin: str = "upstream") -> Policy:
    P = Policy(topo)
    everything = frozenset(range(topo.V))
    for doc in docs:
        if not doc or doc.get("kind") != "NetworkPolicy":
            continue
        spec = doc["spec"]
        subj = frozenset(_match_workloads(topo, spec.get("podSelector", {})))
        if not subj:
            continue
        types = spec.get("policyTypes") or (["Ingress"] + (["Egress"] if "egress" in spec else []))
        for direction, key, peer_key in ((IN, "ingress", "from"), (OUT, "egress", "to")):
            if ("Ingress" if direction == IN else "Egress") not in types:
                continue
            rules = spec.get(key) or []
            if not rules:
                P.add_rule(Rule(direction, subj, frozenset(), 0, ALLOW, 0, origin))  # selected, nothing allowed
                continue
            for r in rules:
                ports = _np_ports(topo, r.get("ports"))
                peers_spec = r.get(peer_key)
                if not peers_spec:
                    peers = everything
                else:
                    peers = set()
                    for pe in peers_spec:
                        if "ipBlock" in pe:
                            ib = pe["ipBlock"]
                            peers.update(_cidr_peers(topo, ib["cidr"], ib.get("except", [])))
                        elif "podSelector" in pe or "namespaceSelector" in pe:
                            if "podSelector" in pe:
                                peers.update(_match_workloads(topo, pe["podSelector"]))
                            else:
                                peers.update(range(topo.n))
                    peers = frozenset(peers)
                P.add_rule(Rule(direction, subj, peers, ports, ALLOW, 0, origin))
    return P


def cnps_to_yaml(cnps: List[dict]) -> str:
    import yaml
    return yaml.safe_dump_all(cnps, sort_keys=False)
