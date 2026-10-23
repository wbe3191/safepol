"""Experiment K-A: the shield on the upstream Online Boutique network policies, in a real cluster.

1. Parse the 13 upstream networking.k8s.io/v1 NetworkPolicy objects (v0.10.1 kustomize component)
   into the model (topology with and without the promoservice drift).
2. Verify against Phi; repair with the shield; record violations, latency, edit distance.
3. Translate the (repaired) policy to CiliumNetworkPolicy objects and apply them.
4. Fidelity: for a set of probe flows generated from inside the pods (ephemeral netshoot containers,
   so flows carry the pod identity), compare the model's reachability with Cilium's actual verdict.
Outputs results/k8s/upstream_shield.csv, results/k8s/fidelity.csv, k8s/generated/*.yaml.
"""
from __future__ import annotations

import csv, glob, json, os, sys, time
import yaml

sys.path.insert(0, os.getcwd())
from safepol.bench.topology import online_boutique, make_port_space
from safepol.bench.policies import make_invariants
from safepol.sim.action_space import ActionSpace
from safepol.verify.shield import Shield
from safepol.model.policy_model import ALLOW, DENY, TCP
from safepol.k8s.translate import netpols_to_policy, policy_to_cnps, cnps_to_yaml
from safepol.k8s import cluster as K

OUT = "results/k8s"; os.makedirs(OUT, exist_ok=True); os.makedirs("k8s/generated", exist_ok=True)
DO_CLUSTER = "--no-cluster" not in sys.argv


def load_upstream():
    docs = []
    for f in sorted(glob.glob("k8s/upstream/network-policy-*.yaml")):
        docs += [d for d in yaml.safe_load_all(open(f)) if d]
    return docs


def describe_rule(topo, r):
    subj = ",".join(topo.names[i] for i in sorted(r.subject))
    peer = "*" if len(r.peer) >= topo.V - 1 else ",".join(topo.names[i] for i in sorted(r.peer))
    return f"{r.verdict} {r.direction} {subj} <-> {peer} ports={topo.ps.describe(r.ports)}"


rows = []
policies = {}
for drift in (False, True):
    topo = online_boutique(make_port_space(), with_promo=drift)
    inv = make_invariants(topo)
    shield = Shield(inv, ActionSpace(topo), "maxsat", 2000, prefer_additions=True)
    P = netpols_to_policy(load_upstream(), topo)
    t0 = time.perf_counter(); viol = inv.check(P); t_check = (time.perf_counter() - t0) * 1e3
    # repeat verify_or_repair for latency statistics (deterministic)
    lat = []
    for _ in range(20):
        t0 = time.perf_counter(); ok, P_safe, info = shield.verify_or_repair(P, P); lat.append((time.perf_counter() - t0) * 1e3)
    lat.sort()
    added = [r for r in P_safe.rules if r.rid not in P.rule_ids()]
    removed = [r for r in P.rules if r.rid not in P_safe.rule_ids()]
    print(f"[drift={drift}] rules={len(P.rules)} violations={len(viol)} :", sorted({v.inv for v in viol}))
    for v in viol:
        print("   ", v.inv, v.detail)
    print(f"   check {t_check:.2f} ms; verify_or_repair median {lat[10]:.1f} ms (min {lat[0]:.1f}, max {lat[-1]:.1f}); "
          f"method={info.get('method')} distance={info.get('distance')} added={len(added)} removed={len(removed)}; safe_after={inv.satisfied(P_safe)}")
    for r in added:
        print("    +", describe_rule(topo, r))
    for r in removed:
        print("    -", describe_rule(topo, r))
    if "--dump" in sys.argv:
        print("   == P_safe rules ==")
        for r in P_safe.rules:
            print("     ", r.rid, r.origin, describe_rule(topo, r))
    rows.append(dict(drift=int(drift), n_rules=len(P.rules), n_viol=len(viol), phis=" ".join(sorted({v.inv for v in viol})),
                     check_ms=round(t_check, 3), vor_median_ms=round(lat[10], 2), vor_max_ms=round(lat[-1], 2),
                     method=info.get("method"), distance=info.get("distance"), added=len(added), removed=len(removed),
                     safe_after=int(inv.satisfied(P_safe))))
    cnps = policy_to_cnps(P_safe)
    open(f"k8s/generated/upstream_repaired_drift{int(drift)}.yaml", "w").write(cnps_to_yaml(cnps))
    open(f"k8s/generated/upstream_asis_drift{int(drift)}.yaml", "w").write(cnps_to_yaml(policy_to_cnps(P)))
    policies[drift] = (topo, inv, P, P_safe, cnps)

with open(f"{OUT}/upstream_shield.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if not DO_CLUSTER:
    sys.exit(0)

# ------------------------------------------------------------------------------------------
# Fidelity check on the cluster: repaired policy with drift topology (promoservice is deployed)
# ------------------------------------------------------------------------------------------
topo, inv, P, P_safe, cnps = policies[True]
K.delete_managed(); time.sleep(2)
K.apply_yaml(cnps_to_yaml(cnps))
print("applied", len(cnps), "CiliumNetworkPolicies; waiting for enforcement"); time.sleep(8)

EXT_IP = {"Internet": "1.1.1.1", "MetadataService": "169.254.169.254", "APIServer": "10.96.0.1", "NodeCIDR": None}
EXT_PORT = {"Internet": 443, "MetadataService": 80, "APIServer": 443}
ep = json.loads(K.kubectl("get", "endpoints", "kubernetes", "-o", "json"))["subsets"][0]
API_EP = (ep["addresses"][0]["ip"], ep["ports"][0]["port"])
print("api server backend", API_EP)
sources = ["frontend", "checkoutservice", "cartservice", "currencyservice", "adservice", "promoservice", "loadgenerator", "recommendationservice"]
ips = {topo.names[i]: K.pod_ip(topo.names[i]) for i in range(topo.n)}
pods = {s: K.ensure_debug_container(s) for s in sources}
print("debug containers ready:", pods)

fid = []
for s in sources:
    si = topo.index[s]
    probes = []
    for d in range(topo.V):
        if d == si:
            continue
        if topo.is_workload(d):
            for pn, (proto, port) in topo.workloads[d].ports.items():
                probes.append((d, ips[topo.names[d]], port, topo.ps.mask_of_port(proto, port)))
            probes.append((d, ips[topo.names[d]], 22, topo.ps.mask_of_port(TCP, 22)))
        else:
            kind = topo.ext_kind[d]
            if EXT_IP.get(kind):
                probes.append((d, EXT_IP[kind], EXT_PORT[kind], topo.ps.mask_of_port(TCP, EXT_PORT[kind])))
    for d, ip, port, m in probes:
        model = P_safe.allows(si, d, m)
        ok, el = K.tcp_probe(pods[s], ip, port, wait=2.0)
        # service VIPs are translated before policy (kube-proxy replacement): filter on the backend
        hip, hport = (API_EP if ip == EXT_IP["APIServer"] else (ip, port))
        vs = K.hubble_verdicts(pods[s], hip, hport, since="12s")
        hub = "DROPPED" if "DROPPED" in vs else ("FORWARDED" if ("FORWARDED" in vs or "AUDIT" in vs) else "-")
        actual = (hub == "FORWARDED") if hub != "-" else ok
        fid.append(dict(src=s, dst=topo.names[d], port=port, model=int(model), actual=int(actual), hubble=hub, nc=int(ok), elapsed=round(el, 2)))
        flag = "" if model == actual else "   <-- MISMATCH"
        print(f"{s:>22} -> {topo.names[d]:<22}:{port:<5} model={int(model)} actual={int(actual)} ({hub}, nc={int(ok)}, {el:.2f}s){flag}")

with open(f"{OUT}/fidelity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fid[0].keys())); w.writeheader(); w.writerows(fid)
agree = sum(r["model"] == r["actual"] for r in fid)
print(f"fidelity: {agree}/{len(fid)} probes agree ({100*agree/len(fid):.1f}%)")
