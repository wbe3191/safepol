"""Experiment K-C: the response runbook with and without the shield, closing the loop on the real cluster.

Controller loop (Delta = 5 s):
  telemetry  : Hubble flows of the last period (policy verdicts; sources/destinations mapped to the
               model's endpoint space: pods by label, the node's host identity -> NodeCIDR,
               169.254.169.254 -> MetadataService, the API server backend -> APIServer)
  detector   : 'scan' = a workload opened connections to >= 4 (peer, port) pairs it never used before;
               'metadata' = any egress attempt to 169.254.169.254
  runbook    : B3's playbook on a network alert = DENY_OUT then DENY_IN on the alerting workload with the
               wildcard template "everything except the operator's dependency list" (the static G_dep)
  shield     : (B8 only) verify_or_repair against Phi, phi6's window being the flows Hubble observed as
               forwarded between modelled endpoints in the last 12 periods (alert-filtered, as in the
               simulator); B3 applies the runbook's candidate unverified
  actuation  : Policy -> CiliumNetworkPolicy objects -> kubectl apply
Attacker (benign stand-in): an ephemeral netshoot container in the victim pod issuing TCP connects
(`nc -z`) / `curl`s that reproduce the flow events of S2 (scan, then redis-cart/paymentservice) and S3
(metadata address, then API server).  Nothing is exploited; the pod identity is simply reused.
Objective: S2 = victim reaches redis-cart:6379 or paymentservice:50051; S3 = victim reaches the API server.
Availability: frontend HTTP status sampled from the loadgenerator pod, victim pod readiness and container
restarts (kubelet probes come from the host), and Hubble drops on dependency edges.
Outputs results/k8s/e2e.csv (one row per episode) and results/k8s/e2e_log.jsonl.
"""
from __future__ import annotations
import csv, json, os, subprocess, sys, time
sys.path.insert(0, os.getcwd())
from safepol.bench.topology import online_boutique, make_port_space
from safepol.bench.policies import make_invariants, initial_policy
from safepol.sim.action_space import ActionSpace, Action, K_IDX
from safepol.verify.shield import Shield
from safepol.model.policy_model import TCP
from safepol.k8s.translate import policy_to_cnps, cnps_to_yaml
from safepol.k8s import cluster as K

OUT = "results/k8s"; os.makedirs(OUT, exist_ok=True)
DELTA = 5.0
WINDOW = 12
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SCENARIOS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["S2", "S3"]
VARIANTS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["B3", "B8"]
VICTIMS = {"S2": "currencyservice", "S3": "recommendationservice"}
META_IP, API_VIP = "169.254.169.254", "10.96.0.1"
T_ATTACK, T_END, HORIZON = 2, 10, 16

topo = online_boutique(make_port_space(), with_promo=True)
inv = make_invariants(topo)            # live invariants: phi6 window fed from Hubble
inv_static = make_invariants(topo)     # the operator's dependency list only (runbook template)
aspace = ActionSpace(topo)
shield = Shield(inv, aspace, "maxsat", 2000, prefer_additions=True)
ips = {topo.names[i]: K.pod_ip(topo.names[i]) for i in range(topo.n)}
ip2w = {v: k for k, v in ips.items()}
ep = json.loads(K.kubectl("get", "endpoints", "kubernetes", "-o", "json"))["subsets"][0]
API_EP = (ep["addresses"][0]["ip"], ep["ports"][0]["port"])
dep_pairs = {(topo.names[s], topo.names[d]) for s, d, _ in topo.dep_edges}
API_IS_HOST = K.apiserver_is_host()
print("host identity carries kube-apiserver label:", API_IS_HOST)
log = open(f"{OUT}/e2e_log.jsonl", "a")


def apply_policy(P, notes=None):
    K.apply_yaml(cnps_to_yaml(policy_to_cnps(P, apiserver_is_host=API_IS_HOST, notes=notes)))


def reset_policy():
    K.delete_managed(); time.sleep(2)
    P0 = initial_policy(topo, inv)
    apply_policy(P0); time.sleep(3)
    return P0


def reset_workload(victim):
    """Fresh victim pod (a runbook episode may have left it in CrashLoopBackOff after liveness kills),
    refreshed IP map, debug container re-attached, frontend serving 200 again."""
    global ips, ip2w
    K.delete_managed()
    K.kubectl("-n", "boutique", "rollout", "restart", f"deploy/{victim}")
    K.kubectl("-n", "boutique", "rollout", "status", f"deploy/{victim}", "--timeout=180s")
    time.sleep(3)
    ips = {topo.names[i]: K.pod_ip(topo.names[i]) for i in range(topo.n)}
    ip2w = {v: k for k, v in ips.items()}
    vpod = K.ensure_debug_container(victim)
    lg = K.ensure_debug_container("loadgenerator")
    for _ in range(30):
        if frontend_health(lg, 2) == ["200", "200"]:
            break
        time.sleep(3)
    return vpod, lg


def endpoint_name(side: dict, ip: str, port: int):
    labels = side.get("labels") or []
    pod = side.get("pod_name", "")
    w = next((n for n in ips if pod.startswith(n + "-")), None)
    if w:
        return w
    if "reserved:host" in labels or "reserved:remote-node" in labels:
        return "NodeCIDR"
    if "reserved:kube-apiserver" in labels or (ip, port) == API_EP or ip == API_VIP:
        return "APIServer"
    if ip == META_IP:
        return "MetadataService"
    if port == 53 or "k8s-app=kube-dns" in " ".join(labels):
        return "kube-dns"
    if "reserved:world" in labels:
        return "Internet"
    return None


def period_flows(since_s: float):
    """(src, dst, dst_port, verdict) for the last period, endpoints in the model's name space."""
    out = []
    for f in K.hubble_flows(f"{int(since_s)+1}s", extra=["--type", "policy-verdict", "--type", "trace:to-endpoint", "--type", "drop"]):
        ipf = f.get("IP") or {}
        l4 = (f.get("l4") or {}).get("TCP") or (f.get("l4") or {}).get("UDP") or {}
        sp, dp = l4.get("source_port", 0), l4.get("destination_port", 0)
        src = endpoint_name(f.get("source") or {}, ipf.get("source", ""), sp)
        dst = endpoint_name(f.get("destination") or {}, ipf.get("destination", ""), dp)
        if src is None or dst is None or src == dst:
            continue
        if f.get("is_reply"):
            continue
        out.append((src, dst, dp, f.get("verdict", "")))
    return out


def detect(flows, seen, baseline):
    """``baseline``: every (src, dst, port) attempted before the attack, whatever the verdict (the
    upstream frontend, for instance, probes the cloud metadata address at start-up); ``seen``: forwarded
    flows, which feed phi6.  Alerts are raised on behaviour outside the baseline only."""
    alerts = {}
    new_by_src = {}
    for sw, dn, dp, verdict in flows:
        if sw in ips and dn == "MetadataService" and (sw, dn, dp) not in baseline:
            alerts[sw] = "metadata"
        if sw in ips and dn in ips and (sw, dn, dp) not in seen and (sw, dn, dp) not in baseline:
            new_by_src.setdefault(sw, set()).add((dn, dp))
    for sw, s in new_by_src.items():
        if len(s) >= 4:
            alerts.setdefault(sw, "scan")
    period_legit = set()
    for sw, dn, dp, verdict in flows:
        if verdict == "FORWARDED" and sw not in alerts and dn not in alerts and dn != "kube-dns":
            seen.add((sw, dn, dp)); period_legit.add((sw, dn, dp))
    return alerts, period_legit


def to_model_flow(sw, dn, dp):
    if sw in topo.index and dn in topo.index:
        return (topo.index[sw], topo.index[dn], topo.ps.mask_of_port(TCP, dp))
    return None


def legit_drops(flows):
    return sum(1 for sw, dn, dp, v in flows if v == "DROPPED" and ((sw, dn) in dep_pairs or sw == "NodeCIDR"))


def objective_open(P, sid, victim):
    v = topo.index[victim]
    if sid == "S2":
        return any(P.allows(v, t, aspace.svc_mask[t]) for t in (topo.index["redis-cart"], topo.index["paymentservice"]))
    return P.allows(v, topo.by_kind("APIServer"), topo.ps.mask_of_port(TCP, 443))


def objective_open_real(pod, sid):
    if sid == "S2":
        ok1, _ = K.tcp_probe(pod, ips["redis-cart"], 6379, 2); ok2, _ = K.tcp_probe(pod, ips["paymentservice"], 50051, 2)
        return ok1 or ok2
    ok, _ = K.tcp_probe(pod, API_EP[0], API_EP[1], 2)
    return ok


def attacker_step(pod, sid, stage):
    """Benign stand-in for the attacker's flow events: TCP connect attempts only."""
    if sid == "S2":
        if stage == 1:
            targets = " ".join(f"{ips[n]}:{p}" for n in ips if n != VICTIMS[sid] for p in
                               ([port for _, (_, port) in topo.workloads[topo.index[n]].ports.items()] + [22]))
            cmd = f"for t in {targets}; do nc -z -w 1 ${{t%%:*}} ${{t##*:}} >/dev/null 2>&1 & done; wait"
        else:
            cmd = f"nc -z -w 1 {ips['redis-cart']} 6379; nc -z -w 1 {ips['paymentservice']} 50051"
    else:
        cmd = (f"curl -s -m 1 -o /dev/null http://{META_IP}/latest/meta-data/" if stage == 1
               else f"curl -s -m 2 -k -o /dev/null https://{API_VIP}:443/version")
    subprocess.Popen(["kubectl", "-n", "boutique", "exec", pod, "-c", "dbg", "--", "sh", "-c", cmd],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def frontend_health(lg, n=5):
    codes = []
    for _ in range(n):
        rc, out = K.exec_in(lg, f"curl -s -o /dev/null -m 5 -w '%{{http_code}}' http://{ips['frontend']}:8080/")
        codes.append(out.strip()); time.sleep(0.5)
    return codes


def victim_state(victim):
    j = json.loads(K.kubectl("-n", "boutique", "get", "pod", K.pod_of(victim), "-o", "json"))
    ready = any(c["type"] == "Ready" and c["status"] == "True" for c in j["status"].get("conditions", []))
    restarts = sum(c.get("restartCount", 0) for c in j["status"].get("containerStatuses", []))
    return ready, restarts


rows = []
for sid in SCENARIOS:
    victim = VICTIMS[sid]
    for variant in VARIANTS:
        for epi in range(EPISODES):
            vpod, lg = reset_workload(victim)
            P = reset_policy()
            seen, window, baseline = set(), [], set()
            inv.legit_flows = set(topo.dep_edges)
            for _ in range(2):   # warm-up observation: everything attempted before the attack is baseline
                time.sleep(DELTA)
                fl = period_flows(DELTA)
                baseline |= {(sw, dn, dp) for sw, dn, dp, v in fl}
                pl = {(sw, dn, dp) for sw, dn, dp, v in fl if v == "FORWARDED" and dn != "kube-dns"}
                seen |= pl; window.append(pl)
            ready0, restarts0 = victim_state(victim)
            http0 = frontend_health(lg, 3)
            queue, done, alerts_log, actions_log = [], set(), [], []
            t_attack = t_mit = None; real_closed = None; drops = 0; interv = 0; changes = 0; not_ready_periods = 0
            for t in range(HORIZON):
                if t == T_ATTACK:
                    t_attack = time.time()
                if T_ATTACK <= t < T_END:
                    attacker_step(vpod, sid, 1 if t < T_ATTACK + 2 else 2)
                time.sleep(DELTA)
                flows = period_flows(DELTA)
                drops += legit_drops(flows)
                al, pl = detect(flows, seen, baseline)
                window = (window + [pl])[-WINDOW:]
                inv.legit_flows = set(topo.dep_edges) | {m for s in window for m in (to_model_flow(*x) for x in s) if m}
                act = None
                if queue:
                    act = queue.pop(0)
                else:
                    for w, kind in al.items():
                        if w in ips and w not in done:
                            done.add(w); alerts_log.append((t, w, kind)); wi = topo.index[w]
                            queue = [Action(K_IDX["DENY_IN"], wi, topo.V, 0)]
                            act = Action(K_IDX["DENY_OUT"], wi, topo.V, 0)
                            break
                if act is not None:
                    P_cand = aspace.apply(P, act, None, inv_static)
                    t0 = time.perf_counter()
                    if variant == "B8":
                        ok, P_next, info = shield.verify_or_repair(P_cand, P, changed={act.w})
                    else:
                        ok, P_next, info = True, P_cand, {"violations": inv.check(P_cand), "method": "none", "distance": 0}
                    t_shield = (time.perf_counter() - t0) * 1e3
                    interv += int(not ok)
                    notes = []
                    t0 = time.perf_counter(); apply_policy(P_next, notes); t_apply = (time.perf_counter() - t0) * 1e3
                    P = P_next; changes += 1
                    viol = [(v if isinstance(v, str) else v.inv + ":" + v.detail) for v in info.get("violations", [])]
                    actions_log.append(dict(t=t, action=act.describe(topo), safe=ok, violations=viol, method=info.get("method"),
                                            distance=info.get("distance"), shield_ms=round(t_shield, 1), apply_ms=round(t_apply), notes=notes))
                    print(f"  [{sid} {variant} ep{epi} t={t}] {act.describe(topo)} safe={ok} viol={viol[:4]} method={info.get('method')} "
                          f"d={info.get('distance')} shield={t_shield:.1f}ms apply={t_apply:.0f}ms")
                if t_attack and t_mit is None and not objective_open(P, sid, victim):
                    t_mit = time.time(); time.sleep(1.0)
                    real_closed = not objective_open_real(vpod, sid)
                    print(f"  [{sid} {variant} ep{epi}] objective closed in model at +{t_mit - t_attack:.1f}s; real cluster closed={real_closed}")
                if t >= T_ATTACK + 2:
                    r, _ = victim_state(victim); not_ready_periods += int(not r)
            ready1, restarts1 = victim_state(victim)
            http1 = frontend_health(lg, 5)
            row = dict(scenario=sid, variant=variant, episode=epi, victim=victim,
                       alert_period=(alerts_log[0][0] if alerts_log else None), alert_kind=(alerts_log[0][2] if alerts_log else None),
                       mitigated=int(t_mit is not None), ttm_s=(round(t_mit - t_attack, 1) if t_mit else None),
                       real_closed=int(bool(real_closed)), interventions=interv, changes=changes, inv_ok=int(inv.satisfied(P)),
                       legit_drops=drops, victim_not_ready_periods=not_ready_periods, victim_restarts=restarts1 - restarts0,
                       frontend_200_before=sum(c == "200" for c in http0), frontend_200_after=sum(c == "200" for c in http1),
                       shield_ms_max=max([a["shield_ms"] for a in actions_log], default=0), apply_ms_max=max([a["apply_ms"] for a in actions_log], default=0))
            rows.append(row); print(row)
            log.write(json.dumps(dict(row=row, actions=actions_log, alerts=alerts_log, legit_window=sorted(seen))) + "\n"); log.flush()
            K.delete_managed(); time.sleep(3)
with open(f"{OUT}/e2e.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if f.tell() == 0:
        w.writeheader()
    w.writerows(rows)
