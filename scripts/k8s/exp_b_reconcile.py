"""Experiment K-B: reconciliation latency of a shield-approved policy change in the real cluster.

(1) Network: a poller inside the frontend pod's debug container attempts a TCP connect to
    productcatalogservice:3550 every ~15 ms and logs the attempt start time and result.  While it runs
    we `kubectl apply` an egressDeny rule for that flow and later delete it.  Pod and kubectl share
    one WSL2 kernel clock, so enforcement latency = first failing attempt start - apply return time,
    and re-open latency = first succeeding attempt start - delete return time.
(2) Host: tighten seccomp / drop capabilities is a pod securityContext change that Kubernetes
    realises by re-creating the pod; we measure deployment patch -> new pod Ready for currencyservice
    and confirm the profile from /proc/1/status in the new pod.
Outputs results/k8s/reconcile_net.csv, results/k8s/reconcile_host.csv.
"""
from __future__ import annotations
import csv, json, os, subprocess, sys, time, statistics
sys.path.insert(0, os.getcwd())
from safepol.k8s import cluster as K
from safepol.k8s.translate import seccomp_profile, security_context
from safepol.model.policy_model import EndpointPolicy, SYSCALL_UNIVERSE, DOCKER_DEFAULT_CAPS, SYSCALL_GROUPS

OUT = "results/k8s"; os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NET_ONLY = "--net-only" in sys.argv
SRC, DST, PORT = "frontend", "productcatalogservice", 3550
HOST_SVC = "currencyservice"
pod = K.ensure_debug_container(SRC)
dst_ip = K.pod_ip(DST)
DENY = f"""apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: safepol-toggle
  namespace: boutique
  labels: {{safepol.io/managed: "true"}}
spec:
  endpointSelector: {{matchLabels: {{app: {SRC}}}}}
  egressDeny:
  - toEndpoints: [{{matchLabels: {{app: {DST}}}}}]
    toPorts: [{{ports: [{{port: "{PORT}", protocol: TCP}}]}}]
"""
# poller runs under the pod's python3 (netshoot image): sub-millisecond timestamps from the shared
# WSL2 kernel clock; busybox `date` has only 1 s resolution.
POLL = ("import socket,time,sys\n"
        "end=time.time()+{dur}\n"
        "while time.time()<end:\n"
        "  s=time.time(); c=socket.socket(); c.settimeout(0.1)\n"
        "  try: c.connect((sys.argv[1],int(sys.argv[2]))); r=0\n"
        "  except Exception: r=1\n"
        "  c.close(); print(s, r, time.time(), flush=True); time.sleep(0.01)\n")

def run_trial(i: int) -> dict:
    K.kubectl("-n", "boutique", "delete", "cnp", "safepol-toggle", "--ignore-not-found", check=False)
    time.sleep(1.5)
    p = subprocess.Popen(["kubectl", "-n", "boutique", "exec", pod, "-c", "dbg", "--", "python3", "-c", POLL.format(dur=7), dst_ip, str(PORT)],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    time.sleep(2.0)
    t0 = time.time(); K.apply_yaml(DENY); t_apply = time.time()
    time.sleep(2.5)
    t1 = time.time(); K.kubectl("-n", "boutique", "delete", "cnp", "safepol-toggle"); t_del = time.time()
    out, _ = p.communicate(timeout=60)
    samples = []
    for line in out.splitlines():
        try:
            ts, rc, te = line.split(); samples.append((float(ts), int(rc), float(te)))
        except ValueError:
            pass
    fail = [ts for ts, rc, te in samples if rc != 0 and ts > t0]
    reopen = [ts for ts, rc, te in samples if rc == 0 and ts > t1]
    # the last successful attempt that *finished* before the first failing attempt started
    last_ok_end = max([te for ts, rc, te in samples if rc == 0 and te < (fail[0] if fail else t_apply)], default=float("nan"))
    enforce = (fail[0] - t_apply) if fail else float("nan")
    gap = (fail[0] - last_ok_end) if fail else float("nan")          # measurement resolution around the transition
    # a blocked attempt only fails after its 0.1 s connect timeout, so the first success is observed
    # between L and L+0.11 s after the true re-open instant L: report the previous failure's start too
    reop = (reopen[0] - t_del) if reopen else float("nan")
    last_fail_start = max([ts for ts, rc, te in samples if rc != 0 and ts < (reopen[0] if reopen else t_del)], default=float("nan"))
    reop_lo = (last_fail_start - t_del) if reopen else float("nan")
    return dict(trial=i, samples=len(samples), apply_rtt_s=round(t_apply - t0, 3), enforce_s=round(enforce, 3),
                resolution_s=round(gap, 3), delete_rtt_s=round(t_del - t1, 3), reopen_s=round(reop, 3), reopen_lo_s=round(reop_lo, 3))


rows = [run_trial(i) for i in range(N)]
for r in rows:
    print(r)
with open(f"{OUT}/reconcile_net.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
enf = sorted(r["enforce_s"] for r in rows if r["enforce_s"] == r["enforce_s"])
reo = sorted(r["reopen_s"] for r in rows if r["reopen_s"] == r["reopen_s"])
print(f"network: apply rtt {statistics.median(r['apply_rtt_s'] for r in rows):.3f}s; enforce median {statistics.median(enf):.3f}s "
      f"p95 {enf[max(0, int(0.95*len(enf))-1)]:.3f} max {enf[-1]:.3f}; reopen median {statistics.median(reo):.3f}s max {reo[-1]:.3f}; "
      f"resolution median {statistics.median(r['resolution_s'] for r in rows):.3f}s; reopen lower bound median {statistics.median(r['reopen_lo_s'] for r in rows):.3f}s")

# ---------------------------------------------------------------------------------------------
if NET_ONLY:
    sys.exit(0)
tight = EndpointPolicy(SYSCALL_UNIVERSE - SYSCALL_GROUPS["danger"],
                       DOCKER_DEFAULT_CAPS - {"CAP_NET_RAW", "CAP_SETUID", "CAP_SETGID", "CAP_SYS_CHROOT", "CAP_MKNOD", "CAP_DAC_OVERRIDE", "CAP_FOWNER"})
prof = seccomp_profile(tight)
os.makedirs("k8s/generated", exist_ok=True)
open("k8s/generated/tight.json", "w").write(json.dumps(prof, indent=1))
node = K.kubectl("get", "nodes", "-o", "jsonpath={.items[0].metadata.name}").strip()
K.sh(["docker", "exec", node, "mkdir", "-p", "/var/lib/kubelet/seccomp/safepol"])
K.sh(["docker", "cp", "k8s/generated/tight.json", f"{node}:/var/lib/kubelet/seccomp/safepol/tight.json"])
sc_tight = security_context(tight, "safepol/tight.json")
BASE_SC = {"allowPrivilegeEscalation": False, "privileged": False, "readOnlyRootFilesystem": True}
sc_tight = {**BASE_SC, **sc_tight}
sc_relax = {**BASE_SC, "capabilities": {"drop": []}, "seccompProfile": {"type": "RuntimeDefault"}}


def proc_status(podname: str, tag: str) -> str:
    K.kubectl("-n", "boutique", "debug", podname, "--image=busybox:latest", "--image-pull-policy=IfNotPresent", "--target=server",
              "-c", tag, "--profile=general", "--", "sh", "-c", "grep -E 'Seccomp:|CapBnd' /proc/1/status", check=False)
    for _ in range(40):
        j = json.loads(K.kubectl("-n", "boutique", "get", "pod", podname, "-o", "json"))
        st = [s for s in j["status"].get("ephemeralContainerStatuses", []) if s["name"] == tag]
        if st and "terminated" in st[0].get("state", {}):
            break
        time.sleep(0.5)
    return K.kubectl("-n", "boutique", "logs", podname, "-c", tag, check=False).strip().replace("\n", " | ")


hrows = []
for i in range(max(3, N // 4)):
    for label, sc in (("tighten", sc_tight), ("relax", sc_relax)):
        old = K.pod_of(HOST_SVC)
        patch = [{"op": "replace", "path": "/spec/template/spec/containers/0/securityContext", "value": sc}]
        t0 = time.perf_counter()
        K.kubectl("-n", "boutique", "patch", "deploy", HOST_SVC, "--type", "json", "-p", json.dumps(patch))
        while True:
            j = json.loads(K.kubectl("-n", "boutique", "get", "pod", "-l", f"app={HOST_SVC}", "-o", "json"))
            ready = [p for p in j["items"] if p["metadata"]["name"] != old and not p["metadata"].get("deletionTimestamp") and
                     any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
            if ready:
                t_ready = time.perf_counter() - t0; break
            time.sleep(0.05)
        K.kubectl("-n", "boutique", "rollout", "status", f"deploy/{HOST_SVC}", "--timeout=120s")
        t_done = time.perf_counter() - t0
        status = proc_status(K.pod_of(HOST_SVC), f"chk{i}{label[0]}")
        hrows.append(dict(trial=i, op=label, new_pod_ready_s=round(t_ready, 2), rollout_done_s=round(t_done, 2), status=status))
        print(hrows[-1]); time.sleep(2)
with open(f"{OUT}/reconcile_host.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hrows[0].keys())); w.writeheader(); w.writerows(hrows)
print("host: new pod ready median", statistics.median(r["new_pod_ready_s"] for r in hrows if r["op"] == "tighten"), "s")
