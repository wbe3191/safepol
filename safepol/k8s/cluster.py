"""Thin kubectl/Hubble wrapper used by the real-cluster experiments (no python-kubernetes dependency
on the hot path; everything is `kubectl`, which keeps the scripts auditable)."""
from __future__ import annotations

import json
import subprocess
import time
from typing import Dict, List, Optional, Tuple

NS = "boutique"
CILIUM_NS = "kube-system"


def sh(cmd: List[str], input: Optional[str] = None, check: bool = True, timeout: int = 120) -> str:
    r = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def kubectl(*args: str, input: Optional[str] = None, check: bool = True, timeout: int = 120) -> str:
    return sh(["kubectl", *args], input=input, check=check, timeout=timeout)


def apply_yaml(doc: str) -> str:
    return kubectl("apply", "-f", "-", input=doc)


def delete_managed(ns: str = NS) -> None:
    kubectl("-n", ns, "delete", "cnp", "-l", "safepol.io/managed=true", "--ignore-not-found", "--wait=false", check=False)


def pod_of(app: str, ns: str = NS) -> str:
    """Name of the live (Running, not terminating) pod of a deployment."""
    j = json.loads(kubectl("-n", ns, "get", "pod", "-l", f"app={app}", "-o", "json"))
    live = [p for p in j["items"] if p["status"].get("phase") == "Running" and not p["metadata"].get("deletionTimestamp")]
    live.sort(key=lambda p: p["metadata"]["creationTimestamp"], reverse=True)
    return live[0]["metadata"]["name"]


def pod_ip(app: str, ns: str = NS) -> str:
    return kubectl("-n", ns, "get", "pod", "-l", f"app={app}", "--field-selector=status.phase=Running",
                   "-o", "jsonpath={.items[0].status.podIP}").strip()


def ensure_debug_container(app: str, image: str = "nicolaka/netshoot:v0.13", name: str = "dbg", ns: str = NS) -> str:
    """Attach (once) an ephemeral netshoot container sharing the pod's network namespace, so that
    test flows carry the pod's Cilium identity.  Returns the pod name."""
    pod = pod_of(app, ns)
    j = json.loads(kubectl("-n", ns, "get", "pod", pod, "-o", "json"))
    have = [c["name"] for c in j["spec"].get("ephemeralContainers", [])]
    if name not in have:
        kubectl("-n", ns, "debug", pod, f"--image={image}", "--image-pull-policy=IfNotPresent", "-c", name,
                "--profile=general", "--", "sleep", "infinity", check=False)
        for _ in range(60):
            j = json.loads(kubectl("-n", ns, "get", "pod", pod, "-o", "json"))
            st = [s for s in j["status"].get("ephemeralContainerStatuses", []) if s["name"] == name]
            if st and "running" in st[0].get("state", {}):
                break
            time.sleep(0.5)
    return pod


def exec_in(pod: str, cmd: str, container: str = "dbg", ns: str = NS, timeout: int = 60) -> Tuple[int, str]:
    r = subprocess.run(["kubectl", "-n", ns, "exec", pod, "-c", container, "--", "sh", "-c", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def tcp_probe(pod: str, ip: str, port: int, wait: float = 2.0) -> Tuple[bool, float]:
    """nc -z from the pod's debug container.  Returns (connected_or_refused, elapsed_s): a policy drop
    manifests as a timeout, a refused connection (no listener) as an immediate failure; both the
    model and Cilium treat 'refused' as *allowed by policy*."""
    rc, out = exec_in(pod, f"s=$(date +%s.%N); nc -z -w {int(wait)} {ip} {port} >/dev/null 2>&1; rc=$?; e=$(date +%s.%N); echo RESULT $rc $(echo \"$e - $s\" | bc)")
    try:
        line = [l for l in out.splitlines() if l.startswith("RESULT")][-1]
        rc_nc, el = line.split()[1:3]
        rc_nc, el = int(rc_nc), float(el)
    except Exception:
        return False, float("nan")
    if rc_nc == 0:
        return True, el
    return (el < wait * 0.75), el   # fast failure = refused (reachable); slow = dropped


def hubble_verdicts(src_pod: str, to_ip: str, to_port: int, since: str = "15s", ns: str = NS) -> List[str]:
    out = kubectl("-n", CILIUM_NS, "exec", "ds/cilium", "-c", "cilium-agent", "--", "hubble", "observe",
                  "--from-pod", f"{ns}/{src_pod}", "--to-ip", to_ip, "--to-port", str(to_port),
                  "--type", "policy-verdict", "--since", since, "-o", "json", check=False)
    vs = []
    for line in out.splitlines():
        try:
            f = json.loads(line).get("flow", {})
        except Exception:
            continue
        vs.append(f.get("verdict", "?"))
    return vs


def hubble_flows(since: str, ns: str = NS, extra: Optional[List[str]] = None) -> List[dict]:
    out = kubectl("-n", CILIUM_NS, "exec", "ds/cilium", "-c", "cilium-agent", "--", "hubble", "observe",
                  "-n", ns, "--since", since, "-o", "json", *(extra or []), check=False, timeout=180)
    flows = []
    for line in out.splitlines():
        try:
            flows.append(json.loads(line)["flow"])
        except Exception:
            pass
    return flows


def cilium_policy_revision(ns: str = NS) -> Dict[str, int]:
    """Per-endpoint policy revision from CiliumEndpoint objects."""
    j = json.loads(kubectl("-n", ns, "get", "cep", "-o", "json"))
    return {it["metadata"]["name"]: it.get("status", {}).get("policy", {}).get("realized", {}).get("ingress", {}).get("enforcing", 0)
            for it in j["items"]}


def apiserver_is_host() -> bool:
    """True when the host identity (ID 1) also carries reserved:kube-apiserver (API server on this node)."""
    out = kubectl("-n", CILIUM_NS, "exec", "ds/cilium", "-c", "cilium-agent", "--", "cilium-dbg", "identity", "get", "1", check=False)
    return "reserved:kube-apiserver" in out
