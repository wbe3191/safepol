"""Summarise results/k8s/*.csv into results/k8s/summary.txt and paper/tables/k8s_e2e.tex."""
import pandas as pd, numpy as np, os
os.makedirs("paper/tables", exist_ok=True)
e = pd.read_csv("results/k8s/e2e.csv")
g = e.groupby(["scenario", "variant"])
rows = []
for (sid, var), d in g:
    rows.append(dict(scenario=sid, variant=var, n=len(d), ttm=d.ttm_s.mean(), closed=f"{int(d.real_closed.sum())}/{len(d)}",
                     interv=d.interventions.mean(), drops=d.legit_drops.mean(), notready=d.victim_not_ready_periods.mean(),
                     restarts=d.victim_restarts.mean(), fe_after=d.frontend_200_after.mean() / 5 * 100,
                     shield_ms=d.shield_ms_max.max(), apply_ms=d.apply_ms_max.max()))
s = pd.DataFrame(rows)
print(s.to_string())
name = {"B3": "runbook (\\textsf{B3})", "B8": "shielded (\\textsf{B8})"}
with open("paper/tables/k8s_e2e.tex", "w") as f:
    for r in rows:
        f.write(f"{r['scenario']} & {name[r['variant']]} & {r['ttm']:.1f} & {r['interv']:.1f} & {r['drops']:.0f} & "
                f"{r['notready']:.0f}/12 & {r['restarts']:.0f} & {r['fe_after']:.0f} \\\\\n")
    f.write("\\bottomrule\n")
net = pd.read_csv("results/k8s/reconcile_net.csv"); host = pd.read_csv("results/k8s/reconcile_host.csv")
with open("results/k8s/summary.txt", "w") as f:
    f.write(s.to_string() + "\n\n")
    f.write(f"net: n={len(net)} apply_rtt median {net.apply_rtt_s.median()*1e3:.0f} ms; enforce median {net.enforce_s.median()*1e3:.0f} ms, "
            f"p95 {net.enforce_s.quantile(.95)*1e3:.0f}, max {net.enforce_s.max()*1e3:.0f}; delete rtt {net.delete_rtt_s.median()*1e3:.0f} ms; "
            f"reopen median {net.reopen_s.median()*1e3:.0f} ms max {net.reopen_s.max()*1e3:.0f}\n")
    for op, d in host.groupby("op"):
        f.write(f"host {op}: n={len(d)} ready median {d.new_pod_ready_s.median():.2f} s (min {d.new_pod_ready_s.min():.2f}, max {d.new_pod_ready_s.max():.2f}); {d.status.iloc[0]}\n")
    f.write(open("results/k8s/upstream_shield.csv").read())
    fid = pd.read_csv("results/k8s/fidelity.csv")
    f.write(f"fidelity: {int((fid.model == fid.actual).sum())}/{len(fid)} (allowed {int(fid.model.sum())}, denied {int((1-fid.model).sum())})\n")
print(open("results/k8s/summary.txt").read())
