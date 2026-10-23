"""Figures for the paper (matplotlib, PDF).  usage: PYTHONPATH=. python scripts/figures.py results/evaltest"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "results/evaltest"
FIG = "paper/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "pdf.fonttype": 42})

frames = [pd.read_csv(f) for f in sorted(glob.glob(f"{d}/*_s*.csv")) if re.search(r"_s\d+\.csv$", f)]
df = pd.concat(frames, ignore_index=True)
df["dyn"] = np.select([df.flash & df.job, df.flash, df.job], ["flash+job", "flash", "job"], "stationary")
MAIN = ["B1-static", "B2-observed", "B6-shield-only", "B7-hardened", "B3-responder", "B8-shielded-runbook", "B4-DQN-PSO", "B5-PPO", "SAFEPOL", "SAFEPOL-H"]
MAIN = [m for m in MAIN if m in set(df.method)]
COL = {m: c for m, c in zip(MAIN, plt.cm.tab10(np.linspace(0, 1, 10)))}
COL["SAFEPOL"] = "black"; COL["SAFEPOL-H"] = "dimgray"; COL["B8-shielded-runbook"] = "tab:red"

# Fig 1: security/availability trade-off scatter
fig, ax = plt.subplots(figsize=(3.4, 2.6))
att = df[df.group != "benign"]
for m in MAIN:
    x = att[att.method == m].ASR.mean() * 100
    y = df[df.method == m].SLO_viol_rate.mean() * 100
    ax.scatter(x, y, color=COL[m], s=28, marker="*" if m.startswith("SAFEPOL") else "o", zorder=3)
    ax.annotate(m, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("attack success rate (%)"); ax.set_ylabel("SLO-violating periods (%)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/tradeoff.pdf"); fig.savefig(f"{FIG}/tradeoff.png", dpi=130); plt.close(fig)

# Fig 2: ASR by scenario
scn = sorted(att[att.group == "attack"].scenario.unique())
fig, ax = plt.subplots(figsize=(7, 2.3))
w = 0.8 / len(MAIN)
for i, m in enumerate(MAIN):
    v = [att[(att.method == m) & (att.scenario == s)].ASR.mean() * 100 for s in scn]
    ax.bar(np.arange(len(scn)) + i * w, v, w, label=m, color=COL[m])
ax.set_xticks(np.arange(len(scn)) + 0.4 - w / 2); ax.set_xticklabels(scn)
ax.set_ylabel("ASR (%)"); ax.legend(ncol=4, frameon=False); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/asr_by_scenario.pdf"); fig.savefig(f"{FIG}/asr_by_scenario.png", dpi=130); plt.close(fig)

# Fig 3: availability under workload dynamics
dyns = ["stationary", "flash", "job", "flash+job"]
fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
for ax, met, lab in zip(axes, ["SLO_viol_rate", "FBR_host"], ["SLO-violating periods (%)", "legit. privileged-job syscalls blocked (%)"]):
    for i, m in enumerate(MAIN):
        v = [df[(df.method == m) & (df.dyn == dd)][met].mean() * 100 for dd in dyns]
        ax.bar(np.arange(len(dyns)) + i * w, v, w, label=m, color=COL[m])
    ax.set_xticks(np.arange(len(dyns)) + 0.4 - w / 2); ax.set_xticklabels(dyns)
    ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.3)
axes[0].legend(ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/dynamics.pdf"); fig.savefig(f"{FIG}/dynamics.png", dpi=130); plt.close(fig)

# Fig 4: training curves (SAFEPOL and learned baselines)
tr = []
for f in glob.glob("results/train/*_s*.csv"):
    if re.search(r"_s\d+\.csv$", f):
        t = pd.read_csv(f); t["method"] = os.path.basename(f).rsplit("_s", 1)[0]; tr.append(t)
if tr:
    tr = pd.concat(tr)
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.1))
    for m in ["SAFEPOL", "B5-PPO", "A-noCost", "A-noShield"]:
        g = tr[tr.method == m]
        if g.empty:
            continue
        for ax, met in zip(axes, ["ep_return", "ep_cost", "lambda"]):
            if met not in g:
                continue
            mu = g.groupby("steps")[met].mean(); sd = g.groupby("steps")[met].std().fillna(0)
            ax.plot(mu.index / 1000, mu.values, label=m, color=COL.get(m, None))
            ax.fill_between(mu.index / 1000, (mu - sd).values, (mu + sd).values, alpha=0.2, color=COL.get(m, None))
            ax.set_xlabel("training steps (k)"); ax.set_title(met.replace("_", " "))
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/training.pdf"); fig.savefig(f"{FIG}/training.png", dpi=130); plt.close(fig)

# Fig 5: scalability
if os.path.exists("results/scalability.csv"):
    sc = pd.read_csv("results/scalability.csv")
    med = sc.groupby(["component", "n"]).latency_ms.median().unstack(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for comp, lab, st in [("repair_maxsat_local", "MaxSAT repair (localised)", "-o"), ("repair_maxsat_global", "MaxSAT repair (global)", "--o"),
                          ("repair_greedy", "greedy repair", "-s"), ("repair_pso", "PSO repair", "--^"),
                          ("invariant_check", "invariant check", "-d"), ("reach_incremental", "incremental reach.", "-v"),
                          ("reach_full", "full reach.", "--v"), ("conflicts_local", "conflicts (localised)", "-x")]:
        if comp in med:
            ax.plot(med.index, med[comp], st, label=lab, ms=3)
    ax.axhline(200, color="gray", lw=0.6, ls=":"); ax.text(110, 230, "200 ms budget", fontsize=6, color="gray")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("workloads n"); ax.set_ylabel("median latency (ms)")
    ax.legend(frameon=False, fontsize=5.5); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{FIG}/scalability.pdf"); fig.savefig(f"{FIG}/scalability.png", dpi=130); plt.close(fig)
print("figures written to", FIG)
