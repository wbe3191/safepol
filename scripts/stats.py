"""Statistical analysis of the evaluation CSVs.

Unit of analysis for inferential tests: the replay episode (the paired evaluation set is identical
for every method), averaged over seeds for learned methods.  For every metric and every baseline we
report mean 鍗?std over seeds, a 95% bootstrap CI (10k resamples of episodes), a paired Wilcoxon
signed-rank test SAFEPOL vs. baseline with Holm correction over all baselines, and Cliff's delta.

usage: PYTHONPATH=. python scripts/stats.py results/evaltest
writes results/tables/{main,by_scenario,by_dyn,tests,ablation,seeds}.csv and .tex
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

d = sys.argv[1] if len(sys.argv) > 1 else "results/evaltest"
OUT = "results/tables"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

frames = [pd.read_csv(f) for f in sorted(glob.glob(f"{d}/*_s*.csv")) if re.search(r"_s\d+\.csv$", f)]
df = pd.concat(frames, ignore_index=True)
df["dyn"] = np.select([df.flash & df.job, df.flash, df.job], ["flash+job", "flash", "job"], "stationary")
df["mitigated"] = df.mitigated.astype(float)
METRICS = ["ASR", "mitigated", "TTM", "FBR", "FBR_host", "SLO_viol_rate", "inv_violations", "intervention_rate", "actions", "return"]
LOWER_BETTER = {"ASR", "TTM", "FBR", "FBR_host", "SLO_viol_rate", "inv_violations", "intervention_rate", "actions"}
ORDER = ["B0-none", "B1-static", "B2-observed", "B6-shield-only", "B7-hardened", "B3-responder", "B8-shielded-runbook",
         "B4-DQN-PSO", "B5-PPO", "SAFEPOL", "SAFEPOL-H", "A-noShield", "A-noCost", "A-noMask", "A-flat", "A-noGraph", "A-noCurriculum",
         "A-PSOrepair", "A-greedyRepair", "A-detector"]
methods = [m for m in ORDER if m in set(df.method)] + sorted(set(df.method) - set(ORDER))


def per_seed(sub):
    """One row per (method, seed): episode means (ASR/TTM over attack+drift episodes)."""
    att = sub[sub.group != "benign"]
    a = att.groupby(["method", "seed"]).agg(ASR=("ASR", "mean"), mitigated=("mitigated", "mean"), TTM=("TTM", "mean"))
    b = sub.groupby(["method", "seed"]).agg(FBR=("FBR", "mean"), FBR_host=("FBR_host", "mean"),
                                            SLO_viol_rate=("SLO_viol_rate", "mean"), inv_violations=("inv_violations", "sum"),
                                            intervention_rate=("intervention_rate", "mean"), actions=("actions", "mean"),
                                            **{"return": ("return", "mean")})
    return a.join(b, how="outer").reset_index()


def episode_matrix(sub, metric):
    """replay x method matrix of the metric, averaged over seeds (paired design)."""
    s = sub if metric not in ("ASR", "TTM", "mitigated") else sub[sub.group != "benign"]
    return s.pivot_table(index="replay", columns="method", values=metric, aggfunc="mean")


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b); a, b = a[ok], b[ok]
    if len(a) == 0:
        return np.nan
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


# ---- main table: mean 鍗?std over seeds + bootstrap CI over episodes --------------------------
ps = per_seed(df)
ps.to_csv(f"{OUT}/seeds.csv", index=False)
rows = []
for m in methods:
    r = {"method": m, "seeds": int((ps.method == m).sum())}
    for met in METRICS:
        v = ps.loc[ps.method == m, met]
        lo, hi = boot_ci(episode_matrix(df, met)[m]) if m in episode_matrix(df, met) else (np.nan, np.nan)
        r[met] = v.mean(); r[met + "_std"] = v.std(ddof=0) if len(v) > 1 else 0.0
        r[met + "_lo"], r[met + "_hi"] = lo, hi
    rows.append(r)
main = pd.DataFrame(rows).set_index("method")
main.to_csv(f"{OUT}/main.csv")
show = ["seeds"] + METRICS
print("=== main (mean over seeds) ===")
print(main[show].round(4).to_string())

# ---- paired tests SAFEPOL vs every other method ------------------------------------------------
tests = []
if "SAFEPOL" in methods:
    for met in ["ASR", "TTM", "FBR", "FBR_host", "SLO_viol_rate", "inv_violations", "actions", "return"]:
        M = episode_matrix(df, met)
        for m in methods:
            if m == "SAFEPOL" or m not in M:
                continue
            a, b = M["SAFEPOL"].values, M[m].values
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() < 10 or np.allclose(a[ok], b[ok]):
                p = 1.0
            else:
                try:
                    p = sps.wilcoxon(a[ok], b[ok], zero_method="zsplit").pvalue
                except ValueError:
                    p = 1.0
            tests.append(dict(metric=met, baseline=m, n=int(ok.sum()), safepol=np.nanmean(a[ok]), baseline_mean=np.nanmean(b[ok]),
                              p_raw=p, cliffs_delta=cliffs(a[ok], b[ok])))
    T = pd.DataFrame(tests)
    # Holm correction within each metric over the baselines
    T["p_holm"] = np.nan
    for met, g in T.groupby("metric"):
        p = g.p_raw.values; order = np.argsort(p); k = len(p)
        adj = np.minimum.accumulate(np.minimum(1, (k - np.arange(k)) * p[order])[::-1])[::-1]
        T.loc[g.index[order], "p_holm"] = adj
    T["sig"] = np.where(T.p_holm < 0.01, "**", np.where(T.p_holm < 0.05, "*", ""))
    T.to_csv(f"{OUT}/tests.csv", index=False)
    print("\n=== SAFEPOL vs baselines: paired Wilcoxon (Holm) + Cliff's delta (negative = SAFEPOL lower) ===")
    print(T.round(4).to_string(index=False))

# ---- slices ----------------------------------------------------------------------------------
att = df[df.group == "attack"]
by_scn = att.pivot_table(index="scenario", columns="method", values="ASR", aggfunc="mean").reindex(columns=methods)
by_scn.to_csv(f"{OUT}/by_scenario_ASR.csv")
ttm_scn = att.pivot_table(index="scenario", columns="method", values="TTM", aggfunc="mean").reindex(columns=methods)
ttm_scn.to_csv(f"{OUT}/by_scenario_TTM.csv")
by_dyn = df.groupby(["method", "dyn"]).agg(FBR=("FBR", "mean"), FBR_host=("FBR_host", "mean"), SLO=("SLO_viol_rate", "mean"),
                                           ASR=("ASR", "mean"), n=("FBR", "size")).reset_index()
by_dyn.to_csv(f"{OUT}/by_dyn.csv", index=False)
by_init = df.groupby(["method", "init"]).agg(ASR=("ASR", "mean"), FBR=("FBR", "mean"), SLO=("SLO_viol_rate", "mean"),
                                             inv=("inv_violations", "sum")).reset_index()
by_init.to_csv(f"{OUT}/by_init.csv", index=False)
by_group = df.groupby(["method", "group"]).agg(ASR=("ASR", "mean"), FBR=("FBR", "mean"), SLO=("SLO_viol_rate", "mean"),
                                               inv=("inv_violations", "sum"), actions=("actions", "mean")).reset_index()
by_group.to_csv(f"{OUT}/by_group.csv", index=False)
print("\n=== ASR by scenario ===\n", by_scn.round(3).to_string())
print("\n=== by dynamics ===\n", by_dyn.pivot(index="method", columns="dyn", values="SLO").reindex(methods).round(4).to_string())
print("\n=== by group ===\n", by_group.pivot(index="method", columns="group", values=["ASR", "SLO", "inv"]).reindex(methods).round(4).to_string())

# ---- LaTeX main table ----------------------------------------------------------------------
def fmt(m, met, pct=False, intg=False, nd=3):
    v, s = main.loc[m, met], main.loc[m, met + "_std"]
    if np.isnan(v):
        return "--"
    if intg:
        return f"{v:.0f}"
    if pct:
        return f"{100*v:.1f}" + (f"$\\pm${100*s:.1f}" if s > 0 else "")
    return f"{v:.{nd}f}" + (f"$\\pm${s:.{nd}f}" if s > 0 else "")

lines = []
for m in methods:
    if m.startswith("A-"):
        continue
    lines.append(" & ".join([m.replace("_", "\\_"), fmt(m, "ASR", pct=True), fmt(m, "mitigated", pct=True),
                             fmt(m, "TTM", nd=1), fmt(m, "FBR", pct=True), fmt(m, "FBR_host", pct=True), fmt(m, "SLO_viol_rate", pct=True),
                             fmt(m, "inv_violations", intg=True), fmt(m, "intervention_rate", pct=True), fmt(m, "actions", nd=0)]) + " \\\\")
open(f"{OUT}/main.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")
lines = []
for m in methods:
    if not (m.startswith("A-") or m == "SAFEPOL"):
        continue
    lines.append(" & ".join([m.replace("_", "\\_"), fmt(m, "ASR", pct=True), fmt(m, "mitigated", pct=True), fmt(m, "TTM", nd=1),
                             fmt(m, "FBR", pct=True), fmt(m, "SLO_viol_rate", pct=True), fmt(m, "inv_violations", intg=True),
                             fmt(m, "intervention_rate", pct=True), fmt(m, "return", nd=0)]) + " \\\\")
open(f"{OUT}/ablation.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")
lines = []
for s in by_scn.index:
    lines.append(" & ".join([s] + [("--" if np.isnan(by_scn.loc[s, m]) else f"{100*by_scn.loc[s, m]:.0f}") for m in methods if not m.startswith("A-")]) + " \\\\")
open(f"{OUT}/by_scenario.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")
# dynamics table: SLO (and FBR_host) per slice, plus drift-group SLO, for the non-ablation methods
pv = by_dyn.pivot(index="method", columns="dyn", values="SLO")
pvh = by_dyn.pivot(index="method", columns="dyn", values="FBR_host")
pg = by_group.pivot(index="method", columns="group", values="SLO")
lines = []
for m in methods:
    if m.startswith("A-"):
        continue
    cells = [m]
    for d in ("stationary", "flash", "job", "flash+job"):
        cells.append(f"{100*pv.loc[m, d]:.1f}" if d in pv.columns and not np.isnan(pv.loc[m, d]) else "--")
    cells.append(f"{100*pvh.loc[m, 'job']:.0f}" if "job" in pvh.columns else "--")
    cells.append(f"{100*pg.loc[m, 'drift']:.1f}" if "drift" in pg.columns else "--")
    lines.append(" & ".join(cells) + " \\\\")
open(f"{OUT}/by_dyn.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")
# significance table: SAFEPOL vs each baseline, ASR / SLO / FBR
try:
    T = pd.read_csv(f"{OUT}/tests.csv")
    lines = []
    for b in [m for m in methods if not m.startswith("A-") and m != "SAFEPOL"]:
        cells = [b]
        for met in ("ASR", "SLO_viol_rate", "FBR"):
            r = T[(T.metric == met) & (T.baseline == b)]
            if len(r) == 0:
                cells.append("--"); continue
            r = r.iloc[0]
            p = r.p_holm
            ps = "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")
            cells.append(f"{ps} ({r.cliffs_delta:+.2f})")
        lines.append(" & ".join(cells) + " \\\\")
    open(f"{OUT}/tests.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")
except Exception as e:
    print("tests.tex skipped:", e)
# ---- shielding effect per controller family ------------------------------------------------
pairs = [("Observation-driven allow-list", "B2-observed", "B6-shield-only"),
         ("Response runbook", "B3-responder", "B8-shielded-runbook"),
         ("Constrained PPO learner", "A-noShield", "SAFEPOL")]
lines = []
pc = lambda m, met: f"{100*main.loc[m, met]:.1f}"
for name, u, s in pairs:
    if u in main.index and s in main.index:
        lines.append(" & ".join([name, pc(u, "ASR"), pc(s, "ASR"), pc(u, "SLO_viol_rate"), pc(s, "SLO_viol_rate"),
                                 pc(u, "FBR"), pc(s, "FBR"), fmt(u, "inv_violations", intg=True), fmt(s, "inv_violations", intg=True)]) + " \\\\")
open(f"{OUT}/shield_effect.tex", "w").write("\n".join(lines) + "\n\\bottomrule\n")
# mirror the LaTeX tables into the paper source tree
import shutil
os.makedirs("paper/tables", exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith(".tex"):
        shutil.copy(os.path.join(OUT, f), os.path.join("paper/tables", f))
print("\nwrote", OUT)
