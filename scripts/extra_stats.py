import glob, re, sys, numpy as np, pandas as pd
d = "results/evaltest"
fr = [pd.read_csv(f) for f in sorted(glob.glob(f"{d}/*_s*.csv")) if re.search(r"_s\d+\.csv$", f)]
df = pd.concat(fr, ignore_index=True)
df["dyn"] = np.select([df.flash & df.job, df.flash, df.job], ["flash+job", "flash", "job"], "stationary")
att = df[df.group == "attack"]
print("== SAFEPOL per-scenario ASR / TTM / mitigated ==")
s = att[att.method == "SAFEPOL"].groupby("scenario").agg(ASR=("ASR", "mean"), TTM=("TTM", "mean"), mit=("mitigated", "mean"), FBR=("FBR", "mean"))
print(s.round(3))
print("== SAFEPOL S1/S7 ASR by init ==")
print(att[(att.method == "SAFEPOL") & att.scenario.isin(["S1", "S7"])].groupby(["scenario", "init"]).ASR.mean().round(3))
print("== seeds: SAFEPOL ASR / SLO ==")
print(df[df.method == "SAFEPOL"].groupby("seed").agg(ASR=("ASR", lambda x: x[df.loc[x.index, "group"] != "benign"].mean()), SLO=("SLO_viol_rate", "mean")).round(3))
print("== ASR of ablations per scenario ==")
ab = [m for m in df.method.unique() if m.startswith("A-")] + ["SAFEPOL"]
print(att[att.method.isin(ab)].pivot_table(index="scenario", columns="method", values="ASR").round(2).to_string())
print("== S8 adaptive: ASR, TTM ==")
print(att[att.scenario == "S8"].groupby("method").agg(ASR=("ASR", "mean"), TTM=("TTM", "mean")).round(3).to_string())
print("== drift group: ASR, SLO, FBR ==")
print(df[df.group == "drift"].groupby("method").agg(ASR=("ASR", "mean"), SLO=("SLO_viol_rate", "mean"), FBR=("FBR", "mean"), inv=("inv_violations", "mean")).round(3).to_string())
print("== benign group: SLO, FBR, actions ==")
print(df[df.group == "benign"].groupby("method").agg(SLO=("SLO_viol_rate", "mean"), FBR=("FBR", "mean"), actions=("actions", "mean"), rules=("rules_final", "mean") if "rules_final" in df else ("actions", "mean")).round(3).to_string())
print("== columns ==", list(df.columns))
print("== mean intervention/ep and AS_final, rules_final for shielded ==")
cols = [c for c in ["intervention_rate", "AS_final", "rules_final", "BR"] if c in df.columns]
print(df.groupby("method")[cols].mean().round(3).to_string())
print("== SLO by dyn for ablations ==")
print(df[df.method.isin(ab)].pivot_table(index="method", columns="dyn", values="SLO_viol_rate").round(3).to_string())
