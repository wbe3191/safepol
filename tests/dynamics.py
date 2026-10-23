"""Sanity check of the workload dynamics: a flash crowd must hurt a fixed rate limit, a legitimate
privileged job must be blocked by a tight seccomp profile, and lifting either must fix it."""
from safepol.sim.action_space import Action, K_IDX
from safepol.sim.replay_env import EnvConfig, ReplayEnv

env = ReplayEnv(EnvConfig(seed=1, shield=True))
V = env.topo.V
env.reset(seed=3, options={"scenario": None, "init": "allowlist", "flash": (10, 18, 5.0), "job": (env.topo.index["recommendationservice"], 30, 38)})
fe, rs = env.frontend, env.topo.index["recommendationservice"]
log = []
for t in range(45):
    if t == 0:
        a = Action(K_IDX["RATE_LIMIT"], fe, V, 0)
    elif t == 1:
        a = Action(K_IDX["TIGHTEN_SC"], rs, V, 0)
    elif t == 14:
        a = Action(K_IDX["REMOVE"], fe, V, 0)          # lift the limit mid-flash
    elif t == 34:
        a = Action(K_IDX["RELAX_SC"], rs, V, 0)        # relax mid-job
    else:
        a = Action(K_IDX["NOOP"], 0, V, 0)
    obs, r, done, _, info = env.step(a)
    e = env.episode_log[-1]
    log.append((t, e["action"], e["flash"], e["job"], e["legit"], e["legit_drop"], e["legit_host"], e["legit_host_drop"], e["slo"], round(r, 2), e["cost"]))
print("t action flash job legit legit_drop lh lh_drop slo r cost")
for row in log:
    if row[2] or row[3] or row[0] < 3:
        print(*row)
s = env.summary()
print({k: s[k] for k in ("FBR", "FBR_host", "SLO_viol_rate", "flash", "job", "actions")})
assert any(r[5] > 0 for r in log if r[2] and r[0] < 14), "fixed limit should drop legit flash traffic"
assert all(r[5] == 0 for r in log if r[2] and r[0] >= 15), "lifting the limit should stop the drops"
assert any(r[7] > 0 for r in log if r[3] and r[0] < 34), "tight seccomp should block the job"
assert all(r[7] == 0 for r in log if r[3] and r[0] >= 35), "relaxing should unblock the job"
print("dynamics ok")
