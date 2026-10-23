"""Scripted check: in S1/S7, TIGHTEN_SC / DROP_CAP on the alerted victim must close the objective."""
import numpy as np
from safepol.experiments.methods import METHODS, make_env_config
from safepol.sim.replay_env import ReplayEnv, FEATURES
from safepol.sim.action_space import Action, K_IDX

env = ReplayEnv(make_env_config(METHODS["SAFEPOL"], 0))
for sid, kind in (("S1", "TIGHTEN_SC"), ("S7", "DROP_CAP"), ("S7", "TIGHTEN_SC")):
    obs, _ = env.reset(options={"scenario": sid, "start": 10, "init": "minimal", "job": None, "flash": None})
    V = env.topo.V
    done = False; acted = False; t = 0; hits = 0; rew = 0
    while not done:
        a = Action(K_IDX["NOOP"], 0, V, 0)
        al = env.X[:, FEATURES.index("alerts")]
        if not acted and al.max() > 0:
            w = int(al.argmax()); a = Action(K_IDX[kind], w, V, 0); acted = True
            print(f"{sid}: alert on {env.topo.names[w]} at t={t} (victim={env.topo.names[env.attack.victim]}), apply {kind}; feasible={env.mask.k_mask()[K_IDX[kind]] if hasattr(env,'mask') else '?'}")
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc; t += 1
        e = env.episode_log[-1]
        hits += int(e["goal"]); rew += r
        if 9 <= t <= 16:
            print(f"  t={t} action={e['action']} interv={e['intervened']} viol={e.get('cand_violation')} stage={e['stage']} goal={e['goal']} mal={e['mal']} drop={e['mal_drop']} r={r:.2f}")
    s = env.summary()
    print(f"  => ASR={s['ASR']} TTM={s['TTM']} goal_hits={hits} return={rew:.1f}  syscalls_danger_left={len(env.P.endpoint[env.attack.victim].syscalls & set(['mount','unshare','setns','keyctl']))}")
