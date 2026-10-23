"""PPO sanity check on a synthetic objective: reward = 1 if the agent picks NOOP else 0.
If the machinery works, p(NOOP) must go to ~1 and the entropy must fall within a few iterations."""
import numpy as np, torch
from safepol.experiments.run import build
from safepol.sim.action_space import K_IDX

spec, env, agent, cur = build("SAFEPOL", 0, steps=16000)
cur.apply(env, 0.0)
orig_step = env.step


cnt = {"n": 0, "noop": 0}


def fake_step(a):
    obs, r, term, trunc, info = orig_step(a)
    r = 1.0 if a.kind == "NOOP" else 0.0
    cnt["n"] += 1; cnt["noop"] += int(a.kind == "NOOP")
    info["cost"] = 0.0
    return obs, r, term, trunc, info


env.step = fake_step
agent.cfg.lagrangian = False; agent.lmbda = 0.0


def cb(rec):
    print({k: round(v, 3) for k, v in rec.items() if k in ("steps", "ent", "kl", "pg", "gnorm_a")}, "p_noop=%.3f" % (cnt["noop"] / max(1, cnt["n"])))
    cnt["n"] = cnt["noop"] = 0


hist = agent.train(cb)
