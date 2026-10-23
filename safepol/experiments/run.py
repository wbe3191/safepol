"""Train one learned method for one seed; evaluate any method on the fixed replay set.

Usage:
  python -m safepol.experiments.run train  --method SAFEPOL --seed 0 --steps 150000
  python -m safepol.experiments.run eval   --method SAFEPOL --seed 0
  python -m safepol.experiments.run eval   --method B1-static --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from safepol.agent.dqn import DQNAgent, DQNConfig
from safepol.agent.ppo_lagrangian import PPOConfig, PPOLagrangian
from safepol.baselines import static as static_baselines
from safepol.bench.attacks import ATTACK_IDS
from safepol.experiments.methods import METHODS, FULL_SCENARIOS, make_env_config
from safepol.sim.curriculum import Curriculum
from safepol.sim.replay_env import ReplayEnv

RESULTS = os.environ.get("SAFEPOL_RESULTS", "results")


def replay_set(n_per_scenario: int = 20, n_benign: int = 20, n_drift: int = 24, master_seed: int = 12345) -> List[Dict]:
    """Fixed, method-independent list of evaluation episodes (paired design).
    Every replay is evaluated from both starting policies (minimal P0 and the operator allow-list)."""
    rng = random.Random(master_seed)
    base = []
    for sid in ATTACK_IDS:
        for i in range(n_per_scenario):
            base.append({"scenario": sid, "start": rng.randint(8, 30), "env_seed": rng.randint(0, 10**6), "drift": False, "group": "attack"})
    for i in range(n_benign):
        base.append({"scenario": None, "start": 0, "env_seed": rng.randint(0, 10**6), "drift": False, "group": "benign"})
    for i in range(n_drift):
        sid = ATTACK_IDS[i % len(ATTACK_IDS)]
        base.append({"scenario": sid, "start": rng.randint(8, 30), "env_seed": rng.randint(0, 10**6), "drift": True, "group": "drift"})
    reps = []
    for init in ("minimal", "allowlist"):
        for b in base:
            r = dict(b); r["init"] = init; reps.append(r)
    return reps


def build(method: str, seed: int, steps: int = 150_000, env_extra=None):
    spec = METHODS[method]
    env = ReplayEnv(make_env_config(spec, seed, **(env_extra or {})))
    # optional hyper-parameter overrides for sweeps: SAFEPOL_AGENT_OVERRIDES='{"ent_coef":0.0}'
    import json
    agent_over = json.loads(os.environ.get("SAFEPOL_AGENT_OVERRIDES", "{}"))
    if spec.kind == "ppo":
        cur = Curriculum(enabled=spec.curriculum)
        agent = PPOLagrangian(env, PPOConfig(total_steps=steps, seed=seed, **{**spec.agent, **agent_over}), curriculum=cur)
    elif spec.kind == "dqn":
        cur = Curriculum(enabled=False)
        agent = DQNAgent(env, DQNConfig(total_steps=steps, seed=seed, **spec.agent))
    else:
        cur = Curriculum(enabled=False)
        agent = getattr(static_baselines, spec.controller)()
    return spec, env, agent, cur


def train(method: str, seed: int, steps: int):
    spec, env, agent, cur = build(method, seed, steps)
    os.makedirs(f"{RESULTS}/models", exist_ok=True); os.makedirs(f"{RESULTS}/train", exist_ok=True)
    log_path = f"{RESULTS}/train/{method}_s{seed}.csv"
    rows = []

    def cb(rec):
        rows.append(rec)
        if len(rows) % 5 == 0 or len(rows) == 1:
            print(f"[{method} s{seed}] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items() if k in ("steps", "ep_return", "ep_cost", "ASR", "FBR", "interv", "lambda", "mean_cost", "inv_viol", "elapsed")), flush=True)
            pd.DataFrame(rows).to_csv(log_path, index=False)

    if spec.kind == "static":
        print("static method, nothing to train"); return
    cur.apply(env, 0.0)
    if spec.kind == "ppo" and getattr(agent.cfg, "warm_start_steps", 0) > 0:
        cur.apply(env, 0.55)   # demonstrations are collected on the full (non-drift) scenario mix
        t0 = time.time()
        agent.warm_start(static_baselines.B3RuleResponder(), agent.cfg.warm_start_steps, agent.cfg.warm_start_epochs,
                         callback=lambda rec: print(f"[{method} s{seed}] warm-start " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True))
        print(f"[{method} s{seed}] warm-start done in {time.time()-t0:.0f}s", flush=True)
        cur.apply(env, 0.0)
    agent.train(cb)
    pd.DataFrame(rows).to_csv(log_path, index=False)
    agent.save(f"{RESULTS}/models/{method}_s{seed}.pt")
    pd.DataFrame(agent.episodes).to_csv(f"{RESULTS}/train/{method}_s{seed}_episodes.csv", index=False)


def evaluate(method: str, seed: int, replays=None, tag: str = "eval"):
    spec, env, agent, cur = build(method, seed)
    cur.apply(env, 1.0)   # final (hardest) environment for every method
    env.cfg.scenarios = list(FULL_SCENARIOS)
    if spec.kind != "static":
        agent.load(f"{RESULTS}/models/{method}_s{seed}.pt")
    replays = replays or replay_set()
    rows, step_rows = [], []
    t0 = time.time()
    for i, rep in enumerate(replays):
        obs, info = env.reset(seed=rep["env_seed"], options={"scenario": rep["scenario"], "start": rep["start"],
                                                             "drift": rep["drift"], "init": rep["init"]})
        if spec.kind == "static":
            agent.reset(env)
        done = False
        while not done:
            if spec.kind == "ppo":
                a, _, _ = agent.act(obs, env.masks(), deterministic=True)
            elif spec.kind == "dqn":
                a = agent.act(obs, deterministic=True)
            else:
                a = agent.act(env, obs)
            obs, r, done, _, info = env.step(a)
        s = env.summary()
        s.update({"method": method, "seed": seed, "replay": i, "group": rep["group"], "start": rep["start"],
                  "hops": getattr(env.attack, "hops", 0) if env.attack else 0})
        rows.append(s)
        for e in env.episode_log:
            e2 = dict(e); e2.update({"method": method, "seed": seed, "replay": i}); step_rows.append(e2)
    os.makedirs(f"{RESULTS}/{tag}", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/{tag}/{method}_s{seed}.csv", index=False)
    pd.DataFrame(step_rows).to_csv(f"{RESULTS}/{tag}/{method}_s{seed}_steps.csv.gz", index=False, compression="gzip")
    att = df[df.group == "attack"]
    print(f"[{method} s{seed}] {len(df)} episodes in {time.time()-t0:.0f}s | ASR={att.ASR.mean():.3f} "
          f"TTM={att.TTM.dropna().mean():.1f}s mitigated={att.mitigated.mean():.2f} FBR={df.FBR.mean():.4f} "
          f"SLO={df.SLO_viol_rate.mean():.3f} inv_viol={df.inv_violations.sum()} interv={df.intervention_rate.mean():.3f}", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.method, args.seed, args.steps)
    else:
        evaluate(args.method, args.seed, tag=args.tag)
