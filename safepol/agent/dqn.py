"""DQN baseline (B4): re-implementation of the prior ASEJ'26 design — a three-layer MLP
Q-network over a flat action space of predefined policy "bundles" (action type x workload),
experience replay, target network, epsilon-greedy; conflicts resolved heuristically by PSO
(env.cfg.conflict_resolver = "pso"); no invariants, no shield."""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safepol.sim.action_space import ACTION_TYPES, Action
from safepol.sim.replay_env import N_MAX, ReplayEnv

NK = len(ACTION_TYPES)
NA = NK * N_MAX


@dataclass
class DQNConfig:
    total_steps: int = 150_000
    buffer: int = 50_000
    batch: int = 128
    lr: float = 5e-4
    gamma: float = 0.98
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 60_000
    target_update: int = 1000
    train_every: int = 4
    warmup: int = 2000
    seed: int = 0


class QNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, NA))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, env: ReplayEnv, cfg: DQNConfig):
        self.env, self.cfg = env, cfg
        torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
        self.q = QNet(env.obs_dim); self.qt = QNet(env.obs_dim); self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buf = deque(maxlen=cfg.buffer)
        self.history: List[Dict] = []
        self.steps = 0

    def _decode(self, idx: int) -> Action:
        k, w = divmod(int(idx), N_MAX)
        w = min(w, self.env.topo.n - 1)
        return Action(k, w, self.env.topo.V, 0)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Action:
        eps = 0.0 if deterministic else max(self.cfg.eps_end, self.cfg.eps_start - (self.cfg.eps_start - self.cfg.eps_end) * self.steps / self.cfg.eps_decay_steps)
        if random.random() < eps:
            return self._decode(random.randrange(NK * self.env.topo.n))
        with torch.no_grad():
            qv = self.q(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).view(NK, N_MAX)
            qv[:, self.env.topo.n:] = -1e9
            return self._decode(int(qv.view(-1).argmax()))

    def train(self, callback=None):
        cfg, env = self.cfg, self.env
        obs, _ = env.reset(seed=cfg.seed)
        ep_ret, ep_stats, t0 = 0.0, [], time.time()
        while self.steps < cfg.total_steps:
            a = self.act(obs)
            nobs, r, done, _, info = env.step(a)
            self.buf.append((obs, a.k * N_MAX + a.w, r, nobs, float(done)))
            ep_ret += r; obs = nobs; self.steps += 1
            if done:
                s = info["episode"]; s.update({"ep_return": ep_ret, "steps": self.steps}); ep_stats.append(s)
                obs, _ = env.reset(); ep_ret = 0.0
            if self.steps > cfg.warmup and self.steps % cfg.train_every == 0:
                batch = random.sample(self.buf, cfg.batch)
                O = torch.as_tensor(np.array([b[0] for b in batch])); A = torch.as_tensor([b[1] for b in batch])
                R = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
                O2 = torch.as_tensor(np.array([b[3] for b in batch])); Dn = torch.as_tensor([b[4] for b in batch])
                qsa = self.q(O).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target = R + cfg.gamma * (1 - Dn) * self.qt(O2).max(1).values
                loss = F.smooth_l1_loss(qsa, target)
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
            if self.steps % cfg.target_update == 0:
                self.qt.load_state_dict(self.q.state_dict())
            if self.steps % 2048 == 0:
                recent = ep_stats[-10:]
                rec = {"steps": self.steps, "ep_return": float(np.mean([e["ep_return"] for e in recent])) if recent else float("nan"),
                       "ASR": float(np.nanmean([e["ASR"] for e in recent if e["ASR"] is not None])) if recent else float("nan"),
                       "FBR": float(np.mean([e["FBR"] for e in recent])) if recent else float("nan"),
                       "inv_viol": float(np.mean([e["inv_violations"] for e in recent])) if recent else float("nan"),
                       "elapsed": time.time() - t0}
                self.history.append(rec)
                if callback:
                    callback(rec)
        self.episodes = ep_stats
        return self.history

    def save(self, path):
        torch.save({"q": self.q.state_dict()}, path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location="cpu")["q"]); self.qt.load_state_dict(self.q.state_dict())
