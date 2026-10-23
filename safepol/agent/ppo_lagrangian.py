"""PPO-Lagrangian with factorized, maskable action heads (Sec. 1.5 / 2.4).

Policy:  pi(a|s) = pi(k|s) pi(w|s,k) pi(p|s,k,w) pi(pi|s,k,w,p), each head masked by the Shield's
monotone masks.  Two critics (reward, cost).  Lagrange multiplier lambda >= 0 updated by
projected gradient ascent on (J_c - delta).  Setting `lagrangian=False` gives plain PPO
(baseline B5 / ablation "-cost")."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safepol.sim.action_space import ACTION_TYPES, Action
from safepol.sim.replay_env import N_MAX, V_MAX, D, DG, ReplayEnv

NK, NW, NP, NPI = len(ACTION_TYPES), N_MAX, V_MAX + 1, 3
D_ENT = D + DG + 3          # per-workload entity row: features, graph features, policy-embedding degrees

# Which sub-decisions matter for each action type.  Irrelevant heads are excluded from the joint
# log-probability and entropy (a parameterised-action-space convention); otherwise the PPO ratio
# is dominated by noise from choices that do not influence the outcome.
_NET = {"DENY_IN", "DENY_OUT", "ALLOW_IN", "ALLOW_OUT"}
REL_W = torch.tensor([ACTION_TYPES[k] not in ("NOOP", "ROLLBACK") for k in range(NK)], dtype=torch.float32)
REL_P = torch.tensor([ACTION_TYPES[k] in (_NET | {"REMOVE"}) for k in range(NK)], dtype=torch.float32)
REL_PI = torch.tensor([ACTION_TYPES[k] in _NET for k in range(NK)], dtype=torch.float32)


def entity_rows(obs: torch.Tensor) -> torch.Tensor:
    """(B, obs_dim) -> (B, N_MAX, D_ENT): the per-workload slices of the flat observation."""
    B = obs.shape[0]
    X = obs[:, :N_MAX * D].view(B, N_MAX, D)
    G = obs[:, N_MAX * D:N_MAX * (D + DG)].view(B, N_MAX, DG)
    hp = obs[:, N_MAX * (D + DG) + 7:]
    H = hp.view(B, 3, N_MAX).transpose(1, 2)
    return torch.cat([X, G, H], -1)


class FactorizedPolicy(nn.Module):
    """pi(k|s) pi(w|s,k) pi(p|s,k,w) pi(pi|s,k,w,p).  The subject head is entity-wise (pointer-style):
    every workload is scored from its own feature row conditioned on the context and the action type,
    so 'act on the workload that raised the alert' is a single learned scoring function rather than a
    position-specific mapping that has to be learned for each of the n workloads separately."""

    def __init__(self, obs_dim: int, hidden: int = 256, flat: bool = False):
        super().__init__()
        self.flat = flat
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        if flat:
            self.head_flat = nn.Linear(hidden, NK * NW * NP * NPI)
        else:
            self.emb_k = nn.Embedding(NK, 32)
            self.emb_w = nn.Embedding(NW, 32)
            self.emb_p = nn.Embedding(NP, 32)
            self.head_k = nn.Linear(hidden, NK)
            self.ctx_w = nn.Linear(hidden + 32, 64)
            self.score_w = nn.Sequential(nn.Linear(D_ENT + 64, 64), nn.Tanh(), nn.Linear(64, 1))
            self.head_p = nn.Sequential(nn.Linear(hidden + 64, 128), nn.Tanh(), nn.Linear(128, NP))
            self.head_pi = nn.Sequential(nn.Linear(hidden + 96, 64), nn.Tanh(), nn.Linear(64, NPI))
        self.v_r = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.v_c = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def head_w(self, obs: torch.Tensor, h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        ctx = torch.tanh(self.ctx_w(torch.cat([h, self.emb_k(k)], -1)))          # (B, 64)
        E = entity_rows(obs)                                                      # (B, NW, D_ENT)
        ctxE = ctx.unsqueeze(1).expand(-1, NW, -1)
        return self.score_w(torch.cat([E, ctxE], -1)).squeeze(-1)                 # (B, NW)

    @staticmethod
    def _masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        return logits.masked_fill(~mask, -1e9)

    def heads(self, obs: torch.Tensor, k: torch.Tensor, w: torch.Tensor, p: torch.Tensor):
        """Return logits for all heads given (teacher-forced) previous choices."""
        h = self.trunk(obs)
        lk = self.head_k(h)
        lw = self.head_w(obs, h, k)
        lp = self.head_p(torch.cat([h, self.emb_k(k), self.emb_w(w)], -1))
        lpi = self.head_pi(torch.cat([h, self.emb_k(k), self.emb_w(w), self.emb_p(p)], -1))
        return lk, lw, lp, lpi

    def values(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.v_r(obs).squeeze(-1), self.v_c(obs).squeeze(-1)


@dataclass
class PPOConfig:
    total_steps: int = 150_000
    rollout: int = 1024
    epochs: int = 6
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.98
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03          # early-stop PPO epochs when approx. KL exceeds this
    lr_anneal: bool = True           # linear learning-rate decay to 0 over total_steps
    reward_scale: bool = True        # scale rewards/costs by the running std of discounted returns
    lagrangian: bool = True
    cost_limit: float = 0.10         # delta: admissible mean cost per step
    lambda_lr: float = 0.02
    lambda_init: float = 0.5
    lambda_max: float = 5.0
    flat: bool = False               # ablation: flat action head (no factorization)
    warm_start_steps: int = 0        # behaviour-cloning steps from the runbook teacher before RL (0 = off)
    warm_start_epochs: int = 8
    seed: int = 0
    log_every: int = 1


class _RunningStd:
    """Welford estimate of the std of a discounted-return stream (reward scaling)."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, xs: np.ndarray) -> None:
        for x in xs:
            self.n += 1
            d = x - self.mean
            self.mean += d / self.n
            self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.m2 / max(1, self.n - 1))) if self.n > 1 else 1.0


class PPOLagrangian:
    def __init__(self, env: ReplayEnv, cfg: PPOConfig, curriculum=None):
        self.env, self.cfg = env, cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.net = FactorizedPolicy(env.obs_dim, flat=cfg.flat)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.lmbda = cfg.lambda_init if cfg.lagrangian else 0.0
        self.curriculum = curriculum
        self.history: List[Dict] = []
        # parameters of the actor vs. the two critics: gradient norms are clipped separately so that
        # a large value-regression loss cannot starve the policy gradient (global clipping did).
        crit = set(id(p) for m in (self.net.v_r, self.net.v_c) for p in m.parameters())
        self.actor_params = [p for p in self.net.parameters() if id(p) not in crit]
        self.critic_params = [p for p in self.net.parameters() if id(p) in crit]
        # running scale of discounted returns (reward and cost) used to normalise the regression
        # targets; a standard reward-scaling trick that keeps the critics' loss O(1).
        self._ret_r = _RunningStd(); self._ret_c = _RunningStd()

    # ------------------------------------------------------------------------------
    def _masks_np(self, mask_obj, k=None, w=None, p=None):
        if mask_obj is None:
            return None
        return mask_obj

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask_obj, deterministic: bool = False):
        """Autoregressive sampling with per-head masks. Returns (Action, logp, masks tuple)."""
        net = self.net
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if self.cfg.flat:
            logits = net.head_flat(net.trunk(o)).view(NK, NW, NP, NPI)
            m = torch.ones_like(logits, dtype=torch.bool)
            if mask_obj is not None:
                km = torch.as_tensor(mask_obj.k_mask())
                m &= km.view(NK, 1, 1, 1)
                for k in range(NK):
                    if km[k]:
                        wm = np.zeros(NW, bool); wmk = mask_obj.w_mask(k); wm[:len(wmk)] = wmk
                        m[k] &= torch.as_tensor(wm).view(NW, 1, 1)
            n_env = self.env.topo.n
            m[:, n_env:, :, :] = False
            m[:, :, self.env.topo.V:NP - 1, :] = False      # unused peer slots; NP-1 is the wildcard
            flat_logits = logits.masked_fill(~m, -1e9).view(-1)
            dist = torch.distributions.Categorical(logits=flat_logits)
            idx = flat_logits.argmax() if deterministic else dist.sample()
            k, r = divmod(int(idx), NW * NP * NPI); w, r = divmod(r, NP * NPI); p, pi = divmod(r, NPI)
            p_env = self.env.topo.V if p == NP - 1 else p
            return Action(k, w, p_env, pi), float(dist.log_prob(idx)), (m.view(-1).numpy(),)
        h = net.trunk(o)
        n_env = self.env.topo.n
        km = np.ones(NK, bool) if mask_obj is None else mask_obj.k_mask()
        lk = net.head_k(h).squeeze(0).masked_fill(~torch.as_tensor(km), -1e9)
        dk = torch.distributions.Categorical(logits=lk)
        k = int(lk.argmax() if deterministic else dk.sample())
        wm = np.zeros(NW, bool); wm[:n_env] = True if mask_obj is None else mask_obj.w_mask(k)
        lw = net.head_w(o, h, torch.tensor([k])).squeeze(0).masked_fill(~torch.as_tensor(wm), -1e9)
        dw = torch.distributions.Categorical(logits=lw)
        w = int(lw.argmax() if deterministic else dw.sample())
        pm = np.zeros(NP, bool)
        if mask_obj is None:
            pm[:self.env.topo.V] = True; pm[NP - 1] = True
        else:
            pm_env = mask_obj.p_mask(k, w); pm[:self.env.topo.V] = pm_env[:-1]; pm[NP - 1] = pm_env[-1]
        lp = net.head_p(torch.cat([h, net.emb_k(torch.tensor([k])), net.emb_w(torch.tensor([w]))], -1)).squeeze(0).masked_fill(~torch.as_tensor(pm), -1e9)
        dp = torch.distributions.Categorical(logits=lp)
        p = int(lp.argmax() if deterministic else dp.sample())
        p_env = self.env.topo.V if p == NP - 1 else p
        pim = np.ones(NPI, bool) if mask_obj is None else mask_obj.pi_mask(k, w, p_env)
        lpi = net.head_pi(torch.cat([h, net.emb_k(torch.tensor([k])), net.emb_w(torch.tensor([w])), net.emb_p(torch.tensor([p]))], -1)).squeeze(0).masked_fill(~torch.as_tensor(pim), -1e9)
        dpi = torch.distributions.Categorical(logits=lpi)
        pi = int(lpi.argmax() if deterministic else dpi.sample())
        logp = float(dk.log_prob(torch.tensor(k)) + REL_W[k] * dw.log_prob(torch.tensor(w))
                     + REL_P[k] * dp.log_prob(torch.tensor(p)) + REL_PI[k] * dpi.log_prob(torch.tensor(pi)))
        return Action(k, w, p_env, pi), logp, (km, wm, pm, pim)

    def _logp_entropy(self, obs, acts, masks):
        """Batched log-prob and entropy under current parameters."""
        if self.cfg.flat:
            logits = self.net.head_flat(self.net.trunk(obs)).masked_fill(~masks[0], -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            idx = ((acts[:, 0] * NW + acts[:, 1]) * NP + acts[:, 2]) * NPI + acts[:, 3]
            return dist.log_prob(idx), dist.entropy()
        k, w, p, pi = acts[:, 0], acts[:, 1], acts[:, 2], acts[:, 3]
        lk, lw, lp, lpi = self.net.heads(obs, k, w, p)
        km, wm, pm, pim = masks
        dk = torch.distributions.Categorical(logits=lk.masked_fill(~km, -1e9))
        dw = torch.distributions.Categorical(logits=lw.masked_fill(~wm, -1e9))
        dp = torch.distributions.Categorical(logits=lp.masked_fill(~pm, -1e9))
        dpi = torch.distributions.Categorical(logits=lpi.masked_fill(~pim, -1e9))
        rw, rp, rpi = REL_W[k], REL_P[k], REL_PI[k]
        logp = dk.log_prob(k) + rw * dw.log_prob(w) + rp * dp.log_prob(p) + rpi * dpi.log_prob(pi)
        ent = dk.entropy() + rw * dw.entropy() + rp * dp.entropy() + rpi * dpi.entropy()
        return logp, ent

    # ------------------------------------------------------------------------------
    def warm_start(self, teacher, n_steps: int, epochs: int, callback=None) -> None:
        """Runbook warm start: behaviour-clone the rule-based responder `teacher` (the operator's
        playbook) for n_steps environment steps, then hand over to constrained RL.  Only the
        relevant heads of each demonstrated action are trained (masked cross-entropy); the critics
        are left untouched.  The demonstrations are collected in the agent's own environment, so
        the shield still verifies every teacher action that is actually enforced."""
        if self.cfg.flat or n_steps <= 0:
            if self.cfg.flat and n_steps > 0:
                self._warm_start_flat(teacher, n_steps, epochs, callback)
            return
        env = self.env
        O, A = [], []
        MK, MW, MP, MPI = [], [], [], []
        obs, _ = env.reset(seed=self.cfg.seed + 7919)
        teacher.reset(env)
        for _ in range(n_steps):
            m = env.masks()
            a = teacher.act(env, obs)
            if m is not None and not m.k_mask()[a.k]:
                a = Action(0, 0, env.topo.V, 0)
            O.append(obs.copy()); A.append([a.k, a.w, NP - 1 if a.p >= env.topo.V else a.p, a.pi])
            if m is None:
                MK.append(np.ones(NK, bool)); mw = np.zeros(NW, bool); mw[:env.topo.n] = True; MW.append(mw)
                mp = np.zeros(NP, bool); mp[:env.topo.V] = True; mp[-1] = True; MP.append(mp); MPI.append(np.ones(NPI, bool))
            else:
                MK.append(m.k_mask()); mw = np.zeros(NW, bool); mw[:env.topo.n] = m.w_mask(a.k); MW.append(mw)
                pm = np.zeros(NP, bool); pe = m.p_mask(a.k, a.w); pm[:env.topo.V] = pe[:-1]; pm[-1] = pe[-1]; MP.append(pm)
                MPI.append(m.pi_mask(a.k, a.w, a.p))
            obs, r, done, trunc, info = env.step(a)
            if done:
                obs, _ = env.reset(); teacher.reset(env)
        tO = torch.as_tensor(np.array(O)); tA = torch.as_tensor(np.array(A))
        tM = tuple(torch.as_tensor(np.array(x)) for x in (MK, MW, MP, MPI))
        # the runbook is idle on >95% of steps: re-balance so that responses make up half of every batch
        A_np = np.array(A); pos = np.where(A_np[:, 0] != 0)[0]; neg = np.where(A_np[:, 0] == 0)[0]
        if len(pos) == 0:
            return
        idx = np.concatenate([neg, np.random.choice(pos, size=len(neg), replace=True)])
        n = len(idx)
        for ep in range(epochs):
            np.random.shuffle(idx); tot = 0.0; nb = 0
            for s0 in range(0, n, self.cfg.minibatch):
                mb = torch.as_tensor(idx[s0:s0 + self.cfg.minibatch])
                logp, _ = self._logp_entropy(tO[mb], tA[mb], tuple(m[mb] for m in tM))
                loss = -logp.mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, self.cfg.max_grad_norm); self.opt.step()
                tot += loss.item(); nb += 1
            if callback:
                callback({"warm_start_epoch": ep, "bc_nll": tot / max(1, nb), "n": n, "n_resp": len(pos)})

    def _warm_start_flat(self, teacher, n_steps: int, epochs: int, callback=None) -> None:
        env = self.env
        O, I = [], []
        obs, _ = env.reset(seed=self.cfg.seed + 7919); teacher.reset(env)
        for _ in range(n_steps):
            a = teacher.act(env, obs)
            p = NP - 1 if a.p >= env.topo.V else a.p
            O.append(obs.copy()); I.append(((a.k * NW + a.w) * NP + p) * NPI + a.pi)
            obs, r, done, trunc, info = env.step(a)
            if done:
                obs, _ = env.reset(); teacher.reset(env)
        tO = torch.as_tensor(np.array(O)); tI = torch.as_tensor(np.array(I))
        I_np = np.array(I); kinds = I_np // (NW * NP * NPI)
        pos = np.where(kinds != 0)[0]; neg = np.where(kinds == 0)[0]
        if len(pos) == 0:
            return
        idx = np.concatenate([neg, np.random.choice(pos, size=len(neg), replace=True)]); n = len(idx)
        for ep in range(epochs):
            np.random.shuffle(idx); tot = 0.0; nb = 0
            for s0 in range(0, n, self.cfg.minibatch):
                mb = torch.as_tensor(idx[s0:s0 + self.cfg.minibatch])
                logits = self.net.head_flat(self.net.trunk(tO[mb]))
                loss = F.cross_entropy(logits, tI[mb])
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, self.cfg.max_grad_norm); self.opt.step()
                tot += loss.item(); nb += 1
            if callback:
                callback({"warm_start_epoch": ep, "bc_nll": tot / max(1, nb), "n": n})

    # ------------------------------------------------------------------------------
    def train(self, callback=None) -> List[Dict]:
        cfg, env = self.cfg, self.env
        steps = 0
        obs, _ = env.reset(seed=cfg.seed)
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        ep_stats: List[Dict] = []
        it = 0
        t_start = time.time()
        while steps < cfg.total_steps:
            if self.curriculum is not None:
                self.curriculum.apply(env, steps / cfg.total_steps)
            B = cfg.rollout
            O = np.zeros((B, env.obs_dim), np.float32)
            A = np.zeros((B, 4), np.int64)
            LP = np.zeros(B, np.float32); R = np.zeros(B, np.float32); C = np.zeros(B, np.float32)
            Dn = np.zeros(B, np.float32); VR = np.zeros(B + 1, np.float32); VC = np.zeros(B + 1, np.float32)
            if cfg.flat:
                M0 = np.zeros((B, NK * NW * NP * NPI), bool)
            else:
                MK = np.zeros((B, NK), bool); MW = np.zeros((B, NW), bool); MP = np.zeros((B, NP), bool); MPI = np.zeros((B, NPI), bool)
            for i in range(B):
                a, logp, masks = self.act(obs, env.masks())
                with torch.no_grad():
                    vr, vc = self.net.values(torch.as_tensor(obs).unsqueeze(0))
                nobs, r, done, trunc, info = env.step(a)
                O[i] = obs; A[i] = [a.k, a.w, NP - 1 if a.p >= env.topo.V else a.p, a.pi]
                LP[i] = logp; R[i] = r; C[i] = info["cost"]; Dn[i] = float(done); VR[i] = vr; VC[i] = vc
                if cfg.flat:
                    M0[i] = masks[0]
                else:
                    MK[i], MW[i], MP[i], MPI[i] = masks
                ep_ret += r; ep_cost += info["cost"]; ep_len += 1
                obs = nobs
                if done:
                    s = info["episode"]; s.update({"ep_return": ep_ret, "ep_cost": ep_cost, "steps": steps + i + 1})
                    ep_stats.append(s)
                    obs, _ = env.reset()
                    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            with torch.no_grad():
                vr, vc = self.net.values(torch.as_tensor(obs).unsqueeze(0))
            VR[B], VC[B] = vr, vc
            steps += B
            mean_cost = float(C.mean())          # unscaled, for the Lagrangian constraint
            # reward / cost scaling by the running std of the discounted return (critics see O(1) targets)
            if cfg.reward_scale:
                disc_r = np.zeros(B, np.float32); disc_c = np.zeros(B, np.float32); gr = gc = 0.0
                for i in range(B):
                    gr = R[i] + cfg.gamma * gr * (1.0 - Dn[i - 1] if i > 0 else 1.0)
                    gc = C[i] + cfg.gamma * gc * (1.0 - Dn[i - 1] if i > 0 else 1.0)
                    disc_r[i] = gr; disc_c[i] = gc
                self._ret_r.update(disc_r); self._ret_c.update(disc_c)
                R = R / (self._ret_r.std + 1e-8); C_s = C / (self._ret_c.std + 1e-8)
            else:
                C_s = C
            # GAE for reward and cost
            adv_r = np.zeros(B, np.float32); adv_c = np.zeros(B, np.float32)
            gr = gc = 0.0
            for i in reversed(range(B)):
                nonterm = 1.0 - Dn[i]
                dr = R[i] + cfg.gamma * VR[i + 1] * nonterm - VR[i]
                dc = C_s[i] + cfg.gamma * VC[i + 1] * nonterm - VC[i]
                gr = dr + cfg.gamma * cfg.lam * nonterm * gr
                gc = dc + cfg.gamma * cfg.lam * nonterm * gc
                adv_r[i] = gr; adv_c[i] = gc
            ret_r = adv_r + VR[:B]; ret_c = adv_c + VC[:B]
            # Lagrangian update (projected gradient ascent)
            if cfg.lagrangian:
                self.lmbda = float(np.clip(self.lmbda + cfg.lambda_lr * (mean_cost - cfg.cost_limit), 0.0, cfg.lambda_max))
            adv_rn = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_cn = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)
            adv = (adv_rn - self.lmbda * adv_cn) / (1.0 + self.lmbda)
            # PPO epochs
            tO = torch.as_tensor(O); tA = torch.as_tensor(A); tLP = torch.as_tensor(LP)
            tADV = torch.as_tensor(adv); tRR = torch.as_tensor(ret_r); tRC = torch.as_tensor(ret_c)
            if cfg.flat:
                tM = (torch.as_tensor(M0),)
            else:
                tM = (torch.as_tensor(MK), torch.as_tensor(MW), torch.as_tensor(MP), torch.as_tensor(MPI))
            idx = np.arange(B)
            stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "gnorm_a": 0.0, "n_mb": 0.0}
            if cfg.lr_anneal:
                for g in self.opt.param_groups:
                    g["lr"] = cfg.lr * max(0.25, 1.0 - steps / cfg.total_steps)
            n_mb = 0
            for _ in range(cfg.epochs):
                np.random.shuffle(idx)
                stop = False
                for s0 in range(0, B, cfg.minibatch):
                    mb = torch.as_tensor(idx[s0:s0 + cfg.minibatch])
                    logp, ent = self._logp_entropy(tO[mb], tA[mb], tuple(m[mb] for m in tM))
                    ratio = torch.exp(logp - tLP[mb])
                    approx_kl = (tLP[mb] - logp).mean().item()
                    if cfg.target_kl and approx_kl > 1.5 * cfg.target_kl:
                        stop = True
                        break
                    pg = -torch.min(ratio * tADV[mb], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * tADV[mb]).mean()
                    vr, vc = self.net.values(tO[mb])
                    vf = F.mse_loss(vr, tRR[mb]) + F.mse_loss(vc, tRC[mb])
                    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent.mean()
                    self.opt.zero_grad(); loss.backward()
                    stats["gnorm_a"] += float(nn.utils.clip_grad_norm_(self.actor_params, cfg.max_grad_norm))
                    if os.environ.get("SAFEPOL_DEBUG_KL") and it < 2:
                        print(f"   it={it} mb={n_mb} kl={approx_kl:.4f} pg={pg.item():.4f} ratio_mean={ratio.mean().item():.3f}", flush=True)
                    nn.utils.clip_grad_norm_(self.critic_params, cfg.max_grad_norm)
                    self.opt.step()
                    n_mb += 1; stats["n_mb"] += 1.0
                    stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent"] += ent.mean().item()
                    stats["kl"] += approx_kl
                if stop:
                    break
            it += 1
            recent = ep_stats[-10:]
            rec = {"iter": it, "steps": steps, "lambda": self.lmbda, "mean_cost": mean_cost,
                   "mean_reward": float(R.mean()),
                   "ep_return": float(np.mean([e["ep_return"] for e in recent])) if recent else float("nan"),
                   "ep_cost": float(np.mean([e["ep_cost"] for e in recent])) if recent else float("nan"),
                   "ASR": float(np.nanmean([e["ASR"] for e in recent if e["ASR"] is not None])) if recent else float("nan"),
                   "FBR": float(np.mean([e["FBR"] for e in recent])) if recent else float("nan"),
                   "interv": float(np.mean([e["intervention_rate"] for e in recent])) if recent else float("nan"),
                   "elapsed": time.time() - t_start, **{k: v / max(1, n_mb) for k, v in stats.items()}}
            self.history.append(rec)
            if callback:
                callback(rec)
        self.episodes = ep_stats
        return self.history

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "lambda": self.lmbda, "cfg": self.cfg.__dict__}, path)

    def load(self, path: str):
        d = torch.load(path, map_location="cpu")
        self.net.load_state_dict(d["net"]); self.lmbda = d["lambda"]
