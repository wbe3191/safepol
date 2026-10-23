"""Flow-replay simulator: a gymnasium environment implementing the CMDP of Sec. 1.5.

State  s_t = (X_t per-workload telemetry, G_t flow-graph features, h(P_t) policy embedding)
Action a_t = (k, w, p, pi) factorized (see sim/action_space.py)
Step   P_{t+1} = Shield(P_t (+) a_t); legitimate and attack flows are replayed against P_{t+1}
       and verdicts (allow / drop / rate-limited) produce reward r_t and cost c_t.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None

from safepol.bench.attacks import Attack, FlowEvent, HostEvent, make_attack, ATTACK_IDS
from safepol.bench.policies import allowlist_policy, initial_policy, make_invariants
from safepol.bench.topology import edge_rates, online_boutique, make_port_space
from safepol.model.policy_model import DANGEROUS_SYSCALLS, Policy, Topology
from safepol.sim.action_space import ACTION_TYPES, K_IDX, Action, ActionSpace
from safepol.verify.invariants import Invariants
from safepol.verify.repair import pso_conflict_resolve
from safepol.verify.shield import Shield

FEATURES = ["cpu", "mem", "conns", "new_peers", "drops", "l7_err", "danger_sc", "alerts",
            "vuln", "t_since_change", "ext_egress", "ingress_rate", "sc_denied"]
D = len(FEATURES)
GRAPH_FEATURES = ["g_in", "g_out", "g_new_peers", "g_ext"]
DG = len(GRAPH_FEATURES)
N_MAX, V_MAX = 13, 17          # boutique (+promoservice) and 4 externals


@dataclass
class EnvConfig:
    scenarios: List[str] = field(default_factory=lambda: list(ATTACK_IDS))
    horizon: int = 100              # decision periods per episode (Delta = 5 s)
    delta: float = 5.0
    attack_start: Tuple[int, int] = (8, 30)
    attack_duration: int = 60
    p_no_attack: float = 0.1        # fraction of benign episodes
    drift: bool = False             # topology drift (promoservice appears) in some episodes
    p_drift: float = 0.5
    chaos: bool = False             # Chaos-Mesh-like noise on resource features
    shield: bool = True
    repair: str = "maxsat"          # maxsat | greedy | pso
    repair_timeout_ms: int = 200
    conflict_resolver: str = "none" # when shield=False: none | pso  (prior-work B4 setting)
    masks: bool = True
    graph_features: bool = True
    reward_source: str = "labels"   # labels | detector
    runbook_prior: bool = False     # hybrid: the B3 response playbook fires first on alerts, the agent acts otherwise
    fp_alert_rate: float = 0.003    # false-positive alert probability per workload per period
    init_policy: str = "mixed"      # minimal | allowlist | mixed  (policy at episode start)
    legit_window: int = 12          # T for phi6 (periods)
    weights: Tuple[float, ...] = (1.0, 2.0, 5.0, 0.1, 0.05, 5.0)   # w1..w5 as in Sec. 1.5; w6 = attacker objective open
    lambda_slo: float = 1.0
    frontend_capacity: int = 600
    # workload dynamics that make a fixed hardening level wrong at some times:
    p_flash: float = 0.4            # legitimate flash crowd on Internet->frontend
    flash_factor: Tuple[float, float] = (5.0, 7.0)   # keeps legit load below frontend_capacity
    flash_duration: Tuple[int, int] = (10, 20)
    p_job: float = 0.5              # legitimate privileged jobs (e.g. model reload mounting a volume)
    jobs_per_episode: Tuple[int, int] = (1, 2)
    job_duration: Tuple[int, int] = (10, 15)
    job_syscalls: Tuple[str, ...] = ("mount", "umount2")
    seed: int = 0


class ReplayEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.nprng = np.random.default_rng(cfg.seed)
        self.ps = make_port_space()
        self._topos = {False: online_boutique(self.ps, False), True: online_boutique(self.ps, True)}
        self.topo: Topology = self._topos[False]
        self.obs_dim = N_MAX * (D + DG) + (7 + 3 * N_MAX)
        if gym:
            self.observation_space = spaces.Box(-5, 5, (self.obs_dim,), np.float32)
            self.action_space = spaces.MultiDiscrete([len(ACTION_TYPES), N_MAX, V_MAX + 1, 3])
        self.scenario: Optional[str] = None
        self.attack: Optional[Attack] = None
        self.history: List[Policy] = []
        self.episode_log: List[Dict] = []

    # ------------------------------------------------------------------------------
    def _setup_topology(self, drift: bool):
        self.topo = self._topos[drift]
        self.inv: Invariants = make_invariants(self.topo)
        if drift:
            # the operator's dependency graph G_dep is stale: it does not know promoservice
            base = self._topos[False]
            self.inv.dep_edges_override = [
                (self.topo.index[base.names[s]], self.topo.index[base.names[d]], m) for (s, d, m) in base.dep_edges]
            self.inv.legit_flows = set(self.inv.dep_edges_override)
        self.aspace = ActionSpace(self.topo)
        self.shield = Shield(self.inv, self.aspace, self.cfg.repair, self.cfg.repair_timeout_ms,
                             enabled=self.cfg.shield)
        self.rates = edge_rates(self.topo)
        self.edges = list(self.topo.dep_edges)
        self.inet = self.topo.by_kind("Internet")
        self.frontend = self.topo.workload_idx(lambda w: w.tier == "ingress")[0]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed); self.nprng = np.random.default_rng(seed)
        options = options or {}
        drift = options.get("drift", self.cfg.drift and self.rng.random() < self.cfg.p_drift)
        self._setup_topology(drift)
        self.drift = drift
        self.drift_step = self.rng.randint(5, 25) if drift else None
        self.drift_active = False
        if getattr(self, "_runbook", None) is not None:
            self._runbook.reset(self)
        init = options.get("init", self.cfg.init_policy)
        if init == "mixed":
            init = "allowlist" if self.rng.random() < 0.5 else "minimal"
        self.init_kind = init
        if init == "allowlist":
            self.P = allowlist_policy(self.topo, self.inv, set(self.inv.dep_edges()), "runtime_default")
        else:
            self.P = initial_policy(self.topo, self.inv)
        self.P0_AS = max(1, self.P.attack_surface())
        self.history = []
        self.t = 0
        self.last_change = 0
        # attack: an explicit options["scenario"] (including None = benign) is honoured verbatim
        if "scenario" in options:
            sc = options["scenario"]
        else:
            sc = None if self.rng.random() < self.cfg.p_no_attack else self.rng.choice(self.cfg.scenarios)
        self.scenario = sc
        if sc:
            start = options.get("start", self.rng.randint(*self.cfg.attack_start))
            self.attack = make_attack(sc, self.topo, self.rng, start, self.cfg.attack_duration, options.get("victim"))
        else:
            self.attack = None
        # legitimate workload dynamics (fixed per replay through the seeded rng)
        self.flash = None
        if self.rng.random() < self.cfg.p_flash:
            fs = self.rng.randint(5, self.cfg.horizon - 25)
            self.flash = (fs, fs + self.rng.randint(*self.cfg.flash_duration), self.rng.uniform(*self.cfg.flash_factor))
        self.jobs: List[Tuple[int, int, int]] = []
        if self.rng.random() < self.cfg.p_job:
            cands = [i for i, w in enumerate(self.topo.workloads)
                     if w.tier == "app" and i not in self.inv.untrusted and w.name != "loadgenerator"]
            for w in self.rng.sample(cands, self.rng.randint(*self.cfg.jobs_per_episode)):
                js = self.rng.randint(5, self.cfg.horizon - 20)
                self.jobs.append((w, js, js + self.rng.randint(*self.cfg.job_duration)))
        if "flash" in options:
            self.flash = options["flash"]
        if "job" in options:
            self.jobs = [options["job"]] if options["job"] else []
        self.exposure = 0
        self.goal_hits = 0
        self.inv_violations = 0        # of enforced policies (0 by construction with shield)
        self.legit_window: List[Set] = []
        self.observed_window: List[Set] = []
        self.alerts = np.zeros(self.topo.n, np.float32)
        self.X = np.zeros((N_MAX, D), np.float32)
        self.G = np.zeros((N_MAX, DG), np.float32)
        self.prev_peers: Dict[int, Set[int]] = {w: set() for w in range(self.topo.n)}
        self.episode_log = []
        self._init_features()
        return self._obs(), {"scenario": sc, "drift": drift}

    def _init_features(self):
        n = self.topo.n
        for w in range(n):
            self.X[w, FEATURES.index("cpu")] = 0.3 + 0.05 * self.nprng.standard_normal()
            self.X[w, FEATURES.index("mem")] = 0.4 + 0.05 * self.nprng.standard_normal()
            self.X[w, FEATURES.index("vuln")] = self.topo.workloads[w].image_vuln_score

    # ------------------------------------------------------------------------------
    def _legit_flows(self) -> List[FlowEvent]:
        fl = []
        edges = self.edges if not (self.drift and not self.drift_active) else \
            [e for e in self.edges if self.topo.names[e[1]] != "promoservice" and self.topo.names[e[0]] != "promoservice"]
        for (s, d, m) in edges:
            lam = self.rates[(s, d)] * (1.0 + 0.3 * np.sin(self.t / 15.0))   # diurnal-like load
            if self.cfg.chaos:
                lam *= self.nprng.uniform(0.3, 1.8)
            if self.flash and self.flash[0] <= self.t < self.flash[1] and s == self.inet and d == self.frontend:
                lam *= self.flash[2]                                           # legitimate flash crowd
            c = int(self.nprng.poisson(lam))
            if c > 0:
                fl.append(FlowEvent(s, d, m, c, "legit"))
        return fl

    def _legit_host_events(self) -> List[HostEvent]:
        return [HostEvent(w, set(self.cfg.job_syscalls), set(), "legit")
                for (w, s, e) in self.jobs if s <= self.t < e]

    def _job_active(self) -> bool:
        return any(s <= self.t < e for (_, s, e) in self.jobs)

    def _enforce(self, flows: List[FlowEvent], P: Policy) -> Tuple[Dict[str, int], np.ndarray, np.ndarray, int]:
        """Replay flows against P: returns counts, per-workload drop counts, per-workload load, overload flag."""
        n = self.topo.n
        counts = {"legit": 0, "legit_drop": 0, "mal": 0, "mal_drop": 0}
        drops = np.zeros(n); load_in = np.zeros(n); load_out = np.zeros(n)
        passed: List[Tuple[FlowEvent, int]] = []
        for f in flows:
            allowed = P.allows(f.src, f.dst, f.port_mask)
            c = f.count
            if allowed and self.topo.is_workload(f.dst):
                q = P.endpoint[f.dst].rate_limit
                if q is not None:
                    cap = int(q * self.cfg.delta)
                    c = min(c, cap)                       # per-source rate limiting
            dropped = f.count - (c if allowed else 0)
            counts[f.label] += f.count
            counts[f.label + "_drop"] += dropped
            if self.topo.is_workload(f.src):
                drops[f.src] += dropped; load_out[f.src] += c if allowed else 0
            if self.topo.is_workload(f.dst):
                drops[f.dst] += dropped; load_in[f.dst] += c if allowed else 0
            if allowed and c > 0:
                passed.append((f, c))
        # overload of the frontend (SLO): capacity in requests per period
        overload = 0
        if load_in[self.frontend] > self.cfg.frontend_capacity:
            frac = 1 - self.cfg.frontend_capacity / load_in[self.frontend]
            for f, c in passed:
                if f.dst == self.frontend and f.label == "legit":
                    lost = int(frac * c)
                    counts["legit_drop"] += lost; drops[f.dst] += lost
            overload = 1
        return counts, drops, load_in + load_out, overload

    def _host_events(self, events: List[HostEvent], P: Policy) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
        """Per-label (mal / legit) totals and blocked counts, per-workload danger and blocked counts."""
        counts = {"mal": 0, "mal_blocked": 0, "legit": 0, "legit_blocked": 0}
        danger = np.zeros(self.topo.n)
        blocked_w = np.zeros(self.topo.n)
        for e in events:
            ep = P.endpoint[e.workload]
            for sc in e.syscalls:
                counts[e.label] += 1
                if sc not in ep.syscalls:
                    counts[e.label + "_blocked"] += 1
                    blocked_w[e.workload] += 1
                if sc in DANGEROUS_SYSCALLS and e.label != "legit":
                    # process-aware runtime detection (Tetragon-style) attributes privileged
                    # syscalls of an allow-listed job binary to the job and does not alert on
                    # them; a seccomp profile is process-agnostic and still blocks them.
                    danger[e.workload] += 1
            for cp in e.caps:
                counts[e.label] += 1
                if cp not in ep.caps:
                    counts[e.label + "_blocked"] += 1
                    blocked_w[e.workload] += 1
                danger[e.workload] += 0.5
        return counts, danger, blocked_w

    # ------------------------------------------------------------------------------
    def decode(self, action) -> Action:
        if isinstance(action, Action):
            return action
        k, w, p, pi = [int(x) for x in action]
        w = min(w, self.topo.n - 1)
        p = min(p, self.topo.V)   # V = wildcard
        return Action(k, w, p, pi)

    def step(self, action):
        cfg = self.cfg
        a = self.decode(action)
        if self.drift and not self.drift_active and self.t >= self.drift_step:
            self.drift_active = True
        # 0. optional runbook prior: the operator's response playbook fires first on an alert;
        #    the learned policy acts on every other step (hybrid SAFEPOL-H)
        self.runbook_fired = False
        if cfg.runbook_prior:
            if getattr(self, "_runbook", None) is None:
                from safepol.baselines.static import B3RuleResponder  # lazy: avoids circular import
                self._runbook = B3RuleResponder(); self._runbook.reset(self)
            ra = self._runbook.act(self, None)
            if ra.kind != "NOOP":
                a = ra; self.runbook_fired = True
        # 1. candidate policy and shield
        P_cand = self.aspace.apply(self.P, a, self.history, self.inv)
        if not cfg.shield and cfg.conflict_resolver == "pso":
            P_cand, _ = pso_conflict_resolve(P_cand, seed=self.t)
        subjects = None if a.kind in ("ROLLBACK", "NOOP") else {a.w}
        ok, P_safe, sinfo = self.shield.verify_or_repair(P_cand, self.P, subjects)
        intervened = not ok and (sinfo["method"] is not None)
        changed = P_safe.edit_distance(self.P) > 0 or a.kind == "ROLLBACK"
        if changed:
            self.history.append(self.P)
            self.history = self.history[-5:]
            self.last_change = self.t
        enforced_viol = self.inv.check(P_safe) if not cfg.shield else []
        self.inv_violations += 1 if enforced_viol else 0
        self.P = P_safe
        # 2. traffic
        flows = self._legit_flows()
        host: List[HostEvent] = self._legit_host_events()
        attack_active = self.attack is not None and self.attack.active(self.t)
        if attack_active:
            afl, ahost = self.attack.events(self.t, self.P)
            flows += afl; host += ahost
        counts, drops, load, overload = self._enforce(flows, self.P)
        hc, danger, blocked_w = self._host_events(host, self.P)
        h_total, h_blocked = hc["mal"], hc["mal_blocked"]
        lh_total, lh_blocked = hc["legit"], hc["legit_blocked"]
        drops += blocked_w                     # seccomp denials are kernel-observable like DROPs
        goal = attack_active and self.attack.goal_reached(self.t, self.P)
        if attack_active:
            mal_total = counts["mal"] + h_total
            mal_drop = counts["mal_drop"] + h_blocked
            if self.attack.sid == "S5":
                q = self.P.endpoint[self.attack.victim].rate_limit
                mal_drop = mal_total if (q is not None and q * cfg.delta <= 400) else 0
            self.attack.observe_outcome(self.t, self.P, mal_drop, mal_total)
            if self.attack.stage(self.t) == 2:
                self.exposure += int(goal)
                self.goal_hits += int(goal)
        # 3. legit window for phi6
        legit_now = {(f.src, f.dst, f.port_mask) for f in flows if f.label == "legit"}
        self.legit_window.append(legit_now)
        self.legit_window = self.legit_window[-cfg.legit_window:]
        self.inv.legit_flows = set().union(*self.legit_window) | set(self.inv.dep_edges())
        # 4. telemetry / features
        alerts = self._alerts(attack_active, danger)
        # unlabeled observation window (what an observation-driven generator would see):
        # flows that were allowed and whose endpoints raised no alert this period
        self.observed_window.append({(f.src, f.dst, f.port_mask) for f in flows
                                     if self.P.allows(f.src, f.dst, f.port_mask)
                                     and not ((self.topo.is_workload(f.src) and alerts[f.src] > 0)
                                              or (self.topo.is_workload(f.dst) and alerts[f.dst] > 0))})
        self.observed_window = self.observed_window[-cfg.legit_window:]
        self.alerts = alerts
        if cfg.reward_source == "detector":
            # no ground-truth labels anywhere: phi6 is built from the alert-filtered observation window
            self.inv.legit_flows = set().union(*self.observed_window) | set(self.inv.dep_edges())
        self._update_features(flows, drops, load, danger, alerts, overload, blocked_w)
        # 5. reward and cost
        AS_prev = self.P0_AS if not hasattr(self, "_AS_last") else self._AS_last
        AS_now = self.P.attack_surface()
        self._AS_last = AS_now
        w1, w2, w3, w4, w5, w6 = cfg.weights
        if cfg.reward_source == "labels":
            mal_t, mal_d = counts["mal"] + h_total, counts["mal_drop"] + h_blocked
            leg_t, leg_d = counts["legit"], counts["legit_drop"]
        else:  # detector-driven: flows from alerted workloads are treated as malicious
            alerted = {w for w in range(self.topo.n) if alerts[w] > 0}
            mal_t = sum(f.count for f in flows if f.src in alerted or f.dst in alerted) + h_total
            mal_d = sum(f.count - (f.count if self.P.allows(f.src, f.dst, f.port_mask) else 0)
                        for f in flows if f.src in alerted or f.dst in alerted) + h_blocked
            leg_t = sum(f.count for f in flows if not (f.src in alerted or f.dst in alerted))
            leg_d = max(0, counts["legit_drop"] + counts["mal_drop"] - mal_d)
        goal_pen = 1.0 if goal else 0.0
        r = (w1 * (AS_prev - AS_now) / self.P0_AS
             + w2 * (mal_d / (mal_t + 1) if mal_t else 0.0)
             - w3 * (leg_d / (leg_t + 1))
             - w3 * (lh_blocked / (lh_total + 1))          # legitimate privileged job blocked by seccomp
             - w4 * len(self.P.rules) / 50.0
             - w5 * float(a.kind != "NOOP")
             - w6 * goal_pen)
        slo = float(overload or (counts["legit"] > 0 and counts["legit_drop"] / counts["legit"] > 0.02)
                    or lh_blocked > 0)
        cost = float(not ok and sinfo["violations"] != []) + cfg.lambda_slo * slo
        # 6. bookkeeping
        self.episode_log.append({
            "t": self.t, "scenario": self.scenario, "action": a.kind, "intervened": intervened,
            "cand_violation": bool(sinfo["violations"]), "conflicts": sinfo["conflicts"],
            "enforced_violation": bool(enforced_viol), "legit": counts["legit"], "legit_drop": counts["legit_drop"],
            "legit_host": lh_total, "legit_host_drop": lh_blocked,
            "flash": bool(self.flash and self.flash[0] <= self.t < self.flash[1]),
            "job": self._job_active(),
            "mal": counts["mal"] + h_total, "mal_drop": counts["mal_drop"] + h_blocked, "goal": bool(goal),
            "attack_active": attack_active, "stage": self.attack.stage(self.t) if attack_active else 0,
            "slo": slo, "AS": AS_now, "rules": len(self.P.rules), "reward": r, "cost": cost,
            "repair_method": sinfo["method"], "repair_distance": sinfo["distance"]})
        self.t += 1
        done = self.t >= cfg.horizon
        info = {"cost": cost, "slo": slo, "goal": goal, "intervened": intervened, "ok": ok}
        if done:
            info["episode"] = self.summary()
        return self._obs(), float(r), done, False, info

    # ------------------------------------------------------------------------------
    def _alerts(self, attack_active: bool, danger: np.ndarray) -> np.ndarray:
        n = self.topo.n
        al = (self.nprng.random(n) < self.cfg.fp_alert_rate).astype(np.float32)
        if attack_active and self.rng.random() < self.attack.detect_prob:
            al[self.attack.victim] = 1.0
        al += (danger > 0).astype(np.float32)
        return np.minimum(al, 1.0)

    def _update_features(self, flows, drops, load, danger, alerts, overload, blocked_w):
        n = self.topo.n
        X = np.zeros((N_MAX, D), np.float32)
        G = np.zeros((N_MAX, DG), np.float32)
        peers: Dict[int, Set[int]] = {w: set() for w in range(n)}
        fin = np.zeros(n); fout = np.zeros(n); ext = np.zeros(n)
        for f in flows:
            if self.topo.is_workload(f.src):
                fout[f.src] += f.count; peers[f.src].add(f.dst)
                if not self.topo.is_workload(f.dst):
                    ext[f.src] += f.count
            if self.topo.is_workload(f.dst):
                fin[f.dst] += f.count; peers[f.dst].add(f.src)
        sig = self.attack.feature_signature() if (self.attack and self.attack.active(self.t)) else None
        victim = self.attack.victim if sig else -1
        for w in range(n):
            new_peers = len(peers[w] - self.prev_peers[w])
            self.prev_peers[w] = peers[w]
            cpu = 0.3 + 0.0004 * load[w] + 0.03 * self.nprng.standard_normal()
            mem = 0.4 + 0.02 * self.nprng.standard_normal()
            if self.cfg.chaos:
                cpu += self.nprng.uniform(-0.1, 0.4) * (self.nprng.random() < 0.1)
            if w == victim:
                cpu += sig["cpu"]
            X[w] = [cpu, mem, min(1.0, (fin[w] + fout[w]) / 300.0 + (sig["conns"] if w == victim else 0)),
                    min(1.0, new_peers / 5.0 + (sig["new_peers"] if w == victim else 0)),
                    min(1.0, drops[w] / 50.0), float(overload and w == self.frontend),
                    min(1.0, danger[w] / 3.0 + (sig["danger_sc"] * 0.3 if w == victim else 0)),
                    alerts[w], self.topo.workloads[w].image_vuln_score,
                    min(1.0, (self.t - self.last_change) / 20.0),
                    min(1.0, ext[w] / 20.0 + (sig["ext_egress"] if w == victim else 0)),
                    min(1.0, fin[w] / 300.0), min(1.0, blocked_w[w] / 2.0)]
            G[w] = [min(1.0, fin[w] / 300.0), min(1.0, fout[w] / 300.0), min(1.0, new_peers / 5.0), min(1.0, ext[w] / 20.0)]
        self.X, self.G = X, G

    def _obs(self) -> np.ndarray:
        h = self.P.embedding()
        hp = np.zeros(7 + 3 * N_MAX, np.float32)
        n = self.topo.n
        hp[:7] = h[:7]
        hp[7:7 + n] = h[7:7 + n]; hp[7 + N_MAX:7 + N_MAX + n] = h[7 + n:7 + 2 * n]
        hp[7 + 2 * N_MAX:7 + 2 * N_MAX + n] = h[7 + 2 * n:7 + 3 * n]
        G = self.G if self.cfg.graph_features else np.zeros_like(self.G)
        return np.concatenate([self.X.ravel(), G.ravel(), hp]).astype(np.float32)

    def masks(self):
        if not self.cfg.masks:
            return None
        return self.shield.mask(self.P, len(self.history))

    # ------------------------------------------------------------------------------
    def summary(self) -> Dict:
        log = self.episode_log
        legit = sum(e["legit"] for e in log); ldrop = sum(e["legit_drop"] for e in log)
        mal = sum(e["mal"] for e in log); mdrop = sum(e["mal_drop"] for e in log)
        lh = sum(e["legit_host"] for e in log); lhd = sum(e["legit_host_drop"] for e in log)
        out = {"scenario": self.scenario or "none", "drift": self.drift, "init": self.init_kind,
               "flash": self.flash is not None, "job": bool(self.jobs), "n_jobs": len(self.jobs),
               "FBR": ldrop / max(1, legit), "MBR": mdrop / max(1, mal),
               "FBR_host": (lhd / lh) if lh else float("nan"),
               "SLO_viol_rate": float(np.mean([e["slo"] for e in log])),
               "inv_violations": sum(e["enforced_violation"] for e in log),
               "cand_violations": sum(e["cand_violation"] for e in log),
               "interventions": sum(e["intervened"] for e in log),
               "intervention_rate": sum(e["intervened"] for e in log) / len(log),
               "actions": sum(e["action"] != "NOOP" for e in log),
               "rules_final": len(self.P.rules), "AS_final": self.P.attack_surface(), "AS0": self.P0_AS,
               "BR_mean_final": float(self.P.blast_radius().mean()),
               "return": sum(e["reward"] for e in log), "cost": sum(e["cost"] for e in log),
               "conflicts_seen": sum(e["conflicts"] > 0 for e in log)}
        if self.attack:
            a = self.attack
            ttm = None
            if a.first_event_step is not None and a.first_blocked_step is not None:
                ttm = (a.first_blocked_step - a.first_event_step) * self.cfg.delta
            out.update({"TTM": ttm, "mitigated": ttm is not None, "ASR": float(self.goal_hits > 0),
                        "exposure_s": self.exposure * self.cfg.delta, "victim": self.topo.names[a.victim]})
        else:
            out.update({"TTM": None, "mitigated": None, "ASR": None, "exposure_s": 0.0, "victim": None})
        st = self.shield.stats
        out["verify_p50_ms"] = float(np.median(st.verify_latency) * 1000) if st.verify_latency else 0.0
        out["repair_p50_ms"] = float(np.median(st.repair_latency) * 1000) if st.repair_latency else 0.0
        out["repair_distance_mean"] = float(np.mean(st.distances)) if st.distances else 0.0
        return out
