# SAFEPOL — verify-or-repair shielding for adaptive container security policies

Repository: https://github.com/wbe3191/safepol — code, benchmark, per-episode results and cluster adapter (Apache-2.0).

Research prototype accompanying the paper "SAFEPOL: Verify-or-Repair Shielding for Adaptive Container Security Policies" (defensive security research: a controller-agnostic
*shield* verifies every candidate Kubernetes/Cilium network policy, seccomp profile, capability set and
rate limit against security/availability invariants and repairs it with a localised MaxSAT encoding, so
that the enforced policy satisfies the invariants at every step; three controller families — an
observation-driven allow-list, a detection-triggered runbook and a constrained RL agent — are evaluated
behind it).

## License
Apache-2.0 (see `LICENSE`).

## Layout
```
safepol/sim/          replay simulator (Online Boutique topology, workload dynamics, policy semantics, CMDP env)
safepol/bench/        attack scenarios S1–S8 (MITRE ATT&CK for Containers), adaptive attacker, paired eval set
safepol/verify/       Shield: incremental reachability, invariants φ1–φ8, exact conflict analysis, MaxSAT/greedy/PSO repair
safepol/agent/        PPO-Lagrangian over factorised masked heads, runbook warm start; DQN baseline
safepol/baselines/    static controllers B0–B3, B6, B7 (B8 = B3 behind the shield)
safepol/experiments/  method registry (SAFEPOL, SAFEPOL-H, B0–B8, A-*), train/eval entry point (`python -m safepol.experiments.run`)
safepol/k8s/          cluster adapter: Policy -> CiliumNetworkPolicy / seccomp / securityContext, NetworkPolicy parser, kubectl + Hubble helpers
scripts/              job runner, statistics, figures, scalability benchmark, reproduce.sh
scripts/k8s/          real-cluster validation (RQ7): deploy.sh, exp_a_upstream.py, exp_b_reconcile.py, exp_c_e2e.py, summarise.py
k8s/                  Online Boutique v0.10.1 manifests (+ promoservice), upstream NetworkPolicies, generated CNPs
tests/                smoke / dynamics / incremental / local_repair / host_response / ppo_sanity / logp_consistency
results/evaltest/     per-episode CSVs of the main experiment (method × seed); results/tables, results/scalability.csv
results/k8s/          real-cluster CSVs (upstream_shield, fidelity, reconcile_net/host, e2e) and summary.txt
```

## Reproduce
```
python -m venv ~/safepol-venv && ~/safepol-venv/bin/pip install -r requirements.txt
bash scripts/reproduce.sh          # ~3.5 h on 8 cores: trains 70 models, evaluates 105 method×seed pairs, builds tables/figures
bash scripts/reproduce.sh tables   # only statistics/figures from the shipped results/
```
`stats.py` / `figures.py` write the paper tables and figures to `paper/tables` and `paper/figures` (created on demand; the manuscript sources are not part of this repository).

Real-cluster validation (Section RQ7) needs a Kubernetes cluster with Cilium (policy enforcement, Hubble) and `kubectl`
pointing at it; it was run on Docker Desktop Kubernetes 1.34 + Cilium 1.17.3 on one node:
```
bash scripts/k8s/deploy.sh                                   # namespace boutique: Online Boutique + promoservice + probe pod
PYTHONPATH=. python scripts/k8s/exp_a_upstream.py            # shield on the 13 upstream NetworkPolicies + 209-probe fidelity check
PYTHONPATH=. python scripts/k8s/exp_b_reconcile.py 20        # apply->enforce latency (network) and seccomp/caps pod re-create (host)
PYTHONPATH=. python scripts/k8s/exp_c_e2e.py 3 S2,S3 B3,B8   # runbook with/without shield, benign attacker stand-in (~1 h)
PYTHONPATH=. python scripts/k8s/summarise.py                 # results/k8s/summary.txt, paper/tables/k8s_e2e.tex
```
The "attacker" is an ephemeral debug container in the victim pod issuing `nc -z` / `curl` connection attempts; nothing is exploited.

## Key results (5 seeds, 408 paired episodes, strong workload dynamics)
| | ASR | mitigated | TTM | FBR | SLO | inv. viol. | changes/ep |
|---|---|---|---|---|---|---|---|
| B3 runbook (unshielded) | 7.9 % | 96.2 % | 3.7 s | 0.7 % | 10.5 % | 2011 | 7 |
| **B8 runbook + shield** | **7.9 %** | 96.2 % | **3.7 s** | **0.4 %** | **6.2 %** | **0** | 7 |
| B7 hardened (static, shielded) | **0 %** | 100 % | – | 1.1 % | 13.0 % | 0 | 36 |
| SAFEPOL (constrained RL + shield) | 12.7±4.5 % | **100 %** | 5.0 s | 0.5 % | 8.8 % | 0 | 79 |
| SAFEPOL-H (runbook first, RL otherwise) | 8.3±1.2 % | 97.2 % | 4.5 s | 0.5 % | 7.0 % | 0 | 61 |
| B5 PPO (no shield) | 17.3 % | 98 % | 6.3 s | 0.9 % | 14.5 % | 3683 | 83 |

Effect of the shield on each controller family (SLO %, inv. violations): allow-list 23.6→15.5, 4022→0;
runbook 10.5→6.2, 2011→0; learner 14.2→8.8, 2848→0.

SLO-violating periods by dynamics slice (stationary / flash / job / flash+job / drift):
B8 0.0 / 4.6 / 8.6 / 12.6 / 7.5 %; SAFEPOL 0.2 / 6.3 / 12.6 / 17.6 / 9.8 %; B7 0.9 / 10.5 / 17.6 / 25.1 / 14.7 %;
B3 3.9 / 11.0 / 11.9 / 17.4 / 44.5 %.

Negative result reported in the paper: on this benchmark the shielded runbook beats the learned controller
and the hybrid; the shield, not learning, accounts for the availability gains.
