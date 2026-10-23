"""SAFEPOL: Shielded Adaptive Formally-verified Enforcement of POLicies.

Package layout
--------------
model/     policy objects (workloads, endpoints, rules, policy state) and reachability semantics
verify/    invariants (Phi), SMT conflict analysis, MaxSAT repair, Shield
bench/     Online-Boutique topology, legitimate traffic model, ATT&CK attack scenarios
sim/       flow-replay simulator (gymnasium Env) with factorized, maskable actions
agent/     PPO-Lagrangian (factorized heads), DQN+PSO (prior work), PPO (unconstrained)
baselines/ non-learning baselines B0-B3, B6
experiments/ RQ drivers, statistics and plotting
"""
__version__ = "0.1.0"
