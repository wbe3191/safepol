import numpy as np, torch
from safepol.experiments.run import build
from safepol.agent.ppo_lagrangian import NP, NK, NW, NPI

spec, env, agent, cur = build("SAFEPOL", 0, steps=1000)
cur.apply(env, 0.0)
obs, _ = env.reset(seed=1)
B = 300
O = np.zeros((B, env.obs_dim), np.float32); A = np.zeros((B, 4), np.int64); LP = np.zeros(B, np.float32)
MK = np.zeros((B, NK), bool); MW = np.zeros((B, NW), bool); MP = np.zeros((B, NP), bool); MPI = np.zeros((B, NPI), bool)
for i in range(B):
    a, logp, masks = agent.act(obs, env.masks())
    O[i] = obs; A[i] = [a.k, a.w, NP - 1 if a.p >= env.topo.V else a.p, a.pi]; LP[i] = logp
    MK[i], MW[i], MP[i], MPI[i] = masks
    obs, r, done, trunc, info = env.step(a)
    if done:
        obs, _ = env.reset()
tO = torch.as_tensor(O); tA = torch.as_tensor(A)
tM = (torch.as_tensor(MK), torch.as_tensor(MW), torch.as_tensor(MP), torch.as_tensor(MPI))
with torch.no_grad():
    lp, ent = agent._logp_entropy(tO, tA, tM)
d = LP - lp.numpy()
print("mean diff", d.mean(), "max abs", np.abs(d).max())
bad = np.argsort(-np.abs(d))[:5]
for i in bad:
    print(i, A[i], LP[i], lp[i].item(), "masks sum", MK[i].sum(), MW[i].sum(), MP[i].sum(), MPI[i].sum())
    # per-sample recomputation
    lp1, _ = agent._logp_entropy(tO[i:i+1], tA[i:i+1], tuple(m[i:i+1] for m in tM))
    print("   single:", lp1.item())
