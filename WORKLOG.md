# SAFEPOL 工作日志

> 目的：记录实验进度、关键发现与决策，便于中断后恢复。运行环境：WSL `Ubuntu-22.04`，venv `~/safepol-venv`，
> 入口 `scripts/run_jobs.sh JOBFILE PARALLEL`（8 核 CPU）。汇总：`scripts/summarize.py results/<tag>`；
> 动作诊断：`scripts/inspect_actions.py results/<tag> <method> <seed>`。

## 2026-10-19 pilot-1（60k 步，seed 0，`results/evaltest`）

| 方法 | ASR | mitigated | TTM | FBR | SLO | inv_viol | actions/ep |
|---|---|---|---|---|---|---|---|
| B0-none | 1.000 | 0.00 | – | 0.006 | 0.080 | 40800 | 0 |
| B1-static | 0.250 | 0.75 | 0.0 | 0.014 | 0.171 | 4022 | 0 |
| B2-observed | 0.125 | 0.875 | 0.0 | 0.014 | 0.171 | 4022 | 0 |
| B3-responder | 0.073 | 0.96 | 3.6 | 0.004 | 0.050 | 2011 | 7.2 |
| B4-DQN-PSO | 0.492 | 0.61 | 8.1 | 0.009 | 0.069 | 3024 | 77 |
| B5-PPO | 0.614 | 0.39 | 0.0 | 0.010 | 0.125 | 2011 | 97 |
| B6-shield-only | 0.125 | 0.875 | 0.0 | 0.007 | 0.081 | 0 | 0 |
| SAFEPOL (pilot1) | 0.128 | 0.875 | 0.16 | 0.007 | 0.081 | 0 | 100 |

成立的假设：H3（SAFEPOL/B6 不变量违反 = 0，B4/B5 > 0）；drift 组 B1/B2 SLO=0.85、inv_viol=4022，SAFEPOL/B6 SLO=0.08、0 违反（RQ6 故事）。

### 发现的问题（pilot1 SAFEPOL 模型已归档为 `*_s0.pilot1.*`）
1. **策略坍缩**：确定性评估下 99.5% 的步都是 `TIGHTEN_SC`——seccomp 已收紧后它是零变更的伪 NOOP，
   但 reward 的 `mal_drop/mal_total` 把被 seccomp 拦下的系统调用与网络流同权计数，所以它"看起来"一直在得分。
2. **掩码漏洞**（根因）：`ActionMask.k_mask` 只屏蔽 ROLLBACK/REMOVE；`w_mask` 在无可行主体时整体回退为 True，
   所以效果已耗尽的动作类型永远可选。**已修**：`k_mask` 现在按"是否存在可行主体"屏蔽（`verify/shield.py`）。
3. **Shield 被当作"加固宏"**：minimal 初始下智能体在 t=0 发 `ISOLATE`（必违反 φ6/φ8）→ MaxSAT 修复
   → 得到 25 条规则、AS=63 的策略，之后不再动作。这是合法行为（计入干预率与代价），但说明智能体本身学到的很少。
4. **S5 从未被遏制**：`RATE_LIMIT(frontend)` 从未被选中（12/12 个 S5 episode 全程 goal 命中）。这是 SAFEPOL ASR
   (0.128) 高于 B3 (0.073) 的唯一原因；其余场景 SAFEPOL 全部 ASR=0。
5. B3-responder 是很强的基线（playbook 直接用环境动作空间 + 告警特征）；其弱点是依赖过期的 G_dep：
   drift 组 inv_viol=2011、SLO=0.42。SAFEPOL 的卖点应是"ASR 持平或更低 + 0 违反 + drift 下可用性"。

### 决策
- 先修掩码，重训 SAFEPOL 60k（pilot-2）看坍缩是否消失、RATE_LIMIT 是否出现；再决定是否调 reward。
- 大规模实验（5 seeds × 150k × 全部方法与消融）必须等 pilot-2 通过后再启动。

## pilot-2（掩码修复后，60k，seed 0）—— 结果归档在 `results/v0/`
SAFEPOL：ASR=0.000、mitigated=1.00、FBR=0.0008、SLO=0.010、inv_viol=0、interv=0.006。动作分布健康
（NOOP 50%，RATE_LIMIT 覆盖 12 个工作负载，DENY_OUT/DROP_CAP/TIGHTEN_SC 各司其职），S5 全部遏制。
**但新增的 B7-hardened 基线（B6 + 开局对所有工作负载 TIGHTEN_SC/DROP_CAP/RATE_LIMIT，经 Shield，无学习）
拿到完全一样的数字**（ASR 0 / FBR 0.0007 / SLO 0.009 / 0 违反）。结论：v0 基准上"把不变量允许的一切加固到底"
的静态策略就是最优，RL 没有必要——不能用 v0 基准做论文主实验。

修过的 bug：
- B7 初版：allow-list 每 20 周期刷新时 `allowlist_policy()` 重建 endpoint，把加固抹掉（t=40 后 S5 得手）。
  已改为刷新时保留 endpoint 加固。
- `reset(options={"scenario": None})` 被当作"未指定"而随机采样攻击 → 评估集 **benign 组 90% 其实有攻击**。
  已改为显式 `"scenario" in options`。v0 的 benign 组数字因此不可信。
- Windows 侧 `Set-Content` 写的任务文件带 CRLF → tag 变成 `evaltest\r`。任务文件一律用 LF；`scripts/fix_crlf.sh` 可修。

## v1 基准：加入"过度加固有代价"的工作负载动态（2026-10-19 19:20）
论文论点是"静态策略部署时正确、运行后过时"，v0 基准里却没有任何让固定加固等级出错的动态。新增（`sim/replay_env.py`）：
1. **合法流量洪峰 flash crowd**（p=0.3）：Internet→frontend 合法速率 ×4–6、持续 5–10 周期。固定限速
   （q=50 → 250/周期）会丢合法请求 → FBR/SLO；自适应策略应只在攻击特征下限速，之后解除。
2. **合法特权作业**（p=0.35）：随机 app 工作负载在 8 个周期内执行需要 `mount/umount2` 的作业（如模型热加载）。
   收紧的 seccomp 会让作业失败（`legit_host_drop`，计入 SLO 与 reward 的 −w3 项；summary 新增 `FBR_host`）；
   自适应策略应 RELAX_SC 于作业窗口、告警时再 TIGHTEN_SC。φ7 保持"基线观测"语义不变。
3. `RATE_LIMIT` 改为 **toggle**（安装/解除），掩码恒可行；`edit_distance` 计入 rate_limit 变化。
4. seccomp 拦截计入 `drops` 特征（内核侧可观测）。
`tests/dynamics.py` 验证：固定限速在洪峰丢合法流量、解除后恢复；紧 seccomp 阻断作业、放宽后恢复。

框架上诚实的写法：**B7 在平稳工作负载下是最优的；它的代价出现在洪峰与合法特权作业窗口**。这就是 adaptivity 的价值所在，
需要在论文里明确报告 B7 并按 dynamics 切片（`summarize.py` 已加 by-dynamics 表）。

## pilot-3（v1 基准，60k，seed 0）—— 学习模型结果归档在 `results/v1a/`
- 静态基线（`results/evaltest/`，仍有效）：B7 ASR 0 / FBR 0.0033 / SLO 0.052；B3 ASR 0.079 / SLO 0.079 / inv_viol 2011；
  B6 ASR 0.125 / SLO 0.097。B7 在 job 切片 FBR_host=1.0、SLO 0.089（vs 平稳 0.0085）；flash 切片 SLO 0.049。
- SAFEPOL 退步到 ASR 0.144、S5 ASR=1.0：**RATE_LIMIT toggle 语义导致反复装/拆限速**（S5 中 12.7% 步是 RATE_LIMIT）。
  已改回：`RATE_LIMIT` 幂等安装；解除限速由 `REMOVE(w, *)`（"撤销智能体对 w 的改动"：先解除限速，否则删最近 agent 规则）承担。

## pilot-4（v1 基准 + REMOVE 解除限速，60k，seed 0）—— 当前 `results/evaltest/`
| 方法 | ASR | mitigated | TTM | FBR | SLO | inv_viol | actions/ep | return |
|---|---|---|---|---|---|---|---|---|
| B3-responder | 0.079 | 0.96 | 3.6 | 0.004 | 0.079 | 2011 | 7.7 | 53.7 |
| B6-shield-only | 0.125 | 0.875 | 0.0 | 0.006 | 0.097 | 0 | 0 | 28.5 |
| B7-hardened | **0.000** | 1.00 | 0.0 | 0.003 | 0.052 | 0 | 36 | 56.7 |
| SAFEPOL | 0.030 | 1.00 | 1.0 | **0.001** | **0.037** | 0 | 14 | 55.9 |
| B4-DQN-PSO | 0.591 | 0.54 | 7.3 | 0.009 | 0.117 | 7569 | 69 | – |
| B5-PPO | 0.625 | 0.38 | 0.0 | 0.009 | 0.125 | 2011 | 50 | – |

- SAFEPOL 现在在 FBR/SLO 上**优于 B7**，ASR 略差（S5 0.25：限速是攻击开始后才反应，TTM≈1s）。
- 切片：flash — SAFEPOL SLO 0.016 vs B7 0.049（自适应的价值体现）；stationary — SAFEPOL SLO 0.0015 vs B7 0.0085；
  job — SAFEPOL FBR_host 0.98（**尚未学会 RELAX_SC**，60k 步不够），SLO 0.080 vs B7 0.089。
- 动作分布：NOOP 87%，RATE_LIMIT 只在 S5 中大量出现（12%），TIGHTEN_SC 12%，t=0 一次 ISOLATE→修复宏仍存在。

## 待决策 / 下一步
1. 基准动态强度：flash（×4–6、5–10 周期）与 job（8 周期）对 B7 的 SLO 代价约 +0.04 / +0.08；若希望 adaptivity 的收益
   更明显，可加大（更长作业窗口、更多作业工作负载、洪峰更高），但要和威胁模型/论文叙述一致——需作者定夺。
2. 训练预算：60k 明显未收敛（RELAX_SC 未学到）；主实验建议 ≥150k–300k/seed。单 SAFEPOL 60k ≈ 5.5 min（单线程），
   5 seeds × 150k × (3 学习方法 + 9 消融) ≈ 数小时 CPU，可用 `run_jobs.sh` 并行 6。
3. 评估集 benign 组修复后需重新审视 v0 阶段的所有数字（已全部归档，不再引用）。
4. 论文写法：明确报告 B7，并按 dynamics 切片呈现；把"ISOLATE→Shield 修复宏"作为一个观察写进讨论。

## 2026-10-19 20:00– 用户决定：基准动态加大，其余按建议（≥150k/seed、5 seeds、显式报告 B7、dynamics 切片、写完整论文）

### v1d 基准（强动态）
`EnvConfig`: p_flash=0.4, flash_factor=(5,7), flash_duration=(10,20); p_job=0.5, jobs_per_episode=(1,2), job_duration=(10,15)，
多个作业窗口（`self.jobs` 列表）。静态基线（seed 0，`results/evaltest/`，有效）：
B0 ASR 1.0/SLO 0.076；B1 0.25/0.236/inv 4022；B2 0.125/0.236/4022；B3 0.072–0.084/0.105–0.139/2011；B6 0.125/0.155/0；
**B7 ASR 0.000 FBR 0.0109 SLO 0.130**，切片 SLO：stationary 0.009 / flash 0.105 / job 0.176 / flash+job 0.251 → B7 的代价清晰可见。

### 工程：增量可达 + 局部修复 + 可扩展性（已完成，`results/scalability.csv`，n=100/500/1000/5000，中位数 ms）
| 组件 | 100 | 500 | 1000 | 5000 |
|---|---|---|---|---|
| reach_full | 2.97 | 16.8 | 40.2 | 529 |
| reach_incremental | 0.23 | 1.61 | 4.87 | 222 |
| invariant_check | 0.18 | 0.95 | 2.10 | 13.6 |
| conflicts_global / local | 1.64/0.05 | 14.4/0.23 | 49.0/0.51 | 500/2.15 |
| verify_candidate | 0.17 | 0.95 | 2.27 | 11.3 |
| repair_maxsat_local | 62.8 | 56.8 | 46.2 | 376 |
| repair_maxsat_global | 529 | 4817 | 16414 | – |
| repair_greedy | 4.69 | 33.5 | 106 | 1649 |
| repair_pso | 2254 | 15053 | 42142(全部 infeasible) | – |
局部 MaxSAT 修复与 n 基本无关（变量数 ≈ 11–110），全局编码线性增长到不可用；PSO 在 n≥500 已找不到可行解。

### pilot-5…8：强动态下 PPO 学习性问题（200k，3 seeds，均归档 results/v1d–v1g）
- v1d（原 PPO 配置）：ASR 0.10–0.26，S1 0.61 / S7 0.78；动作 96% DENY_OUT（确定性 argmax 下的近均匀策略）。
- v1e：PPO 稳定化（ent 0.02、target_kl 0.03、lr 退火、cost_limit 0.10、λ_max 5）：ASR 0.11–0.18，S1 0.78 / S7 0.30。
- v1f：`danger_sc` 特征不再计合法作业的特权调用（过程感知检测语义）+ 新特征 `sc_denied`：S1 0.81 / S7 0.62。
- v1g：目标开放惩罚独立权重 w6=5（原与 w2 共用）：S1 0.61 / S7 0.56，S2–S4、S6、S8 全为 0。
- v1h：主体头改为实体化（pointer）评分：无改善。
- **诊断（`scripts/probe_policy.py`）：策略头几乎均匀（p(k) 最大 0.15，p(w|k)≈1/11），训练全程熵≈6 不下降，KL≈0.001。
  即策略根本没有锐化——不是"学不会 S1"，而是 PPO 更新几乎无效。** 手工脚本（`tests/host_response.py`）证实
  TIGHTEN_SC/DROP_CAP 作用于受害者可立即关闭 S1/S7 目标（ASR 0，TTM 5s），环境侧无误。
- 正在跑 `scripts/sweep.sh`：ent_coef ∈ {0, 0.003}、lr 1e-3、无 target_kl、epochs 10、γ 0.95（80k 步）。

### 2026-10-19 21:40–22:10 根因：PPO 实现的两个静默缺陷（`tests/ppo_sanity.py` 合成任务：reward=1[a=NOOP]）
合成任务下旧实现 16k 步内 p(NOOP) 几乎不动、熵 6.2→4.7；逐 minibatch 打印发现 **KL 早停在第 1 个 minibatch 之后就触发**（n_mb=1/迭代）。
1. **联合 log-prob 包含无关子动作**：NOOP/ROLLBACK 的 w 头、非网络动作的 p/π 头对结果无影响却进入 logp 与熵，
   PPO ratio 被这些噪声主导，approx_kl 在一次更新后即 >0.045。修复：`REL_W/REL_P/REL_PI` 相关性掩码——无关头不计 logp/熵
   （参数化动作空间惯例）；`w_mask` 对 NOOP/ROLLBACK 只留一个选择。
2. **critic 主导的全局梯度裁剪**：vf loss 数百（回报未归一化）→ 全局 clip_grad_norm 把 actor 梯度压到几乎为 0。
   修复：reward/cost 按折扣回报的运行 std 缩放（critic 目标 O(1)）、actor 与 critic 分别裁剪、优势分别归一化后再做拉格朗日组合。
修复后合成任务：p(NOOP) 0.09→0.62（8 次迭代，熵 5.7→3.1）。其他尝试（ent_coef 0/0.003、lr 1e-3、无 KL 早停、epochs 10、γ 0.95、
实体化主体头）均无效——因为问题不在超参。rollout 2048→1024，ent_coef 0.02→0.01。

### pilot-9（修复后 PPO，200k，3 seeds）—— 归档 `results/v1i/`
SAFEPOL ASR 0.091/0.244/0.244，SLO 0.087/0.039/0.031，interv 0.007–0.014；λ 下降到 0.40（约束满足，mean_cost 0.05–0.09 < 0.10）。
**S2–S6、S8 全为 0，但 S1 0.80 / S7 0.74**：策略在确定性下坍缩到 DENY_OUT 64% + DENY_IN 20%（全场景一致）。
原因推断：课程阶段 0–1 只有网络类场景，策略先对 DENY_* 收敛，后期熵已降到 ~5，host 类响应（TIGHTEN_SC/DROP_CAP）再难被探索到。
B5-PPO（无 Shield/成本/掩码）在修复后 ASR 0.166（S1 0.025、S7 0.05！）但 inv_viol 4022、SLO 0.172。
→ pilot-10：课程每阶段同时含 S1 与 S2（host+network），lr 下限 0.25。

### pilot-10（课程每阶段含 S1+S2，lr 下限 0.25）—— 归档 `results/v1j/`
更差：ASR 0.30/0.62/0.18，S7 0.975；DENY_OUT 占 82–89%；训练全程熵≈6 不降。结论：修复后的 PPO 在合成任务上能学，
但在真实奖励下 200k 步内信号太弱/太噪（正确 host 响应 = (TIGHTEN_SC, 受害者) 随机命中率 ≈0.7%/步），策略无法锐化。

### 设计决策：runbook 热启动（behaviour cloning from the operator playbook = B3）
动机：运营方本来就有 runbook（Falco/Tetragon 响应剧本）；把它当示教数据初始化策略，再用受约束 RL 在 Shield 下改进，
既合理又可解释 "SAFEPOL ≥ B3"。实现 `PPOLagrangian.warm_start(teacher, n_steps, epochs)`：在智能体自己的环境里（Shield 仍然
校验每个被执行的教师动作）收集 (obs, action, masks)，只对相关头做掩码交叉熵；**类别平衡**（runbook >95% 步为 NOOP，
响应样本过采样到一半）；flat 头有对应实现。`warm_start_steps=20_000`（≈1.3k 条响应示教，17 s）。
纯 BC 策略（不经 RL）评估：ASR 0.175、FBR 0.0039、SLO 0.066、TIGHTEN_SC/DROP_CAP 开始出现在 S1/S7。
所有 PPO 方法（含 B5、消融）同样热启动以保证公平；新增消融 **A-noWarmStart**。
→ pilot-11 运行中。

### pilot-11（runbook 热启动 + 修复后 PPO-Lagrangian，200k，3 seeds）—— **采用为最终配置**，保留在 `results/evaltest/`
SAFEPOL ASR 0.047/0.113/0.169（均 0.11），mitigated 1.00，TTM 2.7–6.3 s，FBR 0.004–0.006，SLO 0.073–0.090，inv_viol 0，interv 0.002–0.008。
S1 0.32 / S7 0.27 / S2 0 / S3 0.008 / S4 0 / S5 0.033 / S6 0.117 / S8 0.133。动作分布终于随场景变化（TIGHTEN_SC 在 S1/S7 为 12%，其他 3%）。
动态切片 SLO：SAFEPOL flash 0.059 / flash+job 0.169 / job 0.118 / stationary 0.0018，
B7 0.105 / 0.251 / 0.176 / 0.009 —— SAFEPOL 在每个切片都最低。drift 组 SLO：SAFEPOL 0.091，B3 0.445，B5 0.77，B7 0.147。
B3 ASR 0.072 仍低于 SAFEPOL（诚实报告：runbook 在已知场景上精确；SAFEPOL 的收益在可用性/漂移/保证）。
B5-PPO（热启动，无 Shield）ASR 0.203、inv_viol 3934、drift SLO 0.77。
22:50 启动 main-B：`scripts/jobs_main_b_train.txt`（60 训练）+ `jobs_main_b_eval.txt`（84 评估），并行 6/5，预计 ~1.7 h。

### 2026-10-20 01:37 main-B 完成（results/evaltest/，90 个 CSV；A-flat 因 w_mask 长度 bug 失败，修复后 01:40 重跑中）
主表（5 seeds）：SAFEPOL ASR 12.7±4.5 / mitigated 100 / TTM 5.0 s / FBR 0.5 / FBR_host 73 / SLO 8.8±0.9 / inv 0 / interv 1.0。
B3 ASR 7.9 SLO 10.5 inv 2011；B5 ASR 17.3 SLO 14.5 inv 3683；B4 ASR 41.5 SLO 22.8 inv 19568；B7 ASR 0 SLO 13.0 inv 0。
动态切片 SLO（stationary/flash/job/flash+job/drift）：SAFEPOL 0.2/6.3/12.6/17.6/9.8；B7 0.9/10.5/17.6/25.1/14.7；B3 3.9/11.0/11.9/17.4/44.5。
Wilcoxon(Holm)：SAFEPOL SLO 显著低于所有基线（B5 例外 p=0.028 δ=+0.04：B5 benign 更低但 drift 72%）；ASR 显著高于 B3/B7。
消融：noWarmStart ASR 41（S1 65/S7 98/S5 60）；noShield ASR 22 inv 2848 drift SLO 59；noCost ASR 8.8 但 interv 12±15；
detector ASR 38 SLO 16.4；PSOrepair interv 5.4 SLO 13.7；greedy ASR 16.2；noMask 18.2；noGraph 12.1(p=.26)；noCurriculum 14.6。
诚实点：B3 在 S1/S7 优于 SAFEPOL（RL 阶段部分遗忘 runbook：熵 0.9→3.7）；B7 stationary 不可战胜；A-noCost ASR 更低但靠修复。
论文已写完 abstract/intro/overview/design(+training)/implementation/evaluation(RQ1-7)/discussion/related/conclusion/appendix；
表由 scripts/stats.py 输出到 results/tables/*.tex，图由 scripts/figures.py 输出到 paper/figures/。

### 2026-10-20 02:10 收尾
A-flat 重跑完成并入表：ASR 19.0±7.3，mitigated 89，S1 40/S7 63（扁平头无法跨元组共享示教）。
论文全部章节完成；stats.py 同时把 .tex 表镜像到 paper/tables/；figures.py 同时输出 PNG；
scripts/texcheck.py 静态检查（环境/花括号/$ 配对/未转义 %）通过；scripts/reproduce.sh 一键复现；README.md。
未做（无 LaTeX 编译器）：pdflatex 编译验证。论文中的数字均来自 results/tables 与 scripts/extra_stats.py 输出。

### 2026-10-20 12:30 pdflatex 编译验证（MiKTeX）
首次编译 "Misplaced \noalign"：\input 的表体文件以 "\\" 结尾、\bottomrule 在外层时 TeX 不在行首——改为 stats.py 把 \bottomrule 写进生成的表文件。
修掉 5 处宽 overfull（动作类型集合改为文字列举、reward 公式折三行、场景表改 p 列、消融表改 table* 并精简数字格式、显著性表 footnotesize）。
最终：pdflatex ×2 + bibtex + pdflatex ×2 → main.pdf 12 页，无错误、无未定义引用/文献，仅 1 pt overfull 与标题 bold-smallcaps 字体替换警告。
3 张 PDF 图全部嵌入（dynamics/training/scalability），7 表 4 图编号正常。

### 2026-10-20 13:40 参考文献更新
删除不存在的占位文献 [4] prev（"our earlier work"），正文改为 "a DQN over policy bundles" 的中性表述。
文献由 15 条扩到 35 条，新增均经 Crossref API 核对（标题/作者/会议/年份/DOI）：
C2C (CCS'22)、SysPart (CCS'23)、Confine 期刊版 (C&S'23)、Kano (TNSM'23)、Rahman K8s manifests (TOSEM'23)、
Hammar&Stadler (TNSM'22, DSN'24)、Nguyen&Reddi (TNNLS'23)、Safety Gymnasium (NeurIPS'23)、OmniSafe (JMLR'24)、
Gu safe RL review (TPAMI'24)、PID-Lagrangian (ICML'20)、Probabilistic logic shields (IJCAI'23)、Shields CACM'25、
Huang&Ontañón masking (FLAIRS'22)、action branching (AAAI'18)、Engstrom (ICLR'20)、Andrychowicz (ICLR'21)、Jump-start RL (ICML'23)、
Tetragon/Cilium/Falco 项目页。相关工作章节重写为 5 段并与引言/背景/设计/讨论对应加引。
编译：14 页，无错误、无未定义引用。

### 2026-10-20 14:55 加入前期工作引用
wang2026dynamic：Wang Ben, Chen Hongchu, "Dynamic security policy generation and evaluation algorithm for trusted containers based on AI",
Ain Shams Eng. J. 17(1):103902, 2026, doi 10.1016/j.asej.2025.103902（PII S2090447925006434，经 Crossref + PDF 首页核对）。
引言/评估（RQ 列表、B4 定义、RQ4、RQ5）/相关工作恢复 "our earlier work" 表述并引用；相关工作段落概述其 DQN+PSO+回归评估设计。
事故：PowerShell 哈希表单元素数组被展平，导致 intro.tex 全部 a→g、related.tex 全部 \→e；
已按原文手工重写两文件，并用上一版 PDF 词表（scripts/texwords.py）核对其余章节无损。编译 14 页无错误，共 36 条文献。

### 2026-10-21 15:35 投稿前补充实验：B8 shielded runbook 与 SAFEPOL-H（hybrid）
B8 = B3RuleResponder + Shield（methods.py 新增，仅评估，5 seeds）：ASR 7.9 / mitig 96.2 / TTM 3.7 / FBR 0.4 / FBR_host 51 / SLO 6.2 / inv 0 / interv 0.2 / changes 7。
动态切片 SLO：0.0 / 4.6 / 8.6 / 12.6 / drift 7.5 —— 全面优于 SAFEPOL（0.2/6.3/12.6/17.6/9.8）。
SAFEPOL-H = runbook 先手 + 受约束 RL 其余步（EnvConfig.runbook_prior，训练 200k × 5 seeds）：ASR 8.3±1.2 / mitig 97.2 / SLO 7.0 / changes 61 —— 与 B8 持平略差，RL 残差未带来收益。
结论：在本基准上 RL 层不优于"Shield + runbook"；SAFEPOL 仅在 mitigated（100 vs 96.2）与 S2/S3 上更好，代价是 79 次/集的策略变更。
Shield 本身是最大贡献：B3→B8 inv 2011→0 且 SLO 10.5→6.2、drift 44.5→7.5。待用户决定论文重构方向。
IEEEtran 未安装（MiKTeX 自动安装挂起、无法在线获取），待处理。

### 2026-10-21 16:30 投稿准备（目标 IEEE TNSM）
1. 论文重构为"Shield 为核心"（用户选 A）：标题改为 "SAFEPOL: Verify-or-Repair Shielding for Adaptive Container Security Policies"；
   摘要/引言/概览/评估/讨论/结论重写；评估 RQ 重排：RQ1 = Shield 对三类控制器的作用（新表 tab:shield，由 stats.py 生成 shield_effect.tex：
   allow-list B2→B6、runbook B3→B8、learner A-noShield→SAFEPOL），RQ4 = "学习是否带来增益"（负结果：B8 优于 SAFEPOL 与 SAFEPOL-H）。
   方法排序改为 B0,B1,B2,B6,B7,B3,B8,B4,B5,SAFEPOL,SAFEPOL-H（stats.py ORDER、figures.py MAIN）。
2. 模板换为 IEEEtran journal（CTAN 下载 V1.8b 的 cls/bst 放 paper/；MiKTeX pdflatex 内置自动安装会挂起，改用 `mpm --install` 装 urw-base35/cm-super，
   initexmf --set-config-value [MPM]AutoInstall=0）。架构图改为 TikZ figure*（替换 fbox 占位）。\paragraph 标题去句号（IEEE 风格自带冒号）。
3. 引理（单调掩码）补证明，含选择器语义的例外（allow 规则加到未被选中的工作负载会使其 default-deny，此时全量验证）；定理证明展开为三出口归纳，
   指出保证仅依赖第 8 行的全量复查，求解器/编码 bug 只会导致拒绝。
4. 讨论新增 "What the shield is for"、"Why the learned controller did not beat the shielded runbook"、Ethical considerations。
5. Kano 作者修正为 Crossref 记录（Li Yifan, Hu Xiaohe, Jia Chengjun, Wang Kai, Li Jun; 20(3):3747–3764）。
6. README/reproduce.sh/requirements.txt/.gitignore 更新（v0、v1b–j、sweep、models 不入库；v1a 为附录弱动态数据来源）。
编译：13 页，36 条文献，无 overfull、无未定义引用；剩 T1/ptm/m/scit 字体替换警告（Times 无斜体小型大写，无害）。

### 2026-10-23 11:35 中文版论文（paper_zh/）
ctexart + xelatex（mpm 装 ctex/xecjk/zhnumber 等；系统 SimSun/SimHei）。全部章节逐段翻译自 Shield 为核心的英文稿（IEEEtran 版），
表格数据直接 \input ../paper/tables/，图直接引用 ../paper/figures/，文献共用 ../paper/refs.bib（IEEEtran.bst）。
架构图 TikZ 标签中文化；三张窄表用 \resizebox 适配 A4 双栏。编译 16 页、36 条文献、无 overfull/未定义引用。
事故：PowerShell Get-Content -Raw 按 GBK 读 UTF-8 文件后回写，evaluation.tex 乱码；已用 write 工具重写，其后改用 [IO.File]::ReadAllText(UTF8)。

### 2026-10-23 12:05 研究方案 v2（proposal/）
按任务书（多约束关键任务场景下大规模容器安全性与确定性保障技术）写研究方案 v1，再据近期文献细化为 v2。
文献检索：web_search 插件不可用，改用 Crossref API（proposal/litscan.py 主题扫描 + litverify.py 按标题逐条核验 + litdoi.py）；
50 条文献全部经 Crossref 核验（猜 DOI 的方式不可靠，多个命中为无关论文，已弃用）。v2 新增 1.2 研究现状（6 小节）、
各研究内容关键技术与已有工作的对接（LiME 模型抽取、SPH 评测/HSP-V、REMAP、MemGuard/MemPol、RDT 局限、BARO/RCAEval、故障注入分析等）、
风险表增补 Hypervisor 选型风险，参考文献 50 条（引用编号经脚本校验无缺漏）。

### 2026-10-23 12:45 专利交底书（Shield 为核心）
按 patent-disclosure-skill 流程：intake（发明 / 作者信息 / Shield 核心）→ 国知局公布站查新 3 轮（容器安全策略/网络策略验证/策略冲突修复/可达性验证 + H04L41/G06F21 收口 + 策略自动修复/不变量验证）→
D1=CN118282881A（瞻博《网络策略验证》，只验证不修复），D2 补行 CN119814567A（意图冲突解决）、CN118152068A（容器多级缓解执行）；门禁「带疑点过」（摘要级）。
区别特征 F1 不变量校验 / F2 MaxSAT 最小修复 / F3 局部化编码+三级退化 / F4 索引化精确冲突分析 / F5 增量可达性（优选）/ F6 控制器无关+干预反馈 / F7 网络+端点联合校验。
formula_plan 6 式全部 source（design.tex），check 通过；成文 → outputs/safepol_shield/一种容器集群安全策略的校验与修复方法及系统_20261023124201.{md,docx}，mermaid 2 图 PNG，OMML 82/82。
正文脱敏：不出现 Online Boutique/Cilium/Tetragon，实验数值用范围。RL 控制器只作实施例三与 5.2 优选。

### 2026-10-23 14:40 真实集群验证（RQ7，paper Sec. k8s）
环境：Docker Desktop k8s 1.34.3（单节点 containerd）+ Cilium 1.17.3（Hubble、Tetragon 已在 kube-system）；gcr.io 不可达，镜像经 DaoCloud 镜像站拉取。ns boutique 部署上游 Online Boutique v0.10.1 + promoservice（busybox httpd）+ probe pod，15 pod Running。
适配层 safepol/k8s/：translate.py（Policy→每工作负载一个 CiliumNetworkPolicy：ingress/egress + ingressDeny/egressDeny，外部实体 world/host,remote-node/kube-apiserver、Metadata CIDR，egress 加 kube-dns:53 豁免；seccomp profile/securityContext；NetworkPolicy→模型解析），cluster.py（kubectl/apply/debug 容器/nc 探测/Hubble 读取）。
(a) 上游 13 条 NetworkPolicy → 25 规则，违反 φ2（全部可达 metadata）、φ4（adservice 无 seccomp）共 13 处；drift 下 +φ6/φ8 共 17。verify 0.3 ms，MaxSAT 修复 19/21 ms，仅新增 12/15 条规则、不删规则。修复后策略下发集群，209 条探测流（43 允/166 拒）与模型全部一致。
(b) 单条 egressDeny 切换 20 次（pod 内 python 轮询 10 ms）：apply RTT 88 ms，生效 10 ms（p95 17，max 18），删除后重开 88 ms。seccomp/caps 变更 = pod 重建，Ready 2.7 s（各 5 次），CapBnd a80425fb→a0000531。
(c) 端到端 B3 vs B8，S2/S3 各 3 回合，Δ=5 s，攻击者 = 受害 pod 内 netshoot 调试容器 nc/curl（无利用）。两者均首周期告警、5.5 s 关闭目标（模型与集群探测一致）。B3 通配 deny 把节点也拒了 → kubelet 探针被丢，pod NotReady、liveness 重启 1–3 次，S2 时 frontend 全程 500。B8 以 φ6（NodeCIDR→victim 在 Hubble 观测窗口内）拦截，距离 1 修复 3–8 ms，受害者保持 Ready、0 丢流、frontend 100% 200。
发现：单节点上 host 身份同时带 reserved:kube-apiserver 标签，deny APIServer 等于 deny kubelet；translate.py 加 apiserver_is_host 检测，拒绝生成"含 APIServer 不含 NodeCIDR"的 deny 并通过 notes 上报。
MaxSAT 次级目标（倾向新增而非删除）改为 prefer_additions 可选项，仅集群脚本启用；验证开启时会改变模拟器结果（B8 24/408 回合 rules_final 不同），故模拟器数据保持原纯编辑距离，论文脚注说明。
论文：新增 RQ7 小节 + tab:k8s（tables/k8s_e2e.tex 由 scripts/k8s/summarise.py 生成），RQ 列表 8 个，摘要/引言贡献/实现/讨论 threats & ethics/结论更新；14 页，无 overfull。

### 2026-10-23 15:00 收尾
6 个测试脚本（smoke/incremental/local_repair/host_response/dynamics/logp_consistency）在 prefer_additions 改动后全部通过。中文版 paper_zh 同步 RQ7 小节、RQ 列表、摘要/引言/实现/讨论/结论（17 页，xelatex 无 overfull）。正文数字与 results/k8s/summary.txt 逐项核对一致；适配层行数按实际（translate 247 + cluster 122）改为 370 行。


### 2026-10-23 15:05 仓库
仓库地址 https://github.com/wbe3191/safepol 写入论文（引言贡献脚注、实现、结论；中英文版）与 README。本地 git init（main），一次性脚本移入 scripts/dev/ 并忽略，专利交底书/研究方案/ASEJ PDF/模型 checkpoint 不入库；首个提交 371 个文件，45.7 MB（evaltest 42 MB）。本机当前无法连接 github.com:443（代理 127.0.0.1:8887 未运行），尚未 push。


### 2026-10-23 15:35 LICENSE 与论文源码
加 Apache-2.0 LICENSE（版权 Ben Wang, Hongchu Chen）。paper/ 与 paper_zh/ 加入 .gitignore，稿件源码在录用前不入公开仓库；README 相应更新（stats/figures 仍按需写 paper/tables、paper/figures）。

