cd /mnt/e/dshWorkspace/SAFEPOL
# main-B: everything not covered by main-a (SAFEPOL/B5/B4 s0-2 + static s0 already exist)
{
for s in 3 4; do echo "train SAFEPOL $s 200000"; done
for s in 1 2 3 4; do echo "train B5-PPO $s 200000"; echo "train B4-DQN-PSO $s 200000"; done
for m in A-noWarmStart A-noShield A-noCost A-noMask A-flat A-noGraph A-noCurriculum A-PSOrepair A-greedyRepair A-detector; do
  for s in 0 1 2 3 4; do echo "train $m $s 200000"; done
done
} > scripts/jobs_main_b_train.txt
{
for s in 3 4; do echo "eval SAFEPOL $s 0 evaltest"; done
for s in 1 2 3 4; do echo "eval B5-PPO $s 0 evaltest"; echo "eval B4-DQN-PSO $s 0 evaltest"; done
for m in A-noWarmStart A-noShield A-noCost A-noMask A-flat A-noGraph A-noCurriculum A-PSOrepair A-greedyRepair A-detector; do
  for s in 0 1 2 3 4; do echo "eval $m $s 0 evaltest"; done
done
for m in B0-none B1-static B2-observed B3-responder B6-shield-only B7-hardened; do
  for s in 1 2 3 4; do echo "eval $m $s 0 evaltest"; done
done
} > scripts/jobs_main_b_eval.txt
wc -l scripts/jobs_main_b_train.txt scripts/jobs_main_b_eval.txt
