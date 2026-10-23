import sys, pandas as pd
pd.set_option("display.width", 200)
for f in sys.argv[1:]:
    t = pd.read_csv(f)
    cols = [c for c in ["steps", "lambda", "mean_cost", "ep_return", "ASR", "pg", "vf", "ent", "kl"] if c in t]
    print("---", f)
    print(t[cols].iloc[::10].round(3).to_string())
