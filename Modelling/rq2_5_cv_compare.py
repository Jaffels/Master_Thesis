"""Compare default / holdout-tuned (rq2_5t_*) / CV-tuned (rq2_5cv_*) Tier 1 models on the small RQ2 series.
MAE skill vs same-slot-yesterday and DM tests (|error|) on common rows.
    python Modelling/rq2_5_cv_compare.py Modelling/Output/rq2_5cv_compare.csv   (run from Master_Thesis)"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, rq2_common as C
import os; OUTF = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output", "rq2_5cv_compare.csv"); os.chdir(os.path.dirname(os.path.abspath(__file__))); O = "Output/"
runs = {"default": {"aFRR": "rq2_5_afrr", "mFRR": "rq2_5_mfrr"},
        "holdout": {"aFRR": "rq2_5t_afrr", "mFRR": "rq2_5t_mfrr"},
        "cv": {"aFRR": "rq2_5cv_afrr", "mFRR|up|week": "rq2_5cv_mfrr_upw", "mFRR|down|week": "rq2_5cv_mfrr_dnw"}}
SER = ["aFRR|up|4h", "aFRR|down|4h", "aFRR|up|week", "aFRR|down|week", "mFRR|up|week", "mFRR|down|week"]
M = ["lear", "lgbm_q", "cat_q", "qrf", "svr"]
def load(run, s):
    d = runs[run]; f = d.get(s) or d[s.split("|")[0]]
    p = pd.read_parquet(O + f + "/preds.parquet"); return p[p.series == s].set_index("block_start_utc")
rows = []
for s in SER:
    P = {r: load(r, s) for r in runs}
    idx = P["default"].index
    for r in runs: idx = idx.intersection(P[r].index)
    y = P["default"].loc[idx, "y"]; nv = P["default"].loc[idx, "b_prev_slot"]
    ok = y.notna() & nv.notna()
    for m in M:
        c = f"p_{m}"; ok2 = ok.copy()
        for r in runs: ok2 &= P[r].loc[idx, c].notna()
        yy, nn = y[ok2], nv[ok2]; mae_n = (nn - yy).abs().mean()
        row = {"series": s, "model": m, "n": int(ok2.sum())}
        for r in runs:
            row[f"skill_{r}"] = (1 - (P[r].loc[idx, c][ok2] - yy).abs().mean() / mae_n) * 100
        L = 4 if s.endswith("week") else 12
        st, p = C.dm_test(yy, P["cv"].loc[idx, c][ok2], P["holdout"].loc[idx, c][ok2], L); row["p_cv_vs_holdout"] = p; row["cv_better"] = st < 0
        st, p = C.dm_test(yy, P["cv"].loc[idx, c][ok2], P["default"].loc[idx, c][ok2], L); row["p_cv_vs_default"] = p; row["cv_better_def"] = st < 0
        rows.append(row)
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
R.to_csv(OUTF, index=False)
