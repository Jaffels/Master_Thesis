"""RQ2 step 3 (todo 9.2d): linear benchmark (ridge regression) with impute=True.

Same series, targets, rolling-origin folds and delta target as rq2_2_lightgbm.py:
  target = asinh(price) - asinh(same slot, previous day / week); prediction = sinh(offset + ridge).
Views are loaded with impute=True (interior gaps <= 24 h linearly interpolated, todo 5.5).
Features: strict ex-ante set (views.feature_columns), minus cal_year and the *_imputed flags;
categorical columns one-hot encoded; per fold: columns standardised on the training window,
clipped at +-5 SD, remaining NaN set to the training mean (= 0), constant columns dropped.
Penalty alpha chosen from ALPHAS on the last 15 % (in time) of the training window, then refit.
Feature sets:  ridge_all   all strict ex-ante features
               ridge_core  lagged targets + calendar + regimes only (small linear model)
The report adds a comparison with LightGBM (preds from Output/rq2_2_delta/preds_exante.parquet)
on the common rows, with Diebold-Mariano tests (stat < 0: first model better).

    python Modelling/rq2_3_linear.py [--products mfrr] [--only "mFRR|up|4h"] [--sets ridge_all ridge_core]
Output: Modelling/Output/rq2_3/{preds_<set>.parquet, metrics.csv, report_rq2_3.txt}
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
import warnings
import rq2_common as C
warnings.filterwarnings("ignore", category=RuntimeWarning)

ALPHAS = [10.0, 100.0, 1e3, 1e4, 1e5]
DROP = {"cal_year"}
SETS = ["ridge_all", "ridge_core"]


def design(df: pd.DataFrame, view: str, fset: str):
    feats = [c for c in C.V.feature_columns(df, view, "exante") if c not in DROP and not c.endswith("_imputed")]
    if fset == "ridge_core":
        feats = [c for c in feats if c.startswith(("tgt_prev", "cal_", "regime_"))]
    X = df[feats].copy()
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c].dtype) and not pd.api.types.is_bool_dtype(X[c].dtype)]
    if cat:
        X = pd.get_dummies(X, columns=cat, dummy_na=False, dtype="float32")
    for c in ["cal_hour_local", "cal_dow", "cal_month"]:              # periodic / categorical calendar -> dummies
        if c in X.columns:
            X = pd.concat([X.drop(columns=c), pd.get_dummies(X[c], prefix=c, dtype="float32")], axis=1)
    return X.astype("float64")


def prep(Xtr: np.ndarray, Xte: np.ndarray):
    mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0)
    keep = np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-9)
    f = lambda A: np.nan_to_num(np.clip((A[:, keep] - mu[keep]) / sd[keep], -5, 5), nan=0.0)
    return f(Xtr), f(Xte)


def ridge(A, y, alpha):
    ym = y.mean()
    G = A.T @ A + alpha * np.eye(A.shape[1])
    b = np.linalg.solve(G, A.T @ (y - ym))
    return b, ym


def fit_predict(Xtr, ytr, Xte):
    A, B = prep(Xtr, Xte)
    nv = max(10, int(0.15 * len(A)))
    best, besta = np.inf, ALPHAS[0]
    for al in ALPHAS:
        b, ym = ridge(A[:-nv], ytr[:-nv], al)
        e = np.abs(A[-nv:] @ b + ym - ytr[-nv:]).mean()
        if e < best: best, besta = e, al
    b, ym = ridge(A, ytr, besta)
    return B @ b + ym, besta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=C.VIEWS)
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    ap.add_argument("--only", default=None)
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--refit-months-week", type=int, default=12)
    ap.add_argument("--out", default=str(C.ROOT / "Modelling" / "Output" / "rq2_3"))
    ap.add_argument("--lgbm", default=str(C.ROOT / "Modelling" / "Output" / "rq2_2_delta" / "preds_exante.parquet"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = ["RQ2 9.2d - ridge benchmark (impute=True), delta target, rolling origin (same folds as LightGBM)", ""]
    store = {s: [] for s in a.sets}
    for view in a.products:
        t0 = time.time()
        df = C.load_view(view, impute=True)
        print(f"{view}: loaded + imputed in {time.time()-t0:.0f}s", flush=True)
        keys = ["series", "auction_id", "block_start_utc", "y", *C.BASELINES]
        for fset in a.sets:
            X = design(df, view, fset)
            for s in df["series"].unique():
                if a.only and a.only != s: continue
                gi = np.flatnonzero((df["series"] == s).to_numpy()); g = df.iloc[gi]
                Xg = X.iloc[gi].to_numpy()
                ya = np.arcsinh(g["y"].to_numpy(float)); off = np.arcsinh(g["b_prev_slot"].to_numpy(float))
                yt = ya - off; ok = ~np.isnan(yt)
                rm = a.refit_months_week if s.endswith("|week") else a.refit_months
                mt = C.MIN_TRAIN[s.split("|")[2]]
                t1, preds = time.time(), []
                for f, start, tr, te in C.fold_plan(g.reset_index(drop=True), rm, mt):
                    tr = tr[ok[tr]]
                    if len(tr) < mt: continue
                    p, al = fit_predict(Xg[tr], yt[tr], Xg[te])
                    r = g.iloc[te][keys].copy(); r["pred"] = np.sinh(p + off[te]); r["fold"] = f; r["alpha"] = al; r["set"] = fset
                    preds.append(r)
                print(f"{view} {fset:11s} {s:16s} folds={len(preds):3d} {time.time()-t1:6.1f}s", flush=True)
                if preds: store[fset].append(pd.concat(preds))
    allp = {k: pd.concat(v) for k, v in store.items() if v}
    if not allp: print("nothing evaluated"); return
    lg = pd.read_parquet(a.lgbm)[["series", "auction_id", "block_start_utc", "pred"]].rename(columns={"pred": "lgbm"}) \
        if Path(a.lgbm).exists() else None
    rows = []
    for fset, P in allp.items():
        P.to_parquet(out / f"preds_{fset}.parquet")
        for s, g in P.groupby("series", sort=False):
            g = g.dropna(subset=["y", "pred", *C.BASELINES]).sort_values("block_start_utc")
            if lg is not None:
                g = g.merge(lg, on=["series", "auction_id", "block_start_utc"], how="left")
            else:
                g["lgbm"] = np.nan
            h = g.dropna(subset=["lgbm"])
            L = 4 if s.endswith("|week") else 12
            r = {"series": s, "set": fset, "n": len(g), "MAE_ridge": C.metrics(g.y, g.pred)["MAE"],
                 "RMSE_ridge": C.metrics(g.y, g.pred)["RMSE"], "MAE_naive": C.metrics(g.y, g.b_prev_slot)["MAE"]}
            r["skill_vs_naive_%"] = (1 - r["MAE_ridge"] / r["MAE_naive"]) * 100
            r["DM_vs_naive_p"] = C.dm_test(g.y, g.pred, g.b_prev_slot, L)[1]
            if len(h):
                r["n_common_lgbm"] = len(h); r["MAE_ridge_common"] = C.metrics(h.y, h.pred)["MAE"]
                r["MAE_lgbm_common"] = C.metrics(h.y, h.lgbm)["MAE"]
                st, p = C.dm_test(h.y, h.lgbm, h.pred, L); r["DM_lgbm_vs_ridge_stat"], r["DM_lgbm_vs_ridge_p"] = st, p
            rows.append(r)
    M = pd.DataFrame(rows); M.to_csv(out / "metrics.csv", index=False)
    lines.append("MAE in EUR/MWh on the common rows of the model and all baselines (skill = 1 - MAE_ridge / MAE_naive).")
    lines.append("DM_lgbm_vs_ridge: stat < 0 = LightGBM more accurate than ridge (same rows).")
    lines.append(C.fmt(M.set_index(["series", "set"])))
    (out / "report_rq2_3.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
