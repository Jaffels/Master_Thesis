"""RQ1 extension (done_todo 2.5 item 15): SHAP values of the RQ2 tree models -> which drivers move Swiss
balancing-capacity prices, in which direction, and how stable that is over time.

Models (refitted exactly as in RQ2; the fit is checked against the stored RQ2 predictions):
  lgbm   LightGBM L1, round-1 main model (rq2_2_lightgbm.py, --mode delta defaults)   check vs Output/rq2_2_delta
  cat    CatBoost MultiQuantile, median output (rq2_5_tier1.py defaults, seed 1)     check vs Output/rq2_5_<view>
Same series, ex-ante features, rolling-origin folds (expanding window, refit every 3 months / weekly 12) and
delta target asinh(y) - asinh(same slot yesterday). SHAP = exact TreeSHAP (LightGBM pred_contrib, CatBoost
ShapValues), computed OUT OF SAMPLE on every test fold with that fold's model -> one SHAP row per test block
(about 35k blocks). No shap package needed.

Reading the numbers:
  * SHAP is on the model scale = change of asinh(price) vs the same slot yesterday. "lagged target" features
    therefore measure how much the model departs from persistence because of the price history, not the
    persistence itself (which sits in the offset). mean_abs_shap_eur = |SHAP| x cosh(prediction) ~ EUR/MWh effect.
  * direction = Spearman correlation between the feature value and its SHAP value (+ = higher value -> higher
    price forecast); categorical features use their category codes (direction not meaningful -> flagged).
  * families = rq2_2_lightgbm.family (calendar, lagged target, cross-market DE prices, published forecasts,
    outages, lookback, weather, regime / source).
  * TFT comparison (mFRR up 4h): variable-selection weights already saved by rq2_6 (importance_tft.csv /
    importance_tft_past.csv) aggregated to the same families.

Run from Master_Thesis with .venv active:
    python Modelling/rq1c_shap.py --only "mFRR|up|week" --out Modelling/Output/rq1c_shap_quick   # check, ~1 min
    python Modelling/rq1c_shap.py                                                              # all 9 series
Rough runtime on the MacBook Air: aFRR + weekly series a few minutes, FCR ~15-25 min, mFRR 4h ~30-50 min
(CatBoost dominates; --models lgbm for a fast LightGBM-only run). Read-only on Clean/Data and Modelling/Output/rq2_*.
Outputs (Modelling/Output/rq1c_shap/):
  report_rq1c.txt              summary (check vs RQ2, family shares, top features with direction, periods, agreement, TFT)
  shap_importance.csv          series x model x feature: mean |SHAP|, share, EUR effect, mean SHAP, direction, gain share
  shap_family.csv              series x model x family share of mean |SHAP|
  shap_family_by_year.csv / shap_family_by_period.csv   the same per year / per period (pre-crisis, crisis,
                               post-crisis, post-merger >= 28 Sep 2025)
  shap_dependence.csv          top-10 features per series x model: feature-value deciles -> mean SHAP
  model_agreement.csv          rank correlation lgbm vs cat SHAP, SHAP vs gain, top-20 overlap
  tft_vs_shap_family.csv       TFT selection weights vs SHAP by family (mFRR up 4h)
  fit_check.csv                max / mean |refit - stored RQ2 prediction| per series x model
  shap_values_<model>.parquet  per-block SHAP of the top --top-store features (+ feature values) for own plots
  plots/<series>/              bar_<model>.png (top 20, coloured by family), beeswarm_<model>.png, dependence_<model>.png,
                               family_by_year_<model>.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification
import lightgbm as lgb
import rq2_common as C
from rq2_2_lightgbm import family, feature_list, fit_predict, make_X

QS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
MED = 5
CRISIS = (pd.Timestamp("2021-09-01", tz=C.TZ), pd.Timestamp("2023-07-01", tz=C.TZ))
MERGER = pd.Timestamp("2025-09-28", tz=C.TZ)
OUT_DEFAULT = C.ROOT / "Modelling" / "Output" / "rq1c_shap"


def period(ts: pd.Series) -> np.ndarray:
    loc = ts.dt.tz_convert(C.TZ)
    return np.select([loc < CRISIS[0], loc < CRISIS[1], loc < MERGER],
                     ["1_pre_crisis", "2_crisis", "3_post_crisis"], "4_post_merger")


# ---------------------------------------------------------------- models with SHAP
def lgbm_shap(Xtr, ytr, Xte, a):
    """rq2_2 fit (early stopping on the last 15 %, refit with 1.15 x best iteration), then TreeSHAP on the test fold."""
    p, m, best = fit_predict(Xtr, ytr, Xte, a)
    contrib = m.predict(Xte, pred_contrib=True)
    sv, base = contrib[:, :-1], contrib[:, -1]
    gain = pd.Series(m.feature_importance("gain"), index=Xtr.columns)
    return p, sv, base, gain / max(gain.sum(), 1e-12), {"best_iter": best}


def _cat_frame(X: pd.DataFrame):
    Z = X.copy()
    ci = [i for i, c in enumerate(Z.columns) if isinstance(Z[c].dtype, pd.CategoricalDtype)]
    for i in ci:
        c = Z.columns[i]
        Z[c] = np.where(Z[c].isna(), "NA", Z[c].astype(object).astype(str)).astype(object)
    return Z, ci


def cat_shap(Xtr, ytr, Xte, a):
    """Exactly rq2_5_tier1.fit_cat_q (seed 1), plus ShapValues of the median output on the test fold."""
    from catboost import CatBoostRegressor, Pool
    Ztr, ci = _cat_frame(Xtr); Zte, _ = _cat_frame(Xte)
    n = len(Ztr); nv = max(10, int(0.15 * n))
    kw = dict(loss_function="MultiQuantile:alpha=" + ",".join(f"{q:g}" for q in QS),
              learning_rate=a.cat_lr, depth=a.cat_depth, rsm=a.colsample if a.colsample < 1 else None,
              random_seed=1, verbose=0, allow_writing_files=False,
              **({"l2_leaf_reg": a.cat_l2} if a.cat_l2 is not None else {}),
              thread_count=a.threads if a.threads > 0 else -1)
    m0 = CatBoostRegressor(iterations=a.cat_iters, od_type="Iter", od_wait=50, **kw)
    m0.fit(Pool(Ztr.iloc[:-nv], ytr[:-nv], cat_features=ci),
           eval_set=Pool(Ztr.iloc[-nv:], ytr[-nv:], cat_features=ci), use_best_model=True)
    best = max(20, int((m0.get_best_iteration() or 20) * 1.15))
    m = CatBoostRegressor(iterations=best, **kw)
    m.fit(Pool(Ztr, ytr, cat_features=ci))
    pool = Pool(Zte, cat_features=ci)
    Q = np.asarray(m.predict(pool)).reshape(len(Zte), len(QS))
    S = np.asarray(m.get_feature_importance(pool, type="ShapValues"))
    if S.ndim == 3:                                   # (n, n_outputs, F + 1)
        S = S[:, MED, :]
    sv, base = S[:, :-1], S[:, -1]
    pvc = pd.Series(m.get_feature_importance(), index=Xtr.columns)
    return Q, sv, base, pvc / max(pvc.sum(), 1e-12), {"best_iter": best}


# ---------------------------------------------------------------- aggregation
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 30 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def numeric_values(X: pd.DataFrame) -> tuple[np.ndarray, list[bool]]:
    is_cat = [isinstance(X[c].dtype, pd.CategoricalDtype) for c in X.columns]
    V = np.column_stack([X[c].cat.codes.replace(-1, np.nan).to_numpy(float) if ic else X[c].to_numpy(float)
                         for c, ic in zip(X.columns, is_cat)])
    return V, is_cat


def summarise(s, mname, SV, V, is_cat, scale, meta, feats, weather, gains):
    """SV: n x F SHAP, V: n x F feature values, scale: n (cosh of the prediction), meta: series rows (time, year, period)."""
    fam = np.array([family(c, weather) for c in feats])
    absv = np.abs(SV)
    mabs = absv.mean(axis=0)
    tot = mabs.sum() or 1.0
    gain = pd.concat(gains, axis=1).mean(axis=1).reindex(feats).fillna(0).to_numpy() if gains else np.full(len(feats), np.nan)
    imp = pd.DataFrame({"series": s, "model": mname, "feature": feats, "family": fam, "mean_abs_shap": mabs,
                        "share": mabs / tot, "mean_abs_shap_eur": np.nanmean(absv * scale[:, None], axis=0),   # scale NaN where yesterday's price is missing
                        "mean_shap": SV.mean(axis=0), "categorical": is_cat,
                        "direction_spearman": [spearman(V[:, j], SV[:, j]) for j in range(len(feats))],
                        "share_nonzero": (absv > 0).mean(axis=0), "gain_share": gain, "n": len(SV)})
    imp["rank"] = imp["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    imp = imp.sort_values("rank")
    famtab = imp.groupby("family", as_index=False).agg(share=("share", "sum"), n_features=("feature", "size")).assign(series=s, model=mname)
    by = []
    for col in ("year", "period"):
        for k, ix in meta.groupby(col).indices.items():
            m = absv[ix].mean(axis=0)
            fs = pd.Series(m / (m.sum() or 1.0)).groupby(fam).sum()
            by.append(pd.DataFrame({"series": s, "model": mname, "by": col, "group": k, "family": fs.index,
                                    "share": fs.to_numpy(), "n": len(ix)}))
    dep = []
    for j in imp.head(10).index:
        v, sv = V[:, j], SV[:, j]
        ok = ~np.isnan(v)
        if ok.sum() < 30:
            continue
        nb = 10 if len(np.unique(v[ok])) > 10 else len(np.unique(v[ok]))
        b = pd.qcut(pd.Series(v[ok]).rank(method="first"), nb, labels=False)
        t = pd.DataFrame({"bin": b, "v": v[ok], "sv": sv[ok]}).groupby("bin").agg(value_lo=("v", "min"), value_hi=("v", "max"),
                                                                                  value_mean=("v", "mean"), mean_shap=("sv", "mean"),
                                                                                  n=("sv", "size")).reset_index()
        dep.append(t.assign(series=s, model=mname, feature=feats[j], rank=int(imp.loc[j, "rank"])))
    return imp, famtab, (pd.concat(by) if by else pd.DataFrame()), (pd.concat(dep) if dep else pd.DataFrame())


FAM_COL = {"calendar": "#2f6db3", "lagged target": "#d1495b", "cross-market (DE prices)": "#3a9b5d",
           "published forecasts": "#e09f3e", "outages (ex ante)": "#7b5ea7", "lookback (observed)": "#3aa6b9",
           "weather (MeteoSwiss)": "#8c6d46", "regime / source": "#c2549d", "other": "#555555",
           "perfect-forecast block": "#999999"}


def plots(s, mname, imp, SV, V, feats, byt, out):
    import ml_diagnostics as D
    plt = D._plt()
    d = out / "plots" / D._safe(s); d.mkdir(parents=True, exist_ok=True)
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["share"] * 100, color=[FAM_COL.get(f, "#555") for f in top["family"]])
    for fm in dict.fromkeys(top["family"]):
        ax.barh([], [], color=FAM_COL.get(fm, "#555"), label=fm)
    ax.set_xlabel("share of mean |SHAP|, %"); ax.set_title(f"{s} - {mname}: top 20 features (out-of-sample SHAP)")
    ax.legend(fontsize=7, loc="lower right"); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(d / f"bar_{mname}.png"); plt.close(fig)
    # beeswarm-style: SHAP per block, colour = within-series percentile of the feature value
    k = imp.head(15).index.to_numpy()
    rng = np.random.default_rng(0)
    rows = rng.choice(len(SV), min(len(SV), 3000), replace=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, j in enumerate(k[::-1]):
        v = V[rows, j]; pr = pd.Series(v).rank(pct=True).to_numpy()
        y = i + rng.uniform(-0.3, 0.3, len(rows))
        ax.scatter(SV[rows, j], y, c=np.where(np.isnan(pr), 0.5, pr), cmap="coolwarm", s=3, alpha=0.6, vmin=0, vmax=1)
    ax.set_yticks(range(len(k))); ax.set_yticklabels([feats[j] for j in k[::-1]], fontsize=7)
    ax.axvline(0, color="black", lw=0.6); ax.set_xlabel("SHAP (asinh price change vs same slot yesterday)")
    ax.set_title(f"{s} - {mname}: SHAP per block (blue = low feature value, red = high)")
    fig.tight_layout(); fig.savefig(d / f"beeswarm_{mname}.png"); plt.close(fig)
    # dependence: top 6
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, j in zip(axs.ravel(), imp.head(6).index):
        ax.scatter(V[rows, j], SV[rows, j], s=3, alpha=0.4, color="#2f6db3")
        ax.axhline(0, color="black", lw=0.6); ax.set_title(feats[j], fontsize=7)
    axs[0, 0].set_ylabel("SHAP"); fig.suptitle(f"{s} - {mname}: dependence of the top 6 features", fontsize=9)
    fig.tight_layout(); fig.savefig(d / f"dependence_{mname}.png"); plt.close(fig)
    t = byt[(byt.by == "year")].pivot_table(index="group", columns="family", values="share", aggfunc="sum")
    if len(t):
        ax = (t * 100).plot(kind="bar", stacked=True, figsize=(9, 4), color=[FAM_COL.get(c, "#555") for c in t.columns], width=0.8)
        ax.set_ylabel("share of mean |SHAP|, %"); ax.set_xlabel(""); ax.set_title(f"{s} - {mname}: driver families by test year")
        ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.0, 1.0)); fig = ax.get_figure(); fig.tight_layout()
        fig.savefig(d / f"family_by_year_{mname}.png"); plt.close(fig)


def tft_family(out_dir: Path) -> pd.DataFrame:
    """TFT variable-selection weights (rq2_6) aggregated to the same families, per encoder."""
    rows = []
    base = C.ROOT / "Modelling" / "Output" / "rq2_6_tft_mfrr_up" / "diagnostics"
    for fn, part in [("importance_tft.csv", "known / static inputs"), ("importance_tft_past.csv", "past (encoder) inputs")]:
        f = base / fn
        if not f.exists():
            continue
        t = pd.read_csv(f)

        def fam(c):
            c = str(c)
            if c.startswith(("y_", "tgt_prev")): return "lagged target"
            if c in ("weekday", "slot", "hour", "month") or c.startswith(("dow_", "slot_", "hour_", "month_", "doy_", "cal_")): return "calendar"
            return family(c, set())
        t["family"] = t["feature"].map(fam)
        fs = t.groupby("family")["mean"].sum()
        rows.append(pd.DataFrame({"series": "mFRR|up|4h", "source": f"TFT {part}", "family": fs.index,
                                  "share": (fs / fs.sum()).to_numpy()}))
    return pd.concat(rows) if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=C.VIEWS)
    ap.add_argument("--only", default=None, help='series filter, e.g. "mFRR|up|4h"')
    ap.add_argument("--models", nargs="+", default=["lgbm", "cat"], choices=["lgbm", "cat"])
    ap.add_argument("--max-folds", type=int, default=0, help="only the first n folds per series (quick check)")
    ap.add_argument("--top-store", type=int, default=30, help="features per series x model kept in shap_values_*.parquet")
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--refit-months-week", type=int, default=12)
    ap.add_argument("--objective", default="l1")             # rq2_2 defaults
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--colsample", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--cat-iters", type=int, default=1500)   # rq2_5 defaults
    ap.add_argument("--cat-lr", type=float, default=0.05)
    ap.add_argument("--cat-depth", type=int, default=6)
    ap.add_argument("--cat-l2", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT_DEFAULT
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    t_all = time.time()

    ref_l = pd.read_parquet(C.ROOT / "Modelling" / "Output" / "rq2_2_delta" / "preds_exante.parquet",
                            columns=["series", "block_start_utc", "pred"])
    IMP, FAM, BY, DEP, CHK, STORE = [], [], [], [], [], {m: [] for m in a.models}
    for view in a.products:
        df = C.load_view(view)
        feats, weather = feature_list(df, view, "exante")
        X = make_X(df, feats)
        rp = C.ROOT / "Modelling" / "Output" / f"rq2_5_{view}" / "preds.parquet"
        ref_c = pd.read_parquet(rp, columns=["series", "block_start_utc", "p_cat_q"]) if rp.exists() else None
        print(f"{view}: {len(df):,} blocks, {len(feats)} features", flush=True)
        for s in df["series"].unique():
            if a.only and a.only != s:
                continue
            gi = np.flatnonzero((df["series"] == s).to_numpy())
            g, Xg = df.iloc[gi].reset_index(drop=True), X.iloc[gi].reset_index(drop=True)
            off = np.arcsinh(g["b_prev_slot"].to_numpy(float))
            yd = np.arcsinh(g["y"].to_numpy(float)) - off
            ok = ~np.isnan(yd)
            rm = a.refit_months_week if s.endswith("|week") else a.refit_months
            mt = C.MIN_TRAIN[s.split("|")[2]]
            acc = {m: dict(sv=[], te=[], pred=[], gain=[]) for m in a.models}
            nf, t0 = 0, time.time()
            for f, start, tr, te in C.fold_plan(g, rm, mt):
                tr = tr[ok[tr]]
                if len(tr) < mt:
                    continue
                if a.max_folds and nf >= a.max_folds:
                    break
                for m in a.models:
                    if m == "lgbm":
                        p, sv, base, imp, inf = lgbm_shap(Xg.iloc[tr], yd[tr], Xg.iloc[te], a)
                        pm = p
                    else:
                        Q, sv, base, imp, inf = cat_shap(Xg.iloc[tr], yd[tr], Xg.iloc[te], a)
                        pm = Q[:, MED]
                        p = np.sort(Q, axis=1)[:, MED]               # as stored by rq2_5 (sorted quantiles)
                    add = np.abs(sv.sum(axis=1) + base - pm).max()
                    if add > 1e-4:
                        print(f"   WARNING {s} fold {f} {m}: SHAP additivity error {add:.2e}")
                    acc[m]["sv"].append(sv.astype(np.float32)); acc[m]["te"].append(te)
                    acc[m]["pred"].append(np.sinh(p + off[te])); acc[m]["gain"].append(imp)
                    acc[m].setdefault("pmodel", []).append(pm)
                nf += 1
                print(f"   {s:16s} fold {f:2d}: train {len(tr):6,d}, test {len(te):5,d}  {time.time() - t0:6.0f}s", flush=True)
            if nf == 0:
                continue
            for m in a.models:
                te = np.concatenate(acc[m]["te"]); SV = np.vstack(acc[m]["sv"]).astype(np.float64)
                meta = g.iloc[te][["block_start_utc"]].reset_index(drop=True)
                meta["year"] = meta["block_start_utc"].dt.tz_convert(C.TZ).dt.year
                meta["period"] = period(meta["block_start_utc"])
                V, is_cat = numeric_values(Xg.iloc[te])
                scale = np.cosh(np.concatenate(acc[m]["pmodel"]) + off[te])
                # check against the stored RQ2 predictions
                pr = pd.DataFrame({"series": s, "block_start_utc": g["block_start_utc"].iloc[te].to_numpy(),
                                   "refit": np.concatenate(acc[m]["pred"])})
                ref = ref_l.rename(columns={"pred": "stored"}) if m == "lgbm" else \
                    (ref_c.rename(columns={"p_cat_q": "stored"}) if ref_c is not None else None)
                if ref is not None:
                    mm = pr.merge(ref.drop_duplicates(["series", "block_start_utc"]), on=["series", "block_start_utc"], how="inner")
                    e = (mm["refit"] - mm["stored"]).abs()
                    CHK.append(dict(series=s, model=m, n=len(mm), max_abs_diff=e.max(), mean_abs_diff=e.mean(),
                                    share_equal_1e6=(e < 1e-6).mean(), mean_abs_stored=mm["stored"].abs().mean()))
                imp, famt, byt, dep = summarise(s, m, SV, V, is_cat, scale, meta, feats, weather, acc[m]["gain"])
                IMP.append(imp); FAM.append(famt); BY.append(byt); DEP.append(dep)
                keep = imp.head(a.top_store)
                st = pd.DataFrame({"series": s, "block_start_utc": meta["block_start_utc"]})
                for j, c in zip(keep.index, keep["feature"]):
                    st[f"shap__{c}"] = SV[:, j].astype(np.float32); st[f"value__{c}"] = V[:, j].astype(np.float32)
                STORE[m].append(st)
                try:
                    plots(s, m, imp, SV, V, feats, byt, out)
                except Exception as ex:                     # noqa: BLE001
                    print(f"   [plots skipped {s} {m}: {type(ex).__name__}: {ex}]")
            print(f"{view} {s:16s} folds={nf:3d}  {time.time() - t0:6.0f}s", flush=True)

    IMP, FAM, BY, DEP, CHK = (pd.concat(x) if x and isinstance(x[0], pd.DataFrame) else pd.DataFrame(x)
                              for x in (IMP, FAM, BY, DEP, CHK))
    IMP.to_csv(out / "shap_importance.csv", index=False); FAM.to_csv(out / "shap_family.csv", index=False)
    BY[BY.by == "year"].drop(columns="by").rename(columns={"group": "year"}).to_csv(out / "shap_family_by_year.csv", index=False)
    BY[BY.by == "period"].drop(columns="by").rename(columns={"group": "period"}).to_csv(out / "shap_family_by_period.csv", index=False)
    DEP.to_csv(out / "shap_dependence.csv", index=False); CHK.to_csv(out / "fit_check.csv", index=False)
    for m, l in STORE.items():
        if l:
            pd.concat(l).to_parquet(out / f"shap_values_{m}.parquet")

    # agreement between models and between SHAP and gain
    AG = []
    for s, g in IMP.groupby("series", sort=False):
        P = g.pivot_table(index="feature", columns="model", values="mean_abs_shap")
        row = {"series": s}
        for m in P.columns:
            gm = g[g.model == m].set_index("feature")
            row[f"{m}_shap_vs_gain_spearman"] = spearman(gm["mean_abs_shap"].to_numpy(), gm["gain_share"].to_numpy())
            row[f"{m}_top10_share"] = gm.sort_values("rank")["share"].head(10).sum()
        if {"lgbm", "cat"} <= set(P.columns):
            row["lgbm_vs_cat_spearman"] = spearman(P["lgbm"].to_numpy(), P["cat"].to_numpy())
            t1 = set(P["lgbm"].nlargest(20).index); t2 = set(P["cat"].nlargest(20).index)
            row["top20_overlap"] = len(t1 & t2)
        AG.append(row)
    AG = pd.DataFrame(AG); AG.to_csv(out / "model_agreement.csv", index=False)
    TF = tft_family(out)
    if len(TF):
        sh = FAM[FAM.series == "mFRR|up|4h"].rename(columns={"model": "source"})
        sh["source"] = "SHAP " + sh["source"]
        TFc = pd.concat([TF, sh[["series", "source", "family", "share"]]])
        TFc.pivot_table(index="family", columns="source", values="share").to_csv(out / "tft_vs_shap_family.csv")

    # ---- report
    L = ["RQ1c - SHAP drivers of the RQ2 tree models (out-of-sample TreeSHAP on every test fold; delta target "
         "asinh(y) - asinh(same slot yesterday))", f"models: {', '.join(a.models)}; folds per series as RQ2"
         + (f" (first {a.max_folds} only)" if a.max_folds else ""), ""]
    if len(CHK):
        L += ["== Check: refitted models vs stored RQ2 predictions (EUR/MWh; should be ~0)", CHK.to_string(index=False), ""]
    for m in a.models:
        F = FAM[FAM.model == m].pivot_table(index="series", columns="family", values="share") * 100
        N = FAM[FAM.model == m].pivot_table(index="series", columns="family", values="n_features")
        L += [f"== Driver families [{m}]: share of mean |SHAP| (%)", F.round(1).to_string(),
              "   number of features per family (a large family can win on count alone):", N.to_string(), ""]
    for (s, m), g in IMP.groupby(["series", "model"], sort=False):
        t = g.head(10)
        L.append(f"== {s} [{m}] top 10 (share %, EUR/MWh effect, direction: + = higher value -> higher price)")
        for r in t.itertuples():
            dr = "cat" if r.categorical else (f"{r.direction_spearman:+.2f}" if pd.notna(r.direction_spearman) else "n/a")
            L.append(f"   {r.rank:2d}. {r.feature:55s} {r.share * 100:5.1f} %  {r.mean_abs_shap_eur:8.2f}  dir {dr:>5s}  [{r.family}]")
        L.append("")
    if len(BY):
        P = BY[(BY.by == "period") & (BY.model == a.models[0])].pivot_table(index=["series", "group"], columns="family", values="share") * 100
        L += [f"== Driver families by period [{a.models[0]}] (%)", P.round(1).to_string(), ""]
    L += ["== Agreement (rank correlations of feature importance; top-20 overlap of lgbm vs cat)", AG.round(3).to_string(index=False), ""]
    if len(TF):
        L += ["== mFRR up 4h: TFT variable-selection weights vs SHAP, by family (shares; TFT weights are within each encoder)",
              pd.read_csv(out / "tft_vs_shap_family.csv", index_col=0).round(3).to_string(), ""]
    if len(FAM):
        lt = FAM[(FAM.family == "lagged target")].pivot_table(index="series", columns="model", values="share") * 100
        L += ["== Share of the lagged-target family (own price history beyond persistence), % - compare FCR with the others "
              "(Chronos-2, price history only, is the best FCR model)", lt.round(1).to_string(), ""]
    (out / "report_rq1c.txt").write_text("\n".join(L))
    print("\n".join(L))
    for s, g in IMP[IMP.model == a.models[0]].groupby("series", sort=False):
        top = g.head(1).iloc[0]
        RU.summary(f"{s.replace('|', ' ')}: top {top.feature[:30]} ({top.share * 100:.0f} %)")
    print(f"total {time.time() - t_all:.0f}s; written to {out}")


if __name__ == "__main__":
    RU.run(main, "RQ1c SHAP")
