"""RQ3a step 2 (todo 9.3): do long and short spikes have different drivers? Nested logistic models.

Reading of step 1 (rq3a_1): activation, imbalance volume and same-QH system data are concurrent
OUTCOMES of a spike, not drivers; they are NOT used here. Features are grouped in blocks:

  cal    calendar: hour / season harmonics, weekday dummies, holiday, bridge, Christmas, DST day, school share
  fund   D-1 18:00 fundamentals: DA prices (CH level vs trailing 30 d, neighbour spreads), residual-load and
         solar day-ahead forecasts (CH, DE-LU), load forecast
  hist   persistence: spike share in the previous 24 h (same side and other side), D-1 lookback of price
  wx     observed weather (perfect forecast, ex_ante = False): irradiation, ramp, temperature, wind, precip
  err    realised forecast errors of the SAME quarter-hour (actual - day-ahead forecast; ex post)

Models: cumulative (cal, +fund, +hist, +wx, +err) and each block alone. Unit = quarter-hour, QH-price
regime (from 1 Jun 2022). Split: train < 2025-01-01 <= test (post price cap). L2-logistic (numpy Newton),
features standardised on train, NaN -> train median (columns with > 30 % NaN in train dropped).
Metrics on test: ROC-AUC, PR-AUC (average precision), pseudo-R2 (McFadden vs train base rate),
Brier skill vs train base rate. Also standardised coefficients (+ Wald z) of the full model per side.

Run from Master_Thesis with .venv active:
    python Modelling/rq3a_2_logit.py [--split 2025-01-01] [--lam 5]
Needs Modelling/rq3a_1_catalogue.py (imports its loader). Writes only Modelling/Output/rq3a_2/.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Clean"))
sys.path.insert(0, str(HERE))
import views as V  # noqa: E402
import rq3a_1_catalogue as C  # noqa: E402

OUT = HERE / "Output" / "rq3a_2"
TARGETS = {"short": "spike_short_q990", "long": "spike_long_q010"}
lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ data
EXTRA = ["cal_doy_sin", "cal_doy_cos", "cal_day_before_holiday", "cal_day_after_holiday",
         "ch_price_da_eur_mwh__fc_d1", "de_lu_price_da_eur_mwh__fc_d1", "fr_price_da_eur_mwh__fc_d1",
         "it_nord_price_da_eur_mwh__fc_d1", "at_price_da_eur_mwh__fc_d1",
         "ch_resid_load_da_fc_mw__fc_d1", "ch_resid_load_da_fc_mw_ramp__fc_d1",
         "de_lu_resid_load_da_fc_mw__fc_d1", "de_lu_resid_load_da_fc_mw_ramp__fc_d1",
         "ch_gen_solar_da_fc_mw__fc_d1", "ch_gen_solar_da_fc_ramp_mw__fc_d1", "de_lu_gen_solar_da_fc_mw__fc_d1",
         "ch_load_da_fc_mw__fc_d1",
         "spikehist_short_q990__lb24h_d1", "spikehist_long_q010__lb24h_d1",
         "ch_imb_price_short_eur_mwh__lb24h_d1", "ch_imb_price_long_eur_mwh__lb24h_d1",
         "err_ch_load_mw__lb24h_d1", "err_ch_solar_mw__lb24h_d1", "err_de_lu_solar_mw__lb24h_d1",
         "err_de_lu_wind_on_mw__lb24h_d1", "err_de_lu_load_mw__lb24h_d1"]


def build() -> pd.DataFrame:
    df = C.load()
    avail = set(pd.read_csv(V.VIEWS_DIR / "_rq3_dictionary.csv")["column"])
    extra = [c for c in EXTRA if c in avail and c not in df.columns]
    n = len(df)
    ex = V.load_rq3(columns=extra)[["ts_utc"] + extra]
    df = df.merge(ex, on="ts_utc", how="left", validate="one_to_one")
    assert len(df) == n
    df = df.sort_values("ts_utc").reset_index(drop=True)
    h = df["cal_hour"].astype(float)
    for k in (1, 2):
        df[f"hr_sin{k}"] = np.sin(2 * np.pi * k * h / 24)
        df[f"hr_cos{k}"] = np.cos(2 * np.pi * k * h / 24)
    for d in range(1, 7):
        df[f"dow{d}"] = (df["cal_dow"] == d).astype(float)
    # DA price level relative to its trailing 30 days (removes 2021-22 level shift); neighbour spreads vs CH
    ch = df["ch_price_da_eur_mwh__fc_d1"]
    df["da_ch_rel30"] = ch - ch.rolling(96 * 30, min_periods=96 * 7).mean()
    for z in ("de_lu", "fr", "it_nord", "at"):
        c = f"{z}_price_da_eur_mwh__fc_d1"
        if c in df:
            df[f"da_{z}_minus_ch"] = df[c] - ch
    return df


def blocks(df: pd.DataFrame) -> dict[str, list[str]]:
    b = {
        "cal": ["hr_sin1", "hr_cos1", "hr_sin2", "hr_cos2", *[f"dow{d}" for d in range(1, 7)], "cal_holiday",
                "cal_day_before_holiday", "cal_day_after_holiday", "cal_bridge_day", "cal_xmas_period",
                "cal_dst_switch_day", "cal_school_holiday_share", "cal_doy_sin", "cal_doy_cos"],
        "fund": ["da_ch_rel30", "da_de_lu_minus_ch", "da_fr_minus_ch", "da_it_nord_minus_ch", "da_at_minus_ch",
                 "ch_resid_load_da_fc_mw__fc_d1", "ch_resid_load_da_fc_mw_ramp__fc_d1",
                 "de_lu_resid_load_da_fc_mw__fc_d1", "de_lu_resid_load_da_fc_mw_ramp__fc_d1",
                 "ch_gen_solar_da_fc_mw__fc_d1", "ch_gen_solar_da_fc_ramp_mw__fc_d1",
                 "de_lu_gen_solar_da_fc_mw__fc_d1", "ch_load_da_fc_mw__fc_d1"],
        "hist": ["spikehist_short_q990__lb24h_d1", "spikehist_long_q010__lb24h_d1",
                 "err_ch_load_mw__lb24h_d1", "err_ch_solar_mw__lb24h_d1", "err_de_lu_solar_mw__lb24h_d1",
                 "err_de_lu_wind_on_mw__lb24h_d1", "err_de_lu_load_mw__lb24h_d1"],
        "wx": ["ch_ghi_lw_wm2__pf", "ch_ghi_ramp_lw_wm2__pf", "ch_temp_lw_degc__pf", "ch_wind_lw_ms__pf",
               "ch_precip_hydro_mm__pf", "ch_snow_hydro_cm__pf", "ch_melt_dh_hydro_kh__pf"],
        "err": ["err_ch_load_mw", "err_ch_solar_mw", "err_de_lu_load_mw", "err_de_lu_solar_mw",
                "err_de_lu_wind_on_mw", "err_fr_solar_mw", "err_at_load_mw"],
    }
    return {k: [c for c in v if c in df.columns] for k, v in b.items()}


# ------------------------------------------------------------------ model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logit(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 60):
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    pen = np.full(k + 1, lam)
    pen[0] = 0.0
    beta = np.zeros(k + 1)
    beta[0] = math.log(max(y.mean(), 1e-6) / (1 - max(y.mean(), 1e-6)))
    def obj(b):
        pp = np.clip(sigmoid(A @ b), 1e-12, 1 - 1e-12)
        return -(y * np.log(pp) + (1 - y) * np.log(1 - pp)).sum() + 0.5 * (pen * b * b).sum()

    f = obj(beta)
    for _ in range(iters):
        p = sigmoid(A @ beta)
        g = A.T @ (p - y) + pen * beta
        w = p * (1 - p) + 1e-9
        H = (A * w[:, None]).T @ A + np.diag(pen)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6:                      # step halving: penalised NLL must decrease
            nb = beta - t * step
            fn = obj(nb)
            if fn <= f:
                break
            t /= 2
        else:
            break
        beta, f = nb, fn
        if np.abs(t * step).max() < 1e-7:
            break
    p = sigmoid(A @ beta)
    w = p * (1 - p) + 1e-9
    cov = np.linalg.inv((A * w[:, None]).T @ A + np.diag(pen))
    return beta, np.sqrt(np.diag(cov))


def predict(beta, X):
    return sigmoid(np.column_stack([np.ones(len(X)), X]) @ beta)


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def avg_precision(y, p):
    o = np.argsort(-p)
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float(prec[ys == 1].mean()) if ys.sum() else float("nan")


def loglik(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def evaluate(tr, te, cols, target, lam):
    cols = [c for c in cols if tr[c].isna().mean() <= 0.30 and tr[c].nunique() > 1]
    med = tr[cols].median()
    mu, sd = tr[cols].fillna(med).mean(), tr[cols].fillna(med).std().replace(0, 1)
    Xtr = ((tr[cols].fillna(med) - mu) / sd).clip(-6, 6).to_numpy()
    Xte = ((te[cols].fillna(med) - mu) / sd).clip(-6, 6).to_numpy()
    ytr, yte = tr[target].to_numpy(float), te[target].to_numpy(float)
    beta, se = fit_logit(Xtr, ytr, lam)
    p = predict(beta, Xte)
    base = ytr.mean()
    ll0 = loglik(yte, np.full(len(yte), base))
    brier = float(((p - yte) ** 2).mean())
    brier0 = float(((base - yte) ** 2).mean())
    res = {"n_features": len(cols), "n_test": len(yte), "test_pos": int(yte.sum()), "AUC": auc(yte, p),
           "PR_AUC": avg_precision(yte, p), "PR_AUC_base": float(yte.mean()),
           "McFadden_R2": 1 - loglik(yte, p) / ll0, "Brier_skill": 1 - brier / brier0}
    coef = pd.DataFrame({"feature": cols, "coef_std": beta[1:], "z": beta[1:] / se[1:]})
    return res, coef


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--lam", type=float, default=5.0, help="L2 penalty on standardised coefficients")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = build()
    q = df[df["date"] >= C.QH_START].copy()
    split = pd.Timestamp(a.split)
    tr, te = q[q["date"] < split], q[q["date"] >= split]
    log(f"RQ3a step 2: QH regime {len(q):,} rows; train {len(tr):,} (to {split.date()}), test {len(te):,}; lambda {a.lam}")
    B = blocks(q)
    for k, v in B.items():
        log(f"  block {k}: {len(v)} features")
    order = ["cal", "fund", "hist", "wx", "err"]
    rows, coefs = [], {}
    for side, target in TARGETS.items():
        log(f"\n=== {side}: {target}; train positives {int(tr[target].sum())}, test positives {int(te[target].sum())}")
        for i, k in enumerate(order):
            cols = [c for b in order[: i + 1] for c in B[b]]
            r, cf = evaluate(tr, te, cols, target, a.lam)
            rows.append({"side": side, "model": "cumulative to " + k, **r})
            if k == "err":
                coefs[side] = cf
        for k in order:
            r, _ = evaluate(tr, te, B[k], target, a.lam)
            rows.append({"side": side, "model": "block alone: " + k, **r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_comparison.csv", index=False)
    for side in TARGETS:
        log(f"\n-- {side}: test metrics")
        log(res[res["side"] == side].drop(columns="side").set_index("model").to_string(float_format=lambda x: f"{x:,.3f}"))
    # coefficients of the full model, side by side
    cs, cl = coefs["short"].set_index("feature"), coefs["long"].set_index("feature")
    cmp_ = cs.join(cl, lsuffix="_short", rsuffix="_long", how="outer")
    cmp_["sign_differs"] = np.sign(cmp_["coef_std_short"]) != np.sign(cmp_["coef_std_long"])
    cmp_["max_abs_z"] = cmp_[["z_short", "z_long"]].abs().max(axis=1)
    cmp_ = cmp_.sort_values("max_abs_z", ascending=False)
    cmp_.to_csv(OUT / "coefficients_full.csv")
    log("\n-- full model, standardised coefficients (log-odds per 1 sd), sorted by max |z|")
    log(cmp_.head(25).to_string(float_format=lambda x: f"{x:,.2f}"))
    sig = cmp_[(cmp_[["z_short", "z_long"]].abs().min(axis=1) > 3) & cmp_["sign_differs"]]
    log(f"\nfeatures significant (|z|>3) on both sides with opposite sign: {list(sig.index)}")
    (OUT / "report_rq3a_2.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
