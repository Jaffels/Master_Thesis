"""RQ2 step 6 (Tier 2 of ML_Options.md): advanced models on the same task as rq2_5_tier1.py.

Same series, target, rolling-origin folds (expanding window, refit every 3 months / weekly 12, test
from 1 Jan 2021), ex-ante features, quantile levels, sequential CQR and evaluation as rq2_5, so the
results drop straight into Tables 9 / 10. Tier 1 predictions (Output/rq2_5_<product>/preds.parquet or
Output/rq2_5/preds.parquet) are merged so every Tier 2 model is tested against LEAR, round-1
LightGBM, the best Tier 1 model and the naive baselines on exactly the same rows.

Models (--models, default all four):
  tft      Temporal Fusion Transformer (Lim et al. 2021), PyTorch implementation in tier2_nets.py.
           Known inputs of the target block = full ex-ante feature vector; past inputs = observed
           price history on the block grid (--tft-lookback, 4h default 42 = 7 days) + calendar.
           Quantile output, variable importance (VSN weights) and attention by lag are saved.
  nhits    NHITS with exogenous inputs (Challu et al. 2023; N-BEATSx-style exogenous concatenation),
           multi-rate pooling of a longer window (--nhits-lookback, 4h default 168 = 28 days).
  tabpfn   TabPFN regressor (Hollmann et al. 2025; pretrained transformer, in-context learning, no
           training). Same tree feature matrix, delta target, trained on the most recent
           --tabpfn-max-train blocks of the window; > --tabpfn-max-features columns are reduced by
           |Spearman correlation| with the target inside the training window (lagged prices always
           kept). Full predictive distribution -> the 11 quantiles.
  chronos  Chronos-2 (Ansari et al. 2025), zero-shot foundation model: the price history on the
           block grid (asinh scale, missing blocks NaN) up to the auction start, forecast of the
           auction's blocks (4h: up to 6 steps). Optional covariates (--chronos-cov). No training;
           folds only define the evaluation rows.
All models predict asinh(price) (trees / nets / TabPFN: as change vs same slot yesterday), errors in
EUR/MWh. Quantiles are non-crossing (nets: by construction; others: sorted, crossing share logged).

Outputs (Output/rq2_6 or --out):
  preds.parquet             Tier 2 quantiles / medians per test block (+ keys, baselines, CQR bounds)
  metrics_point.csv, metrics_prob.csv, metrics_by_year.csv, dm.csv   (Tier 1 models included)
  fold_log.csv              per fold: sizes, timings, epochs, best epoch, val loss, crossing share
  training_curves.csv       per fold / seed / epoch: train and validation pinball loss (nets)
  tft_attention.csv         TFT attention by lag (grid steps before the auction start), per fold
  importance_*.csv / plots  via diagnostics/ (TFT known-input VSN = 'tft', past-input VSN = 'tft_past')
  report_rq2_6.txt, run.log, run_meta.json, diagnostics/ (see its README.txt)
  preds_partial.parquet     written after every series (a crash keeps the finished series)
macOS notification at the end (run_utils.py); a short one after each series that took > 10 min.

Check the setup first:  python Modelling/tier2_setup_check.py
Quick smoke test:       python Modelling/rq2_6_tier2.py --only "mFRR|up|week" --quick
First real run (todo):  python Modelling/rq2_6_tier2.py --only "mFRR|up|4h" --models tft --out Modelling/Output/rq2_6_tft_mfrr_up
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path

import faulthandler

faulthandler.enable()          # a crash inside a C library prints the Python line instead of a bare 'segmentation fault'
import torch  # noqa: E402,F401  load torch (and its OpenMP) BEFORE LightGBM / CatBoost: on macOS two OpenMP copies in
              # the wrong order crash the process (segmentation fault when torch is imported after lightgbm)

import numpy as np
import pandas as pd
import run_utils as RU  # run.log, run_meta.json, run history, macOS notification

import rq2_common as C
from rq2_2_lightgbm import feature_list, make_X
from rq2_5_tier1 import INTERVALS, MED, QS, QTAG, cqr, dm_loss, linear_frame, pinball, seed_spread, tree_numeric

warnings.filterwarnings("ignore")

MODELS = ["tft", "nhits", "tabpfn", "chronos"]
NET = {"tft", "nhits"}
LABEL = {"tft": "Temporal Fusion Transformer", "nhits": "NHITS-x", "tabpfn": "TabPFN", "chronos": "Chronos-2 (zero-shot)",
         "lear": "LEAR (LASSO)", "lgbm_l1": "LightGBM L1 (round 1)", "lgbm_q": "LightGBM quantile",
         "cat_q": "CatBoost MultiQuantile", "qrf": "Quantile Regression Forest", "svr": "SVR (RBF)"}
T1_POINT = ["lear", "lgbm_l1", "lgbm_q", "cat_q", "qrf", "svr"]
T1_PROB = ["lgbm_q", "cat_q", "qrf"]
LOOKBACK = {"tft": {"4h": 42, "day": 28, "week": 26}, "nhits": {"4h": 168, "day": 56, "week": 52}}
POOLS = {"4h": (6, 3, 1), "day": (7, 2, 1), "week": (4, 2, 1)}


# ---------------------------------------------------------------- Tier 1 predictions
def load_tier1(a, views) -> pd.DataFrame | None:
    """Tier 1 point + quantile columns from rq2_5 (per-product folders first, then Output/rq2_5)."""
    base = C.ROOT / "Modelling" / "Output"
    paths = [Path(p) for p in a.tier1] if a.tier1 else \
        [base / f"rq2_5_{v}" / "preds.parquet" for v in views] + [base / "rq2_5" / "preds.parquet"]
    parts = []
    for p in paths:
        p = p if p.suffix == ".parquet" else p / "preds.parquet"
        if not p.exists():
            continue
        import pyarrow.parquet as pq
        cols = set(pq.read_schema(p).names)
        want = [f"p_{m}" for m in T1_POINT] + [f"q_{m}_{t}" for m in T1_PROB for t in QTAG]
        keep = ["series", "block_start_utc"] + [c for c in want if c in cols]
        parts.append(pd.read_parquet(p, columns=keep))
        print(f"   tier 1 predictions: {p.relative_to(C.ROOT) if str(p).startswith(str(C.ROOT)) else p}")
    if not parts:
        print("   tier 1 predictions: none found (comparisons only vs the naive baselines)")
        return None
    T = pd.concat(parts, ignore_index=True).drop_duplicates(["series", "block_start_utc"], keep="first")
    return T


# ---------------------------------------------------------------- TabPFN
def select_columns(Ztr: pd.DataFrame, y: np.ndarray, k: int, keep=("tgt_prev",)) -> list[str]:
    if Ztr.shape[1] <= k:
        return list(Ztr.columns)
    must = [c for c in Ztr.columns if c.startswith(tuple(keep))]
    ok = Ztr.columns[(Ztr.notna().mean() > 0.05) & (Ztr.std() > 1e-12)]
    r = Ztr[ok].rank().corrwith(pd.Series(y, index=Ztr.index).rank()).abs().fillna(0).sort_values(ascending=False)
    rest = [c for c in r.index if c not in must][:max(0, k - len(must))]
    return list(dict.fromkeys(must + rest))[:k]


def fit_tabpfn(Ztr, ytr, Zte, a, seed, cat_cols):
    from tabpfn import TabPFNRegressor
    cols = select_columns(Ztr, ytr, a.tabpfn_max_features)
    Ztr, Zte = Ztr[cols], Zte[cols]
    cat_idx = [i for i, c in enumerate(cols) if c in cat_cols]
    kw = dict(device=a.tabpfn_device, random_state=seed, ignore_pretraining_limits=True,
              categorical_features_indices=cat_idx or None)
    if a.tabpfn_estimators > 0:
        kw["n_estimators"] = a.tabpfn_estimators
    import torch

    def make(kw):
        if a.tabpfn_version == "default":
            return TabPFNRegressor(**kw)
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion(a.tabpfn_version), **kw)

    def fit_pred(kw):
        m = make(kw)
        m.fit(Xtr_, ytr.astype(np.float32))
        out = []
        for i in range(0, len(Xte_), a.tabpfn_batch):            # small test chunks: bounded GPU memory
            q = m.predict(Xte_[i:i + a.tabpfn_batch], output_type="quantiles", quantiles=[float(x) for x in QS])
            out.append(np.column_stack([np.asarray(v, float) for v in q]))
        del m
        return np.vstack(out)

    Xtr_, Xte_ = Ztr.to_numpy(np.float32), Zte.to_numpy(np.float32)
    dev_used = a.tabpfn_device
    try:
        Q = fit_pred(kw)
    except Exception as e:                                        # noqa: BLE001
        if "out of memory" not in str(e).lower() or a.tabpfn_device == "cpu":
            raise
        print("      TabPFN: Apple-GPU (MPS) out of memory -> this fold on CPU")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        Q = fit_pred({**kw, "device": "cpu"})
        dev_used = "cpu (MPS out of memory)"
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return Q, {"n_features_used": len(cols), "device": dev_used}


# ---------------------------------------------------------------- Chronos-2
_CHRONOS = {}


def chronos_pipe(a):
    if "pipe" not in _CHRONOS:
        import torch
        from chronos import Chronos2Pipeline
        dev = a.chronos_device
        if dev == "auto":
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            _CHRONOS["pipe"] = Chronos2Pipeline.from_pretrained(a.chronos_model, device_map=dev)
        except Exception as e:                       # noqa: BLE001
            print(f"   chronos on {dev} failed ({type(e).__name__}: {e}); falling back to CPU")
            _CHRONOS["pipe"] = Chronos2Pipeline.from_pretrained(a.chronos_model, device_map="cpu")
        _CHRONOS["dev"] = dev
    return _CHRONOS["pipe"]


def chronos_fold(g, Xg, gidx, e0, te, a):
    """Zero-shot quantiles (asinh level) for the test rows te: one context per auction."""
    pipe = chronos_pipe(a)
    ya = np.arcsinh(g["y"].to_numpy(float)) if a.chronos_space == "asinh" else g["y"].to_numpy(float)
    G = int(gidx.max()) + 1
    gy = np.full(G, np.nan)
    gy[gidx] = ya
    cov = {}
    for c in a.chronos_cov:
        v = np.full(G, np.nan)
        v[gidx] = pd.to_numeric(Xg[c], errors="coerce").to_numpy(float)
        cov[c] = pd.Series(v).ffill().to_numpy()
    starts = np.unique(e0[te])
    H = int((gidx[te] - e0[te]).max()) + 1
    inputs = []
    for s in starts:
        lo = max(0, s - a.chronos_context)
        ctx = gy[lo:s]
        if np.isnan(ctx).all():
            ctx = np.array([np.nanmedian(gy[:s])]) if s > 0 and not np.isnan(gy[:s]).all() else np.array([0.0])
        if cov:
            inputs.append({"target": ctx, "past_covariates": {c: v[lo:lo + len(ctx)] for c, v in cov.items()},
                           "future_covariates": {c: np.pad(v[s:s + H], (0, max(0, H - len(v[s:s + H]))), mode="edge")
                                                 for c, v in cov.items()}})
        else:
            inputs.append(ctx.astype(np.float32))
    qs, _ = pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=[float(x) for x in QS],
                                   batch_size=a.chronos_batch)
    Q = np.stack([np.asarray(q.cpu().numpy() if hasattr(q, "cpu") else q, float)[0] for q in qs])  # (A, H, Q)
    pos = {s: i for i, s in enumerate(starts)}
    out = np.stack([Q[pos[e0[r]], gidx[r] - e0[r]] for r in te])
    if a.chronos_space == "raw":
        out = np.arcsinh(out)
    return out, {"n_auctions": len(starts), "horizon": H}


# ---------------------------------------------------------------- per series
def run_series(g: pd.DataFrame, Xg: pd.DataFrame, s: str, a, log, store):
    import torch
    import tier2_nets as N
    proc = s.split("|")[2]
    ya = np.arcsinh(g["y"].to_numpy(float))
    off = np.arcsinh(g["b_prev_slot"].to_numpy(float))
    yd = ya - off
    ok_d = ~np.isnan(yd)
    gidx = N.grid_index(g, proc)
    if len(np.unique(gidx)) < len(gidx):
        log(f"   WARNING {s}: {len(gidx) - len(np.unique(gidx))} blocks share a grid position (kept, history uses the last)")
    e0 = N.auction_start(g, gidx)
    Zl = linear_frame(Xg, g) if NET & set(a.models) else None
    Zt = tree_numeric(Xg) if "tabpfn" in a.models else None
    cat_cols = {c for c in Xg.columns if isinstance(Xg[c].dtype, pd.CategoricalDtype)}
    origin = N.feature_origin(list(Zl.columns), Xg) if Zl is not None else None
    dev = N.pick_device(a.device)
    levels = torch.tensor(QS, dtype=torch.float32, device=dev)
    rm = a.refit_months_week if s.endswith("|week") else a.refit_months
    mt = C.MIN_TRAIN[proc]
    keys = ["series", "product", "direction", "procurement", "auction_id", "block_start_utc", "slot", "y", *C.BASELINES,
            *[c for c in g.columns if c.startswith("regime_")]]
    rows, flog, timers = [], [], {m: 0.0 for m in a.models}
    warm: dict[tuple, dict] = {}
    net_sel = None
    nf = 0
    for f, start, tr_all, te in C.fold_plan(g, rm, mt):
        trd = tr_all[ok_d[tr_all]]
        if len(trd) < mt:
            continue
        nf += 1
        if a.max_folds and nf > a.max_folds:
            break
        r = g.iloc[te][keys].copy(); r["fold"] = f; r["n_train"] = len(trd)
        fl = {"series": s, "fold": f, "fold_start": start.date(), "train_from": g["block_start_utc"].iloc[trd].min().date(),
              "test_from": g["block_start_utc"].iloc[te].min().date(), "test_to": g["block_start_utc"].iloc[te].max().date(),
              "n_train": len(trd), "n_test": len(te)}
        if Zl is not None and a.net_max_features and net_sel is None:
            # ex ante: chosen on the FIRST fold's training window only, then fixed for the series
            net_sel = select_columns(Zl.iloc[trd], yd[trd], a.net_max_features, keep=("tgt_prev", "slot_", "dow_"))
            n_all = Zl.shape[1]
            Zl = Zl[net_sel]
            origin = N.feature_origin(list(Zl.columns), Xg)
            log(f"   {s}: nets use {len(net_sel)} of {n_all} inputs "
                f"(|Spearman| with the target, first training window; lagged prices, slot, weekday always kept)")
        A = N.net_prep(Zl, trd) if Zl is not None else None
        if A is not None:
            fl["net_n_inputs"] = A.shape[1]
        for m in a.models:
            t0 = time.time()
            Qsum = np.zeros((len(te), len(QS)))
            if m in NET:
                L = (a.tft_lookback or LOOKBACK["tft"][proc]) if m == "tft" else (a.nhits_lookback or LOOKBACK["nhits"][proc])
                data = N.FoldData(A, ya, yd, gidx, e0, proc, trd, pad=L)
                for sd in range(a.seeds):
                    if m == "tft":
                        net = N.TFT(A.shape[1], len(N.ENC_VARS), len(QS), MED, d=a.tft_hidden, heads=a.tft_heads,
                                    dropout=a.dropout)
                    else:
                        net = N.NHITSx(L, A.shape[1], len(QS), MED, pools=POOLS[proc], hidden=a.nhits_hidden,
                                       dropout=a.dropout, layers=a.nhits_layers)
                    net = net.to(dev)
                    key = (m, sd)
                    is_warm = a.warm_start and key in warm
                    if is_warm:
                        net.load_state_dict(warm[key])
                    info = N.train(net, m, data, trd, L, a, seed=1 + sd, levels=levels, dev=dev,
                                   max_epochs=a.epochs_warm if is_warm else a.epochs, curve=store["curves"],
                                   tag={"series": s, "model": m, "fold": f, "seed": 1 + sd, "warm_start": is_warm})
                    if a.warm_start:
                        warm[key] = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    q_, w = N.predict(net, m, data, te, L, dev, weights=(m == "tft" and sd == 0))
                    Qsum += q_
                    if a.seeds > 1:                     # per-seed median (seed robustness, done_todo 2.2 item 6)
                        r[f"p_{m}_seed{sd + 1}"] = np.sinh(np.sort(q_, axis=1)[:, MED] + off[te])
                    if sd == 0:
                        fl.update({f"{m}_{k}": v for k, v in info.items()})
                        fl[f"{m}_warm_start"] = is_warm
                        fl[f"{m}_lookback"] = L
                        if w is not None:
                            imp = pd.Series(w["known"].mean(0), index=origin).groupby(level=0).sum()
                            store["imp"].setdefault("tft", {}).setdefault(s, []).append(imp / imp.sum())
                            ip = pd.Series(w["past"].mean(0), index=N.ENC_VARS)
                            store["imp"].setdefault("tft_past", {}).setdefault(s, []).append(ip / ip.sum())
                            att = w["attention"].mean(0)
                            for j, v in enumerate(att):
                                store["att"].append({"series": s, "fold": f, "position": j,
                                                     "steps_before_auction_start": L - j if j < L else 0,
                                                     "is_target_block": j == L, "weight": float(v)})
                Q = Qsum / a.seeds + off[te][:, None]
                fl[f"{m}_crossing_share"] = 0.0
            elif m == "tabpfn":
                tr_t = trd[-a.tabpfn_max_train:] if len(trd) > a.tabpfn_max_train else trd
                for sd in range(a.seeds):
                    q_, info = fit_tabpfn(Zt.iloc[tr_t], yd[tr_t], Zt.iloc[te], a, seed=1 + sd, cat_cols=cat_cols)
                    Qsum += q_
                    if a.seeds > 1:                     # per-seed median, clipped like the average below
                        q1 = np.clip(q_, np.nanmin(yd[tr_t]), np.nanmax(yd[tr_t]))
                        r[f"p_{m}_seed{sd + 1}"] = np.sinh(np.sort(q1, axis=1)[:, MED] + off[te])
                raw = Qsum / a.seeds
                # TabPFN's predictive distribution can put a far tail beyond anything seen in training; through
                # sinh() that becomes absurd prices (one aFRR up week quantile was 1.4 million EUR/MWh). Clip the delta
                # quantiles to the training range of the delta target (same rule as LEAR / SVR in rq2_5).
                lo_, hi_ = np.nanmin(yd[tr_t]), np.nanmax(yd[tr_t])
                fl["tabpfn_clipped_share"] = float(((raw < lo_) | (raw > hi_)).mean())
                raw = np.clip(raw, lo_, hi_)
                fl.update({"tabpfn_n_train_used": len(tr_t), **{f"tabpfn_{k}": v for k, v in info.items()},
                           "tabpfn_crossing_share": float((np.diff(raw, axis=1) < 0).any(axis=1).mean())})
                Q = np.sort(raw, axis=1) + off[te][:, None]
            else:                                                           # chronos (deterministic)
                raw, info = chronos_fold(g, Xg, gidx, e0, te, a)
                fl.update({f"chronos_{k}": v for k, v in info.items()})
                fl["chronos_crossing_share"] = float((np.diff(raw, axis=1) < 0).any(axis=1).mean())
                Q = np.sort(raw, axis=1)
            for k, t in enumerate(QTAG):
                r[f"q_{m}_{t}"] = np.sinh(Q[:, k])
            r[f"p_{m}"] = r[f"q_{m}_50"]
            sec = time.time() - t0
            timers[m] += sec
            fl[f"{m}_seconds"] = round(sec, 1)
            log(f"      {s} fold {f:2d} ({fl['test_from']} -> {fl['test_to']}, n_train {len(trd):,}): {m} {sec:5.0f}s"
                + (f", epochs {fl.get(f'{m}_epochs_run')} (best {fl.get(f'{m}_best_epoch')})" if m in NET else ""))
        rows.append(r); flog.append(fl)
    log(f"   {s:16s} folds={len(rows):3d}  " + "  ".join(f"{m} {t:5.0f}s" for m, t in timers.items()))
    return (pd.concat(rows) if rows else None), flog


# ---------------------------------------------------------------- evaluation
def evaluate(P: pd.DataFrame, s: str, t2: list[str], t1_point: list[str], prob: list[str]):
    L = 4 if s.endswith("|week") else 12
    point = t2 + t1_point
    E = P.dropna(subset=["y", *[f"p_{m}" for m in point], *C.BASELINES]).sort_values("block_start_utc").copy()
    E["year"] = E["block_start_utc"].dt.tz_convert(C.TZ).dt.year
    mp, my, dm, mq = [], [], [], []
    names = {**{f"p_{m}": m for m in point}, **{b: b for b in C.BASELINES}}
    for col, nm in names.items():
        mp.append({"series": s, "model": nm, "tier": 2 if nm in t2 else (1 if nm in t1_point else 0), **C.metrics(E["y"], E[col])})
        for yr, gy in E.groupby("year"):
            my.append({"series": s, "model": nm, "year": yr, **C.metrics(gy["y"], gy[col])})
    y = E["y"].to_numpy(float)
    best_t1 = min(t1_point, key=lambda m: np.abs(E[f"p_{m}"] - E["y"]).mean()) if t1_point else None
    refs = [r for r in ["lear", "lgbm_l1"] if r in t1_point] + ["b_prev_slot"]
    for m in t2:
        e1 = np.abs(E[f"p_{m}"].to_numpy(float) - y)
        vs = refs + ([best_t1] if best_t1 and best_t1 not in refs else []) + [o for o in t2 if o != m]
        for ref in vs:
            rc = ref if ref in C.BASELINES else f"p_{ref}"
            st, p = dm_loss(e1 - np.abs(E[rc].to_numpy(float) - y), L)
            dm.append({"series": s, "loss": "abs", "model": m, "vs": ref, "vs_is_best_tier1": ref == best_t1,
                       "stat": st, "p": p})
    for m in prob:
        pl = np.mean([pinball(y, E[f"q_{m}_{t}"].to_numpy(float), q) for q, t in zip(QS, QTAG)], axis=0)
        row = {"series": s, "model": m, "tier": 2 if m in t2 else 1, "n": len(E), "pinball": pl.mean(), "CRPS_approx": 2 * pl.mean()}
        for lev, (lo_t, hi_t) in INTERVALS.items():
            lo, hi = E[f"q_{m}_{lo_t}"].to_numpy(float), E[f"q_{m}_{hi_t}"].to_numpy(float)
            clo, chi = E[f"cqr{lev}_lo_{m}"].to_numpy(float), E[f"cqr{lev}_hi_{m}"].to_numpy(float)
            h = ~np.isnan(clo)
            row[f"cov{lev}_raw"] = np.mean((y[h] >= lo[h]) & (y[h] <= hi[h])) * 100 if h.any() else np.nan
            row[f"cov{lev}_cqr"] = np.mean((y[h] >= clo[h]) & (y[h] <= chi[h])) * 100 if h.any() else np.nan
            row[f"width{lev}_raw"] = np.mean(hi[h] - lo[h]) if h.any() else np.nan
            row[f"width{lev}_cqr"] = np.mean(chi[h] - clo[h]) if h.any() else np.nan
            row[f"n_cqr{lev}"] = int(h.sum())
        mq.append(row)
        E[f"_pl_{m}"] = pl
    for m in [m for m in prob if m in t2]:
        for ref in [r for r in ["lgbm_q", "cat_q"] if r in prob] + [o for o in prob if o in t2 and o != m]:
            st, p = dm_loss(E[f"_pl_{m}"] - E[f"_pl_{ref}"], L)
            dm.append({"series": s, "loss": "pinball", "model": m, "vs": ref, "vs_is_best_tier1": False, "stat": st, "p": p})
    return mp, my, dm, mq, best_t1


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=C.VIEWS)
    ap.add_argument("--only", nargs="+", default=None, help='series filter(s), e.g. "mFRR|up|4h" "aFRR|down|4h"')
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--refit-months-week", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1, help="average the stochastic models (nets, TabPFN) over n seeds")
    ap.add_argument("--max-folds", type=int, default=0, help="only the first n folds per series (checks)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 folds, 3 epochs, small TabPFN / Chronos context")
    ap.add_argument("--tier1", nargs="*", default=None, help="Tier 1 preds.parquet files / folders (default: Output/rq2_5_<product>, Output/rq2_5)")
    # nets
    ap.add_argument("--device", default="cpu", help="cpu | mps | auto (nets; small nets are often faster on CPU - see tier2_setup_check.py)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--epochs", type=int, default=60, help="max epochs, first fold (cold start)")
    ap.add_argument("--epochs-warm", type=int, default=20, help="max epochs, later folds (warm start)")
    ap.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--final-epochs", type=int, default=2, help="epochs on the full window after early stopping (lr / 3)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tft-hidden", type=int, default=32)
    ap.add_argument("--tft-heads", type=int, default=4)
    ap.add_argument("--tft-lookback", type=int, default=0, help="0 = 4h 42 / day 28 / week 26 grid steps")
    ap.add_argument("--nhits-lookback", type=int, default=0, help="0 = 4h 168 / day 56 / week 52 grid steps")
    ap.add_argument("--net-max-features", type=int, default=0,
                    help="nets: keep only the k inputs most correlated with the target in the first training window (0 = all ~810)")
    ap.add_argument("--nhits-hidden", type=int, default=256)
    ap.add_argument("--nhits-layers", type=int, default=2)
    # TabPFN
    ap.add_argument("--tabpfn-version", default="default", help="default (package default, licence click-through once) | v2 | v2.5 | v2.6 | v3 | v3.5")
    ap.add_argument("--tabpfn-device", default="auto")
    ap.add_argument("--tabpfn-max-train", type=int, default=10000)
    ap.add_argument("--tabpfn-max-features", type=int, default=500)
    ap.add_argument("--tabpfn-estimators", type=int, default=8, help="ensemble members (0 = package default)")
    ap.add_argument("--tabpfn-batch", type=int, default=64, help="test rows per predict call (GPU memory)")
    # Chronos-2
    ap.add_argument("--chronos-model", default="amazon/chronos-2", help="or autogluon/chronos-2-small (28M, faster)")
    ap.add_argument("--chronos-device", default="auto")
    ap.add_argument("--chronos-context", type=int, default=2048, help="max history length in grid steps")
    ap.add_argument("--chronos-batch", type=int, default=64)
    ap.add_argument("--chronos-space", default="asinh", choices=["asinh", "raw"])
    ap.add_argument("--chronos-cov", nargs="*", default=[], help="feature columns passed as known covariates (default none)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick:
        a.max_folds = a.max_folds or 2
        a.epochs, a.epochs_warm, a.patience = 3, 2, 2
        a.tabpfn_max_train, a.tabpfn_estimators = min(a.tabpfn_max_train, 1000), 2
        a.chronos_context = min(a.chronos_context, 256)
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    out = Path(a.out) if a.out else C.ROOT / "Modelling" / "Output" / "rq2_6"
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    log = lambda x: print(x, flush=True)
    log(f"models: {', '.join(LABEL[m] for m in a.models)}; device (nets) {a.device}; seeds {a.seeds}"
        + (f"; max {a.max_folds} folds per series" if a.max_folds else "") + ("  [QUICK MODE - not for the thesis]" if a.quick else ""))

    T1 = load_tier1(a, a.products)
    t_all = time.time()
    allP, MP, MY, DM, MQ, FLOG, BEST = [], [], [], [], [], [], {}
    SS, SM = [], []
    store = {"curves": [], "imp": {}, "att": []}
    for view in a.products:
        df = C.load_view(view)
        feats, _ = feature_list(df, view, "exante")
        X = make_X(df, feats)
        for c in a.chronos_cov:
            if c not in X.columns:
                raise SystemExit(f"--chronos-cov: column {c!r} is not an ex-ante feature of {view}")
        log(f"{view}: {len(df):,} blocks, {len(feats)} features")
        for s in df["series"].unique():
            if a.only and s not in a.only:
                continue
            gi = np.flatnonzero((df["series"] == s).to_numpy())
            g, Xg = df.iloc[gi].reset_index(drop=True), X.iloc[gi].reset_index(drop=True)
            ts = time.time()
            P, flog = run_series(g, Xg, s, a, log, store)
            FLOG += flog
            if P is None:
                log(f"== {s}: no fold with >= {C.MIN_TRAIN[s.split('|')[2]]} training blocks"); continue
            t1p, t1q = [], []
            if T1 is not None:
                P = P.merge(T1, on=["series", "block_start_utc"], how="left")
                t1p = [m for m in T1_POINT if f"p_{m}" in P and P[f"p_{m}"].notna().mean() > 0.95]
                t1q = [m for m in T1_PROB if m in t1p and all(f"q_{m}_{t}" in P for t in QTAG)]
            prob = list(a.models) + t1q
            P = P.sort_values("block_start_utc").reset_index(drop=True)
            for m in prob:
                for lev, (lo_t, hi_t) in INTERVALS.items():
                    P[f"cqr{lev}_lo_{m}"], P[f"cqr{lev}_hi_{m}"] = cqr(P, m, lo_t, hi_t, lev)
            allP.append(P)
            mp, my, dm, mq, best = evaluate(P, s, list(a.models), t1p, prob)
            MP += mp; MY += my; DM += dm; MQ += mq; BEST[s] = best
            ss, sm = seed_spread(P, s, list(a.models), 4 if s.endswith("|week") else 12)
            SS += ss; SM += sm
            own = [c for c in P.columns if not any(c.endswith(f"_{m}") or f"_{m}_" in c for m in T1_POINT)]
            pd.concat([x[[c for c in own if c in x.columns]] for x in allP]).to_parquet(out / "preds_partial.parquet")
            dur = time.time() - ts
            if dur > 600:
                mm = pd.DataFrame(mp).set_index("model")["MAE"]
                bm = mm.drop(index=C.BASELINES, errors="ignore").idxmin()
                RU.notify_mac("RQ2 Tier 2: series done", f"{s}: best {bm}, {(1 - mm[bm] / mm['b_prev_slot']) * 100:+.1f}% vs naive",
                              subtitle=f"{RU.fmt_duration(dur)}", sound="")

    if not MP:
        print("nothing evaluated"); return
    MP, MY, DM, MQ = map(pd.DataFrame, (MP, MY, DM, MQ))
    Pall = pd.concat(allP)
    own = [c for c in Pall.columns if not any(c.endswith(f"_{m}") or f"_{m}_" in c for m in T1_POINT)]
    Pall[own].to_parquet(out / "preds.parquet")
    MP.to_csv(out / "metrics_point.csv", index=False); MY.to_csv(out / "metrics_by_year.csv", index=False)
    DM.to_csv(out / "dm.csv", index=False); MQ.to_csv(out / "metrics_prob.csv", index=False)
    pd.DataFrame(FLOG).to_csv(out / "fold_log.csv", index=False)
    SS, SM = pd.DataFrame(SS), pd.DataFrame(SM)
    if len(SS):
        SS.to_csv(out / "seed_spread.csv", index=False); SM.to_csv(out / "seed_mae.csv", index=False)
    if store["curves"]:
        pd.DataFrame(store["curves"]).to_csv(out / "training_curves.csv", index=False)
    if store["att"]:
        AT = pd.DataFrame(store["att"]); AT.to_csv(out / "tft_attention.csv", index=False)
        _plot_attention(AT, out)
    if store["curves"]:
        _plot_curves(pd.DataFrame(store["curves"]), out)

    import ml_diagnostics as D
    imp_tables = {m: D.importance_table(d) for m, d in store["imp"].items() if any(d.values())}
    point_all = [m for m in [*a.models, *T1_POINT] if f"p_{m}" in Pall and Pall[f"p_{m}"].notna().any()]
    models = {m: f"p_{m}" for m in point_all} | {b: b for b in C.BASELINES}
    quant = {m: {q: f"q_{m}_{t}" for q, t in zip(QS, QTAG)} for m in [*a.models, *T1_PROB]
             if all(f"q_{m}_{t}" in Pall for t in QTAG)}
    D.forecast_diagnostics(Pall, models, out, baseline="b_prev_slot", quantiles=quant, importance=imp_tables,
                           group_cols=[c for c in Pall.columns if c.startswith("regime_")], title="RQ2 Tier 2")

    # ---------------------------------------------------------------- report
    hdr = ["RQ2 Tier 2 (ML_Options.md) - advanced models vs Tier 1 and the naive baselines.",
           f"Rolling origin, expanding window, refit every {a.refit_months} months (weekly {a.refit_months_week}); test from "
           "2021-01-01; errors in EUR/MWh on the common rows of all point models and baselines.",
           f"Tier 2: {', '.join(LABEL[m] for m in a.models)}; seeds {a.seeds}; nets: device {a.device}, max epochs "
           f"{a.epochs} / warm {a.epochs_warm}, patience {a.patience}, warm start {a.warm_start}, final epochs {a.final_epochs}, "
           f"TFT hidden {a.tft_hidden} heads {a.tft_heads}, NHITS hidden {a.nhits_hidden}, lr {a.lr}, wd {a.wd}, dropout {a.dropout}, "
           f"inputs {a.net_max_features or 'all'}; TabPFN {a.tabpfn_version}, "
           f"max train {a.tabpfn_max_train}, max features {a.tabpfn_max_features}; Chronos {a.chronos_model}, context "
           f"{a.chronos_context}, covariates {a.chronos_cov or 'none'}.",
           "Tier 1 (merged from rq2_5): lear, lgbm_l1 (round 1), lgbm_q, cat_q, qrf, svr. best_t1 = Tier 1 model with the lowest "
           "MAE on these rows (chosen ex post, a strict reference)." + ("\n*** QUICK MODE: few folds / epochs - not for the thesis ***" if a.quick else ""), ""]
    order_all = [*a.models, *T1_POINT]
    for s, g in MP.groupby("series", sort=False):
        g = g.set_index("model")
        order = [m for m in order_all if m in g.index] + C.BASELINES
        hdr.append(f"== {s}   n = {int(g['n'].iloc[0]):,}   best Tier 1 = {BEST.get(s)}")
        T = g.loc[order, ["MAE", "RMSE", "sMAPE_%", "MAE_asinh"]].copy()
        T["skill_vs_prev_slot_%"] = (1 - T["MAE"] / g.at["b_prev_slot", "MAE"]) * 100
        if "lear" in g.index:
            T["skill_vs_LEAR_%"] = (1 - T["MAE"] / g.at["lear", "MAE"]) * 100
        if BEST.get(s):
            T["skill_vs_best_t1_%"] = (1 - T["MAE"] / g.at[BEST[s], "MAE"]) * 100
        hdr.append(C.fmt(T))
        Dd = DM[(DM.series == s) & (DM.loss == "abs")]
        if len(Dd):
            hdr.append("   Diebold-Mariano |error| (stat < 0: row model better):")
            hdr.append("   " + C.fmt(Dd.pivot(index="model", columns="vs", values="stat").round(2)).replace("\n", "\n   "))
            hdr.append("   p-values:")
            hdr.append("   " + C.fmt(Dd.pivot(index="model", columns="vs", values="p")).replace("\n", "\n   "))
        Q = MQ[MQ.series == s].set_index("model") if len(MQ) else pd.DataFrame()
        if len(Q):
            hdr.append("   Probabilistic (pinball / CRPS in EUR/MWh; coverage %, width EUR/MWh on CQR-eligible rows):")
            hdr.append("   " + C.fmt(Q.drop(columns=["series", "n"])).replace("\n", "\n   "))
            Dp = DM[(DM.series == s) & (DM.loss == "pinball")]
            if len(Dp):
                hdr.append("   DM pinball: " + "; ".join(f"{r.model} vs {r.vs} stat {r.stat:.2f} p {r.p:.4f}" for r in Dp.itertuples()))
        Y = MY[MY.series == s].pivot(index="year", columns="model", values="MAE")
        hdr.append("   MAE by year:")
        hdr.append("   " + C.fmt(Y[[c for c in order if c in Y.columns]]).replace("\n", "\n   "))
        hdr.append("")
    S = MP.pivot(index="series", columns="model", values="MAE")
    cols = [m for m in order_all if m in S.columns]
    sk = pd.DataFrame({m: (1 - S[m] / S["b_prev_slot"]) * 100 for m in cols}, index=S.index)
    hdr.append("== Summary: MAE skill vs same-slot-yesterday (%)")
    hdr.append(C.fmt(sk))
    hdr.append("   best point model per series: " + "; ".join(f"{i}: {S.loc[i, cols].idxmin()}" for i in S.index))
    if len(MQ):
        hdr.append("== Summary: mean pinball loss (EUR/MWh), lower is better")
        hdr.append(C.fmt(MQ.pivot(index="series", columns="model", values="pinball")))
        hdr.append("== Summary: 90 % interval coverage raw -> CQR (%)")
        cv = MQ.assign(c=lambda d: d["cov90_raw"].round(1).astype(str) + " -> " + d["cov90_cqr"].round(1).astype(str))
        hdr.append(cv.pivot(index="series", columns="model", values="c").to_string())
    for m, t in imp_tables.items():
        if len(t) == 0:
            continue
        lab = "TFT known inputs (VSN weight)" if m == "tft" else "TFT past inputs (VSN weight)"
        hdr.append(f"== Top features: {lab} (mean share over folds; % of folds in top 20)")
        for s_, g_ in t.groupby("series", sort=False):
            g_ = g_[g_["mean"] > 0]
            hdr.append(f"   {s_}: " + "; ".join(f"{r.feature} {r.mean * 100:.1f}% ({r.top20_share_of_folds:.0%})"
                                                for r in g_.head(10).itertuples()))
    if store["att"]:
        AT = pd.DataFrame(store["att"])
        hdr.append("== TFT attention: share on the 6 most attended positions (steps before auction start; 0 = target block)")
        for s_, g_ in AT.groupby("series", sort=False):
            m_ = g_.groupby("steps_before_auction_start")["weight"].mean().sort_values(ascending=False)
            hdr.append(f"   {s_}: " + "; ".join(f"{int(k)}: {v:.3f}" for k, v in m_.head(6).items()))
    hdr.append("\nMore detail: fold_log.csv (per fold sizes / timings / epochs), training_curves.csv (+ plots/), "
               "tft_attention.csv, diagnostics/ (breakdowns, worst errors, calibration, importance, plots; see README.txt)")
    if len(SS):
        hdr.append(f"\n== Seed robustness ({a.seeds} seeds; MAE per seed, spread, gain of the seed average; seed_spread.csv, seed_mae.csv)")
        hdr.append(C.fmt(SS.set_index(["series", "model"])))
    for i_ in S.index:
        t2c = [m for m in a.models if m in S.columns]
        if not t2c:
            continue
        b2 = S.loc[i_, t2c].idxmin()
        msg = f"{i_.replace('|', ' ')}: {b2} {(1 - S.loc[i_, b2] / S.loc[i_, 'b_prev_slot']) * 100:+.1f}%"
        if BEST.get(i_):
            msg += f" (best T1 {BEST[i_]} {(1 - S.loc[i_, BEST[i_]] / S.loc[i_, 'b_prev_slot']) * 100:+.1f}%)"
        RU.summary(msg)
    hdr.append(f"\ntotal runtime {RU.fmt_duration(time.time() - t_all)}")
    (out / "report_rq2_6.txt").write_text("\n".join(hdr))
    print("\n".join(hdr))


def _plot_attention(AT: pd.DataFrame, out: Path) -> None:
    try:
        import ml_diagnostics as D
        plt = D._plt()
        for s, g in AT.groupby("series"):
            m = g.groupby("steps_before_auction_start")["weight"].agg(["mean", "std"]).sort_index()
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(m.index, m["mean"], yerr=m["std"], color="#2f6db3", alpha=0.85)
            ax.set_xlabel("grid steps before auction start (0 = target block)"); ax.set_ylabel("attention weight")
            ax.set_title(f"TFT attention by lag, mean over folds: {s}")
            d = out / "diagnostics" / "plots" / D._safe(s); d.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(d / "tft_attention.png"); plt.close(fig)
    except Exception as e:                           # noqa: BLE001
        print(f"   attention plot skipped: {e}")


def _plot_curves(CV: pd.DataFrame, out: Path) -> None:
    try:
        import ml_diagnostics as D
        plt = D._plt()
        for (s, m), g in CV[CV.phase == "early_stop"].groupby(["series", "model"]):
            fig, ax = plt.subplots(figsize=(8, 3.2))
            folds = sorted(g["fold"].unique())
            cmap = plt.get_cmap("viridis")
            for i, f in enumerate(folds):
                gf = g[(g.fold == f) & (g.seed == g.seed.min())]
                ax.plot(gf["epoch"], gf["val_loss"], color=cmap(i / max(1, len(folds) - 1)), lw=1)
            ax.set_xlabel("epoch"); ax.set_ylabel("validation pinball (scaled)")
            ax.set_title(f"{m}: validation loss per fold (dark = early folds): {s}")
            d = out / "diagnostics" / "plots" / D._safe(s); d.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(d / f"training_curves_{m}.png"); plt.close(fig)
    except Exception as e:                           # noqa: BLE001
        print(f"   training-curve plot skipped: {e}")


if __name__ == "__main__":
    RU.run(main, "RQ2 Tier 2 models")
