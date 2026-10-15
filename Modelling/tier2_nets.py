"""Tier 2 neural forecasters for RQ2 (ML_Options.md): Temporal Fusion Transformer and NHITS-x,
written directly in PyTorch so they use exactly the same series, folds, ex-ante features and
delta target as rq2_2 / rq2_5 (no pytorch-forecasting / neuralforecast data pipeline in between).

One training sample = one auction block. Inputs:
  * history: the regular block grid of the series (4h: 6 steps per day, day: 1, week: 1) up to the
    first block of the block's own auction, i.e. only prices that were public at gate closure.
    Blocks that were not procured are missing (value 0 + mask 0).
  * known inputs of the target block: the full ex-ante feature vector (same columns as LEAR / SVR:
    numeric + asinh of lagged prices + one-hot categories + slot / weekday dummies), median-imputed
    and standardised with training-window statistics.
  * target: asinh(price) - asinh(same slot yesterday), standardised with training statistics
    (the same delta target as the tree models). Output: 11 quantiles, monotone by construction.

TFT (Lim et al. 2021), single-horizon variant: variable selection network (VSN) over the known inputs
of the target block, VSN over the past inputs (observed target, its delta, missing masks, calendar),
LSTM encoder-decoder, gated skip connections, interpretable multi-head attention (shared value
projection, head-averaged weights), position-wise GRN, quantile head. No static covariates
(one model per series). Interpretability: VSN weights (variable importance) and attention by lag.

NHITS-x (Challu et al. 2023, with exogenous inputs as in N-BEATSx, Olivares et al. 2023): stacks of
MLP blocks on multi-rate max-pooled input windows (e.g. 4h: pooling 6 / 3 / 1 = daily, half-day,
raw), doubly residual (backcast removed from the window, forecasts summed), backcast produced by
linear interpolation of a coarse coefficient vector (hierarchical interpolation). The known inputs
of the target block are concatenated to every block input. Horizon = 1 block, so the forecast-side
interpolation is trivial.

Training: AdamW, pinball loss averaged over the quantile levels, early stopping on the last 15 % (in
time) of the training window, best weights restored, then --final-epochs epochs on the full
training window at 1/3 of the learning rate (so the most recent blocks are also learned). With warm
start (default) each fold starts from the previous fold's weights (that model only saw data before
the previous fold start, so no look-ahead) and uses the shorter --epochs-warm budget.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TZ = "Europe/Zurich"
STEPS_PER_DAY = {"4h": 6, "day": 1, "week": 1}
GRID_ORIGIN = pd.Timestamp("2014-12-29")                 # a Monday before the first block
ENC_VARS = ["y_level", "y_observed", "y_delta", "y_delta_observed", "slot_sin", "slot_cos",
            "dow_sin", "dow_cos", "season_sin", "season_cos"]


# ====================================================================== data
def grid_index(g: pd.DataFrame, proc: str) -> np.ndarray:
    """Position of every block on a regular grid (4h: day * 6 + slot; day: day; week: week)."""
    loc = g["block_start_utc"].dt.tz_convert(TZ).dt.tz_localize(None)
    days = ((loc.dt.normalize() - GRID_ORIGIN).dt.days).to_numpy()
    if proc == "4h":
        slot = np.clip(np.round(loc.dt.hour.to_numpy() / 4).astype(int), 0, 5)
        return days * 6 + slot
    if proc == "week":
        return days // 7
    return days


def grid_calendar(n: int, proc: str) -> np.ndarray:
    """Calendar inputs (slot, weekday, season as sin / cos) for grid positions 0..n-1."""
    p = np.arange(n)
    spd = STEPS_PER_DAY[proc]
    if proc == "week":
        day = p * 7.0
        slot = np.zeros(n)
    else:
        day = p / spd
        slot = (p % spd) / spd
    dow = np.floor(day) % 7 / 7.0
    season = (day % 365.25) / 365.25
    two = 2 * np.pi
    return np.column_stack([np.sin(two * slot), np.cos(two * slot), np.sin(two * dow), np.cos(two * dow),
                            np.sin(two * season), np.cos(two * season)]).astype(np.float32)


def auction_start(g: pd.DataFrame, gidx: np.ndarray) -> np.ndarray:
    """Grid position of the first block of each row's auction = end (exclusive) of the history
    that was public at gate closure."""
    s = pd.Series(gidx, index=g.index)
    return s.groupby(g["auction_id"].astype(str).to_numpy()).transform("min").to_numpy()


def net_prep(Z: pd.DataFrame, tr: np.ndarray) -> np.ndarray:
    """All rows of Z median-imputed, clipped to the training range and standardised with training
    statistics. Keeps every column (fixed width across folds, needed for warm start)."""
    Ztr = Z.iloc[tr]
    med = Ztr.median().fillna(0.0)
    lo, hi = Ztr.min().fillna(0.0), Ztr.max().fillna(0.0)
    A = Z.fillna(med).clip(lo, hi, axis=1)
    mu = A.iloc[tr].mean()
    sd = A.iloc[tr].std().fillna(1.0)
    sd[sd < 1e-9] = 1.0
    return ((A - mu) / sd).to_numpy(np.float32)


def feature_origin(cols: list[str], X: pd.DataFrame) -> list[str]:
    """Map design-matrix columns (one-hot dummies, slot_/dow_ dummies) back to the original feature."""
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    out = []
    for c in cols:
        if c in X.columns:
            out.append(c)
            continue
        m = [k for k in cat if c.startswith(k + "_")]
        if m:
            out.append(max(m, key=len))
        elif c.startswith("slot_"):
            out.append("slot")
        elif c.startswith("dow_"):
            out.append("weekday")
        else:
            out.append(c)
    return out


class FoldData:
    """Grid arrays for one series and one fold (scaled with that fold's training statistics)."""

    def __init__(self, A: np.ndarray, ya: np.ndarray, yd: np.ndarray, gidx: np.ndarray, e0: np.ndarray,
                 proc: str, tr: np.ndarray, pad: int):
        self.A, self.e0, self.pad = A, e0, pad
        ok_y = ~np.isnan(ya[tr])
        self.mu_y, self.sd_y = float(np.nanmean(ya[tr])), float(np.nanstd(ya[tr][ok_y]) or 1.0)
        self.mu_d, self.sd_d = float(np.nanmean(yd[tr])), float(np.nanstd(yd[tr]) or 1.0)
        G = int(gidx.max()) + 1 + pad
        self.gy, self.gm = np.zeros(G, np.float32), np.zeros(G, np.float32)
        self.gd, self.gdm = np.zeros(G, np.float32), np.zeros(G, np.float32)
        ys = (ya - self.mu_y) / self.sd_y
        ds = (yd - self.mu_d) / self.sd_d
        okd = ~np.isnan(ds)
        oky = ~np.isnan(ys)
        self.gy[gidx[oky] + pad], self.gm[gidx[oky] + pad] = ys[oky], 1.0
        self.gd[gidx[okd] + pad], self.gdm[gidx[okd] + pad] = ds[okd], 1.0
        cal = grid_calendar(G - pad + 1, proc)
        self.gcal = np.vstack([np.zeros((pad, cal.shape[1]), np.float32), cal])
        self.target = np.where(okd, ds, 0.0).astype(np.float32)

    def window(self, rows: np.ndarray, L: int):
        """(B, L) positions of the L grid steps before each row's auction start."""
        return (self.e0[rows][:, None] - L + np.arange(L)[None, :]) + self.pad

    def tft_batch(self, rows, L, dev):
        p = self.window(rows, L)
        enc = np.concatenate([self.gy[p][..., None], self.gm[p][..., None], self.gd[p][..., None],
                              self.gdm[p][..., None], self.gcal[p]], axis=-1)
        return (torch.from_numpy(enc).to(dev), torch.from_numpy(self.A[rows]).to(dev))

    def nhits_batch(self, rows, L, dev):
        p = self.window(rows, L)
        return (torch.from_numpy(self.gy[p]).to(dev), torch.from_numpy(self.gm[p]).to(dev),
                torch.from_numpy(self.A[rows]).to(dev))

    def y(self, rows, dev):
        return torch.from_numpy(self.target[rows]).to(dev)

    def unscale(self, q: np.ndarray) -> np.ndarray:
        """Scaled delta quantiles -> asinh(price) delta."""
        return q * self.sd_d + self.mu_d


# ====================================================================== common layers
def monotone_quantiles(raw: torch.Tensor, med: int) -> torch.Tensor:
    """raw (B, Q) -> non-crossing quantiles: median + cumulative softplus increments."""
    m = raw[:, med:med + 1]
    up = torch.cumsum(F.softplus(raw[:, med + 1:]), 1)
    dn = torch.cumsum(F.softplus(raw[:, :med].flip(1)), 1).flip(1)
    return torch.cat([m - dn, m, m + up], 1)


def pinball_loss(q: torch.Tensor, y: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    d = y[:, None] - q
    return torch.maximum(levels * d, (levels - 1) * d).mean()


class GLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, -1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated residual network (TFT eq. 2-4)."""

    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else None
        self.fc1, self.fc2 = nn.Linear(d_in, d_hidden), nn.Linear(d_hidden, d_hidden)
        self.drop, self.glu, self.ln = nn.Dropout(dropout), GLU(d_hidden, d_out), nn.LayerNorm(d_out)

    def forward(self, x):
        r = self.skip(x) if self.skip is not None else x
        h = self.drop(self.fc2(F.elu(self.fc1(x))))
        return self.ln(r + self.glu(h))


class VarGRN(nn.Module):
    """V independent GRNs (one per variable, d -> d), evaluated in parallel with einsum."""

    def __init__(self, V, d, dropout):
        super().__init__()
        s = 1 / math.sqrt(d)
        self.W1 = nn.Parameter(torch.randn(V, d, d) * s); self.b1 = nn.Parameter(torch.zeros(V, d))
        self.W2 = nn.Parameter(torch.randn(V, d, d) * s); self.b2 = nn.Parameter(torch.zeros(V, d))
        self.W3 = nn.Parameter(torch.randn(V, d, 2 * d) * s); self.b3 = nn.Parameter(torch.zeros(V, 2 * d))
        self.drop, self.ln = nn.Dropout(dropout), nn.LayerNorm(d)

    def forward(self, e):                                   # e (..., V, d)
        h = F.elu(torch.einsum("...vd,vdk->...vk", e, self.W1) + self.b1)
        h = self.drop(torch.einsum("...vd,vdk->...vk", h, self.W2) + self.b2)
        a, b = (torch.einsum("...vd,vdk->...vk", h, self.W3) + self.b3).chunk(2, -1)
        return self.ln(e + a * torch.sigmoid(b))


class VSN(nn.Module):
    """Variable selection network: scalar inputs -> per-variable embeddings -> softmax weights."""

    def __init__(self, V, d, dropout):
        super().__init__()
        self.emb_w = nn.Parameter(torch.randn(V, d) * 0.1)
        self.emb_b = nn.Parameter(torch.zeros(V, d))
        self.select = GRN(V * d, d, V, dropout)
        self.var = VarGRN(V, d, dropout)

    def forward(self, x):                                   # x (..., V)
        e = x.unsqueeze(-1) * self.emb_w + self.emb_b       # (..., V, d)
        w = torch.softmax(self.select(e.flatten(-2)), -1)   # (..., V)
        return (w.unsqueeze(-1) * self.var(e)).sum(-2), w


# ====================================================================== TFT
class TFT(nn.Module):
    def __init__(self, n_known, n_past, n_q, med, d=32, heads=4, dropout=0.1):
        super().__init__()
        self.med = med
        self.vsn_known, self.vsn_past = VSN(n_known, d, dropout), VSN(n_past, d, dropout)
        self.enc, self.dec = nn.LSTM(d, d, batch_first=True), nn.LSTM(d, d, batch_first=True)
        self.lstm_gate, self.lstm_ln = GLU(d, d), nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, dropout)
        self.heads, self.dh = heads, max(4, d // heads)
        self.wq, self.wk = nn.Linear(d, heads * self.dh), nn.Linear(d, heads * self.dh)
        self.wv, self.wo = nn.Linear(d, self.dh), nn.Linear(self.dh, d)
        self.att_gate, self.att_ln = GLU(d, d), nn.LayerNorm(d)
        self.pos_grn = GRN(d, d, d, dropout)
        self.out_gate, self.out_ln = GLU(d, d), nn.LayerNorm(d)
        self.head = nn.Linear(d, n_q)
        self.drop = nn.Dropout(dropout)

    def forward(self, enc_x, known_x, return_weights=False):
        B, L, _ = enc_x.shape
        xe, w_past = self.vsn_past(enc_x)                   # (B, L, d), (B, L, Vp)
        xd, w_known = self.vsn_known(known_x)               # (B, d), (B, Vk)
        xd = xd.unsqueeze(1)
        he, state = self.enc(xe)
        hd, _ = self.dec(xd, state)
        x_in = torch.cat([xe, xd], 1)                       # (B, L+1, d)
        h = self.lstm_ln(x_in + self.lstm_gate(self.drop(torch.cat([he, hd], 1))))
        phi = self.enrich(h)
        q = self.wq(phi[:, -1:]).view(B, 1, self.heads, self.dh).transpose(1, 2)        # (B, H, 1, dh)
        k = self.wk(phi).view(B, L + 1, self.heads, self.dh).transpose(1, 2)            # (B, H, L+1, dh)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), -1)           # (B, H, 1, L+1)
        att_m = att.mean(1)                                                             # (B, 1, L+1)
        a = self.wo(att_m @ self.wv(phi))                                               # (B, 1, d)
        z = self.att_ln(phi[:, -1:] + self.att_gate(self.drop(a)))
        z = self.pos_grn(z)
        z = self.out_ln(h[:, -1:] + self.out_gate(z))[:, 0]
        out = monotone_quantiles(self.head(z), self.med)
        if return_weights:
            return out, {"known": w_known, "past": w_past.mean(1), "attention": att_m[:, 0]}
        return out


# ====================================================================== NHITS-x
class NHITSBlock(nn.Module):
    def __init__(self, L, n_exog, pool, hidden, n_q, dropout, layers=2):
        super().__init__()
        self.pool, self.L = pool, L
        lp = math.ceil(L / pool)
        mods, d_in = [], 2 * lp + n_exog
        for _ in range(layers):
            mods += [nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d_in = hidden
        self.mlp = nn.Sequential(*mods)
        self.n_theta = max(2, L // pool)
        self.back, self.fore = nn.Linear(hidden, self.n_theta), nn.Linear(hidden, n_q)

    def forward(self, res, mask, exog):
        xp = F.max_pool1d(res.unsqueeze(1), self.pool, self.pool, ceil_mode=True)[:, 0]
        mp = F.avg_pool1d(mask.unsqueeze(1), self.pool, self.pool, ceil_mode=True)[:, 0]
        h = self.mlp(torch.cat([xp, mp, exog], 1))
        back = F.interpolate(self.back(h).unsqueeze(1), size=self.L, mode="linear", align_corners=True)[:, 0]
        return back * mask, self.fore(h)


class NHITSx(nn.Module):
    def __init__(self, L, n_exog, n_q, med, pools=(6, 3, 1), hidden=256, dropout=0.1, layers=2):
        super().__init__()
        self.med = med
        self.blocks = nn.ModuleList([NHITSBlock(L, n_exog, p, hidden, n_q, dropout, layers) for p in pools])

    def forward(self, y, mask, exog):
        res, out = y * mask, 0
        for b in self.blocks:
            back, f = b(res, mask, exog)
            res = res - back
            out = out + f
        return monotone_quantiles(out, self.med)


# ====================================================================== training
def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        print("   [tier2] MPS not available, using CPU")
        name = "cpu"
    return torch.device(name)


def _forward(model, kind, data: FoldData, rows, L, dev):
    if kind == "tft":
        enc, known = data.tft_batch(rows, L, dev)
        return model(enc, known)
    y, m, ex = data.nhits_batch(rows, L, dev)
    return model(y, m, ex)


def _eval_loss(model, kind, data, rows, L, dev, levels, bs=1024):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            tot += float(pinball_loss(_forward(model, kind, data, r, L, dev), data.y(r, dev), levels)) * len(r)
            n += len(r)
    return tot / max(n, 1)


def train(model, kind, data: FoldData, tr: np.ndarray, L: int, cfg, seed: int, levels: torch.Tensor,
          dev, max_epochs: int, curve: list, tag: dict):
    """Early stopping on the last 15 % of tr (time order), restore best, then final epochs on all of tr."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    nv = max(10, int(0.15 * len(tr)))
    fit, val = tr[:-nv], tr[-nv:]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    best, best_state, best_ep, bad, ep = np.inf, None, 0, 0, 0
    t0 = time.time()

    def one_epoch(rows):
        model.train()
        perm = rng.permutation(rows)
        tot = 0.0
        for i in range(0, len(perm), cfg.batch):
            r = perm[i:i + cfg.batch]
            if len(r) < 2:
                continue
            loss = pinball_loss(_forward(model, kind, data, r, L, dev), data.y(r, dev), levels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(r)
        return tot / len(perm)

    for ep in range(1, max_epochs + 1):
        trl = one_epoch(fit)
        vl = _eval_loss(model, kind, data, val, L, dev, levels)
        curve.append({**tag, "phase": "early_stop", "epoch": ep, "train_loss": trl, "val_loss": vl,
                      "seconds": round(time.time() - t0, 1)})
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    for g in opt.param_groups:
        g["lr"] = cfg.lr / 3
    for e in range(1, cfg.final_epochs + 1):
        trl = one_epoch(tr)
        curve.append({**tag, "phase": "final_full_window", "epoch": ep + e, "train_loss": trl, "val_loss": np.nan,
                      "seconds": round(time.time() - t0, 1)})
    return {"epochs_run": ep, "best_epoch": best_ep, "best_val_pinball_scaled": best,
            "train_seconds": round(time.time() - t0, 1)}


def predict(model, kind, data: FoldData, rows, L, dev, bs=1024, weights=False):
    model.eval()
    outs, ws = [], {"known": [], "past": [], "attention": []}
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            if kind == "tft" and weights:
                enc, known = data.tft_batch(r, L, dev)
                q, w = model(enc, known, return_weights=True)
                for k in ws:
                    ws[k].append(w[k].cpu().numpy())
            else:
                q = _forward(model, kind, data, r, L, dev)
            outs.append(q.cpu().numpy())
    Q = data.unscale(np.concatenate(outs))
    if kind == "tft" and weights:
        return Q, {k: np.concatenate(v) for k, v in ws.items()}
    return Q, None
