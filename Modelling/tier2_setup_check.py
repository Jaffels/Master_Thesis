"""Tier 2 setup check (run once before rq2_6_tier2.py; takes 2-10 min, mostly downloads).

1. versions: Python, torch, tabpfn, chronos-forecasting, transformers; Apple-GPU (MPS) available?
2. speed: one TFT and one NHITS-x training epoch on random data shaped like mFRR 4h (810 inputs),
   on CPU and on MPS -> recommended --device and a rough runtime estimate for rq2_6 (mFRR up 4h).
3. TabPFN: tiny fit + quantile prediction. The first call downloads the weights; v2.5 and newer
   open a browser page once to accept the Prior Labs licence (free for research). If that is not
   possible, use --tabpfn-version v2 here and in rq2_6 (no licence step).
4. Chronos-2: downloads the model (~0.5 GB for amazon/chronos-2) and forecasts a test sine.
Writes Modelling/Output/tier2_setup/{run.log, run_meta.json, setup_check.json}; macOS notification at the end.

    python Modelling/tier2_setup_check.py
    python Modelling/tier2_setup_check.py --tabpfn-version v2 --skip-chronos
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import run_utils as RU

HERE = Path(__file__).resolve().parent


def _v(p):
    try:
        return metadata.version(p)
    except Exception:
        return None


def bench(dev_name: str, n: int = 2048, F: int = 810) -> dict:
    import torch
    import tier2_nets as N
    from rq2_5_tier1 import MED, QS
    dev = torch.device(dev_name)
    rng = np.random.default_rng(0)
    res = {}
    for kind in ["tft", "nhits"]:
        L = 42 if kind == "tft" else 168
        A = rng.standard_normal((n, F)).astype(np.float32)
        G = n + 400
        gidx = np.arange(200, 200 + n)
        e0 = (gidx // 6) * 6
        ya = rng.standard_normal(n); yd = rng.standard_normal(n)
        data = N.FoldData(A, ya, yd, gidx, e0, "4h", np.arange(n), pad=L)
        net = (N.TFT(F, len(N.ENC_VARS), len(QS), MED) if kind == "tft" else N.NHITSx(L, F, len(QS), MED)).to(dev)
        lv = torch.tensor(QS, dtype=torch.float32, device=dev)
        cfg = argparse.Namespace(lr=1e-3, wd=1e-4, batch=256, patience=99, final_epochs=0)
        N.train(net, kind, data, np.arange(n), L, cfg, seed=1, levels=lv, dev=dev, max_epochs=1, curve=[], tag={})  # warm-up
        t0 = time.time()
        N.train(net, kind, data, np.arange(n), L, cfg, seed=1, levels=lv, dev=dev, max_epochs=2, curve=[], tag={})
        if dev.type == "mps":
            torch.mps.synchronize()
        res[kind] = (time.time() - t0) / 2 / n * 1000           # seconds per 1,000 samples and epoch
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabpfn-version", default="default")
    ap.add_argument("--chronos-model", default="amazon/chronos-2")
    ap.add_argument("--skip-tabpfn", action="store_true")
    ap.add_argument("--skip-chronos", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    a = ap.parse_args()
    out = HERE / "Output" / "tier2_setup"
    out.mkdir(parents=True, exist_ok=True)
    RU.set_output(out)
    rep: dict = {"python": sys.version.split()[0], "platform": platform.platform(),
                 "packages": {p: _v(p) for p in ["torch", "tabpfn", "chronos-forecasting", "transformers", "accelerate",
                                                 "numpy", "pandas", "scikit-learn"]}}
    print("versions:", json.dumps(rep["packages"], indent=1))
    problems = []
    import torch
    rep["mps_available"] = bool(torch.backends.mps.is_available())
    print(f"torch {torch.__version__}, MPS (Apple GPU) available: {rep['mps_available']}, CPU threads {torch.get_num_threads()}")

    if not a.skip_bench:
        devs = ["cpu"] + (["mps"] if rep["mps_available"] else [])
        rep["bench_s_per_1000_samples_epoch"] = {}
        for d in devs:
            try:
                b = bench(d)
                rep["bench_s_per_1000_samples_epoch"][d] = b
                print(f"speed on {d}: TFT {b['tft']:.2f} s, NHITS-x {b['nhits']:.2f} s per 1,000 samples and epoch")
            except Exception as e:                       # noqa: BLE001
                print(f"speed on {d}: FAILED ({type(e).__name__}: {e})")
                if d == "cpu":
                    problems.append(f"nets on cpu: {e}")
        B = rep["bench_s_per_1000_samples_epoch"]
        if B:
            best = min(B, key=lambda d: B[d]["tft"])
            rep["recommended_device"] = best
            # mFRR up 4h: ~22 folds, training window 10k -> 18k blocks (mean ~14k), ~25 epochs cold, ~12 warm (+2 final)
            ep = 25 + 21 * 14
            est = {k: B[best][k] * 14 * ep / 3600 for k in ["tft", "nhits"]}
            rep["estimate_hours_mfrr_up_4h"] = est
            print(f"-> use --device {best}. Rough estimate for mFRR up 4h (one seed): TFT {est['tft']:.1f} h, "
                  f"NHITS-x {est['nhits']:.1f} h (early stopping usually ends sooner)")

    if not a.skip_tabpfn:
        try:
            from tabpfn import TabPFNRegressor
            t0 = time.time()
            kw = dict(n_estimators=2, random_state=0, ignore_pretraining_limits=True)
            if a.tabpfn_version == "default":
                m = TabPFNRegressor(**kw)
            else:
                from tabpfn.constants import ModelVersion
                m = TabPFNRegressor.create_default_for_version(ModelVersion(a.tabpfn_version), **kw)
            X = np.random.default_rng(0).standard_normal((300, 12)).astype(np.float32)
            y = (X[:, 0] * 2 + np.random.default_rng(1).standard_normal(300) * 0.3).astype(np.float32)
            m.fit(X[:250], y[:250])
            q = m.predict(X[250:], output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
            cov = np.mean((y[250:] >= q[0]) & (y[250:] <= q[2]))
            rep["tabpfn"] = {"ok": True, "seconds": round(time.time() - t0, 1), "test_80pct_coverage": float(cov),
                             "version": a.tabpfn_version, "device": str(getattr(m, "devices_", "?"))}
            print(f"TabPFN ok ({a.tabpfn_version}): {time.time() - t0:.0f} s incl. download, 80 % interval coverage on toy data {cov:.0%}")
        except Exception as e:                           # noqa: BLE001
            rep["tabpfn"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            problems.append("TabPFN")
            print(f"TabPFN FAILED: {type(e).__name__}: {e}\n  -> if this is the licence / download step, run again in a "
                  "normal Terminal (browser opens), or use --tabpfn-version v2")

    if not a.skip_chronos:
        try:
            from chronos import Chronos2Pipeline
            t0 = time.time()
            dev = "mps" if rep["mps_available"] else "cpu"
            try:
                pipe = Chronos2Pipeline.from_pretrained(a.chronos_model, device_map=dev)
            except Exception:
                dev = "cpu"
                pipe = Chronos2Pipeline.from_pretrained(a.chronos_model, device_map=dev)
            x = np.sin(np.arange(600) * 2 * np.pi / 6) + 0.05 * np.random.default_rng(0).standard_normal(600)
            q, _ = pipe.predict_quantiles([x[:594].astype(np.float32)], prediction_length=6, quantile_levels=[0.1, 0.5, 0.9])
            med = np.asarray(q[0].cpu().numpy())[0, :, 1]
            err = float(np.abs(med - x[594:]).mean())
            rep["chronos"] = {"ok": True, "seconds": round(time.time() - t0, 1), "device": dev, "toy_mae": err}
            print(f"Chronos-2 ok on {dev}: {time.time() - t0:.0f} s incl. download, toy sine MAE {err:.3f} (should be < 0.2)")
        except Exception as e:                           # noqa: BLE001
            rep["chronos"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            problems.append("Chronos-2")
            print(f"Chronos-2 FAILED: {type(e).__name__}: {e}")

    (out / "setup_check.json").write_text(json.dumps(rep, indent=2, default=str))
    if problems:
        RU.summary("problems: " + ", ".join(problems))
        raise RuntimeError("setup check found problems: " + ", ".join(problems))
    RU.summary(f"all ok; use --device {rep.get('recommended_device', 'cpu')}"
               + (f"; TFT mFRR up 4h ~{rep['estimate_hours_mfrr_up_4h']['tft']:.1f} h" if "estimate_hours_mfrr_up_4h" in rep else ""))


if __name__ == "__main__":
    RU.run(main, "Tier 2 setup check")
