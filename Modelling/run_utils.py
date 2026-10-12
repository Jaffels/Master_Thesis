"""Run bookkeeping for every modelling script: console log, run metadata, run history and a macOS
notification when the script finishes (or fails).

Usage in a script (already wired into all Modelling/rq*.py):
    import run_utils as RU
    def main():
        ...
        out.mkdir(parents=True, exist_ok=True)
        RU.set_output(out)                      # where run.log / run_meta.json go
        ...
        RU.summary("mFRR up 4h: best = cat_q, +9.1 % vs naive")   # optional, shown in the notification
    if __name__ == "__main__":
        RU.run(main, "RQ2 Tier 1 models")

What a run leaves behind (in the script's output folder):
    run.log         everything the script printed (stdout + stderr), also when it crashed
    run_meta.json   command line, start / end / duration, status, git commit (+ uncommitted changes),
                    Python / package versions, machine, list of files written in this run
and one line per run in Modelling/Output/run_history.csv (all scripts, newest last).

Notifications: macOS only (osascript). Success = sound "Glass", failure / interrupt = "Basso".
Set THESIS_NOTIFY=0 to switch notifications off (e.g. for quick checks).
"""
from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY = HERE / "Output" / "run_history.csv"
PACKAGES = ["numpy", "pandas", "scipy", "pyarrow", "duckdb", "lightgbm", "scikit-learn", "catboost",
            "quantile-forest", "statsmodels", "torch", "matplotlib"]

_state: dict = {"out": None, "summary": [], "start": None}


# ---------------------------------------------------------------- notification
def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def notify_mac(title: str, message: str, subtitle: str = "", sound: str = "Glass") -> None:
    """macOS Notification Center message. Never raises; prints instead on other systems."""
    if os.environ.get("THESIS_NOTIFY", "1") == "0":
        return
    if sys.platform != "darwin":
        print(f"[notify] {title} | {subtitle} | {message}")
        return
    script = f'display notification "{_esc(message)[:220]}" with title "{_esc(title)[:80]}"'
    if subtitle:
        script += f' subtitle "{_esc(subtitle)[:80]}"'
    if sound:
        script += f' sound name "{_esc(sound)}"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ---------------------------------------------------------------- console tee
class _Tee(io.TextIOBase):
    def __init__(self, stream, buf):
        self.stream, self.buf = stream, buf

    def write(self, s):
        self.buf.append(s)
        try:
            return self.stream.write(s)
        except Exception:
            return len(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


# ---------------------------------------------------------------- public helpers
def set_output(path) -> None:
    """Register the script's output folder (run.log and run_meta.json are written there)."""
    _state["out"] = Path(path)


def summary(text: str) -> None:
    """Add a one-line result to the notification (and to run_meta.json)."""
    _state["summary"].append(str(text))


def fmt_duration(sec: float) -> str:
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {s:02d} s" if m else f"{s} s")


def _git() -> dict:
    def g(*args):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    st = g("status", "--porcelain", "--", "Modelling", "Clean")
    return {"commit": g("rev-parse", "--short", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_files_in_Modelling_Clean": [l[3:] for l in st.splitlines()][:50]}


def _versions() -> dict:
    v = {}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except Exception:
            pass
    return v


def _files_written(out: Path, since: float) -> list[dict]:
    rows = []
    for f in sorted(out.rglob("*")):
        try:
            if f.is_file() and f.stat().st_mtime >= since - 1 and f.name not in ("run.log", "run_meta.json"):
                rows.append({"file": str(f.relative_to(out)), "kB": round(f.stat().st_size / 1024, 1)})
        except Exception:
            pass
    return rows[:500]


def _append_history(row: dict) -> None:
    try:
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        new = not HISTORY.exists()
        cols = ["start", "script", "status", "duration_s", "args", "output", "git_commit", "summary"]
        with HISTORY.open("a", encoding="utf-8") as fh:
            if new:
                fh.write(",".join(cols) + "\n")
            fh.write(",".join('"' + str(row.get(c, "")).replace('"', "'") + '"' for c in cols) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- runner
def run(main, label: str | None = None) -> None:
    """Run main() with console capture, metadata, history line and a notification at the end."""
    script = Path(sys.argv[0]).stem
    label = label or script
    t0 = time.time()
    _state.update(start=t0, summary=[], out=None)
    started = datetime.now().isoformat(timespec="seconds")
    buf: list[str] = []
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, buf), _Tee(old_err, buf)
    status, err, code = "ok", "", 0
    print(f"[{started}] {script} {' '.join(sys.argv[1:])}")
    try:
        main()
    except SystemExit as e:                          # argparse --help / errors
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code == 0:
            sys.stdout, sys.stderr = old_out, old_err
            raise
        status, err = "failed", f"SystemExit {e.code}"
    except KeyboardInterrupt:
        status, err, code = "interrupted", "KeyboardInterrupt", 130
        print("\n[interrupted by user]")
    except Exception as e:                           # noqa: BLE001
        status, err, code = "failed", f"{type(e).__name__}: {e}", 1
        traceback.print_exc()
    dur = time.time() - t0
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {script}: {status} after {fmt_duration(dur)}")
    sys.stdout, sys.stderr = old_out, old_err

    out = _state["out"] or (HERE / "Output" / "_runs" / f"{script}_{datetime.now():%Y%m%d_%H%M%S}")
    rel = str(out.relative_to(ROOT)) if str(out).startswith(str(ROOT)) else str(out)
    meta = {"script": script, "label": label, "status": status, "error": err, "argv": sys.argv[1:],
            "start": started, "end": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(dur, 1), "duration": fmt_duration(dur), "summary": _state["summary"],
            "output_dir": rel, "git": _git(), "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.node(), "cpu_count": os.cpu_count(),
            "packages": _versions()}
    try:
        out.mkdir(parents=True, exist_ok=True)
        meta["files_written"] = _files_written(out, t0)
        (out / "run.log").write_text("".join(buf), encoding="utf-8")
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    except Exception as e:                           # noqa: BLE001
        print(f"[run_utils] could not write run.log / run_meta.json: {e}")
    _append_history({"start": started, "script": script, "status": status, "duration_s": round(dur),
                     "args": " ".join(sys.argv[1:]), "output": rel, "git_commit": meta["git"]["commit"],
                     "summary": " | ".join(_state["summary"])})

    if status == "ok":
        msg = " | ".join(_state["summary"]) or f"Output: {rel}"
        notify_mac(f"{label} finished", msg, subtitle=f"{script} - {fmt_duration(dur)}", sound="Glass")
    else:
        notify_mac(f"{label} {status.upper()}", err or status, subtitle=f"{script} - after {fmt_duration(dur)}",
                   sound="Basso")
    if code:
        sys.exit(code)


if __name__ == "__main__":                           # python Modelling/run_utils.py -> test notification
    notify_mac("Thesis notification test", "If you can read this, notifications work.", subtitle="run_utils.py")
    print("test notification sent (check Notification Center; allow notifications for 'Script Editor' if nothing shows)")
