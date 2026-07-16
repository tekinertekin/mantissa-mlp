"""Speed + accuracy + AMS + peak-RSS benchmark vs torch, tensorflow and
scikit-learn's MLPClassifier.

Protocol (fixed in bench/protocol.py; do not tune per contender):
  - Architectures: the pinned zoo shape per dataset (bench.protocol.MODELS)
    re-expressed layer-for-layer in each framework (bench/contenders.py;
    parameter counts asserted equal for the softmax family, sklearn's binary
    head documented). scikit-learn IS a full contender here — MLPClassifier
    is a real MLP (solver='sgd', constant lr, momentum 0), unlike in the cnn
    repo where it could not express a convolution.
  - Datasets: the six of bench.protocol.SUBSETS, seeded stratified subsets
    (4000/2000 for higgsml + covertype, 2000/1000 for dimuon/mnist_flat/
    wine_quality, 1000/300 for banknote). Features standardized on train
    statistics only (missing=-999.0 for higgsml).
  - Training: EPOCHS epochs, batch BATCH_SIZE, plain SGD lr=LR, softmax
    cross-entropy, seed SEED, CPU only. A fresh model per repeat.
  - Repeats: REPEATS, INTERLEAVED round-robin (A,B,C,D,E x R) so thermal and
    background drift hit every contender equally; medians reported, raw
    samples kept. time.perf_counter(); fit() wall time only (data prepped and
    framework imported beforehand). One untimed WARMUP_N-sample fit per
    contender first — first-call runtime setup (TF graph machinery, torch
    dispatch caches, our dylib load) is a one-time JIT-like cost, excluded the
    same way imports are.
  - Metrics on the held-out test subset with the model the benchmark actually
    trained (re-seeded per repeat; the last is scored):
      test_acc  mean class accuracy
      ams       higgsml only — tasks.ams at a fixed TOP-15%-by-P(signal)
                selection, the challenge's typical operating point (documented
                below), with the challenge's renormalized event weights; the
                select-everything AMS is recorded as the baseline it beats.
  - Batch predict over the test subset: median of PREDICT_CALLS calls,
    interleaved.
  - PEAK RSS: one (contender, dataset) per fresh subprocess; the child imports
    its own framework, loads the cached subset, fits once, reports
    resource.getrusage(RUSAGE_SELF).ru_maxrss (BYTES on macOS, KiB on Linux —
    normalized). Import cost deliberately included: users pay it. The subset
    is loaded from a small cached .npz rather than re-parsed from the full
    CSV, so RSS measures the framework's footprint, not np.loadtxt of the
    818k-row higgsml file (a 386 MB transient that would swamp every column).

AMS operating point: the metric rewards a threshold, so one must be fixed and
stated. We select the top 15% of test events by predicted signal probability
— the region the HiggsML challenge's own operating points cluster in — and
apply it identically to every contender. argmax selection was also measured
during development; the fixed 15% cut is reported because it is comparable
across contenders whose probability calibration differs.

Machine lock: every TIMED run holds /tmp/mantissa-bench.lock (mkdir spin-wait,
ownership-guarded release) so a sibling benchmark on the same machine never
overlaps a timed region. Data prep and JSON writing happen outside the lock.

Output: bench/results/speed.json
  {"env": {...}, "protocol": {...}, "params": {"<dataset>": {"<contender>": n}},
   "fit_s":      {"<dataset>": {"<contender>": {"median":, "samples": [...]}}},
   "predict_ms": {...same nesting...},
   "test_acc":   {"<dataset>": {"<contender>": float}},
   "final_loss": {"<dataset>": {"<contender>": last-epoch training loss}},
   "ams":        {"higgsml": {"<contender>": {"value":, "select_all_ams":,
                              "select_frac":}}},
   "peak_rss_mb": {"<dataset>": {"<contender>": MB}}}

Run from the repo root:  python -m bench.speed
(the RSS worker re-invokes:  python -m bench.speed --worker <contender> <dataset>)
"""
from __future__ import annotations

import atexit
import json
import os
import platform

# Keep TensorFlow's C++ banner out of benchmark output (set before any TF
# import anywhere in the process).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

# numpy 2.x on Apple Accelerate emits spurious FPE RuntimeWarnings from the
# BLAS matmul kernel even on finite inputs (contender weights stay bounded;
# documented across the family benchmarks). They fire from sklearn's and the
# vanilla-numpy backend's matmuls.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

from mantissa_mlp import datasets, tasks

from .contenders import check_parity, contenders
from .protocol import (BATCH_SIZE, EPOCHS, LR, MODELS, REPEATS, SEED, SUBSETS)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "bench" / "results" / "speed.json"
SUBSET_CACHE = REPO_ROOT / "bench" / "results" / "_subsets"

# Harness constants (not protocol: they do not touch training or the metrics).
PREDICT_CALLS = 20      # batch-predict timing repeats
WARMUP_N = 64           # samples for the one untimed warm-up fit
AMS_SELECT_FRAC = 0.15  # top fraction of test events selected as signal

DATASETS = tuple(SUBSETS)   # protocol order: higgsml first (the centerpiece)

LOCK_DIR = Path("/tmp/mantissa-bench.lock")


# --- machine lock ------------------------------------------------------------

def _acquire_lock() -> None:
    """Spin on mkdir until we own /tmp/mantissa-bench.lock, then arm an
    ownership-guarded release (atexit + SIGINT/SIGTERM). mkdir is atomic, so
    exactly one bench process on the machine holds a timed region at a time."""
    waited = False
    while True:
        try:
            LOCK_DIR.mkdir()
            break
        except FileExistsError:
            if not waited:
                print(f"waiting for {LOCK_DIR} (a sibling benchmark holds "
                      f"it) ...", flush=True)
                waited = True
            time.sleep(5)
    (LOCK_DIR / "owner").write_text(f"{os.getpid()} mantissa-mlp bench\n")
    atexit.register(_release_lock)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_a: sys.exit(1))   # -> runs atexit release


def _release_lock() -> None:
    try:
        owner = (LOCK_DIR / "owner").read_text().split()[0]
    except (FileNotFoundError, IndexError):
        return
    if owner != str(os.getpid()):           # ownership guard: never rmdir a
        return                              # lock we did not create
    for op in ((LOCK_DIR / "owner").unlink, LOCK_DIR.rmdir):
        try:
            op()
        except OSError:
            pass


# --- data prep (outside the lock) --------------------------------------------

def _feasible(y, n) -> int:
    """The largest equal-stratified subset size <= n that y's rarest class can
    supply (the same largest-remainder rule datasets.subset uses). Equals n
    whenever the request fits — only banknote's minority class forces it down
    (see _prepare)."""
    _classes, counts = np.unique(y, return_counts=True)
    kmin = int(counts.min())
    base, extra = divmod(int(n), len(counts))
    ask = base + (1 if extra else 0)          # largest per-class quota
    return int(n) if ask <= kmin else kmin * len(counts)


def _prepare(dataset: str) -> dict:
    """Load the seeded stratified subset, standardize on train statistics
    (missing=-999.0 for higgsml), cache it to a small .npz the RSS worker can
    reload, and return the in-memory arrays + the ACTUAL subset sizes used.

    banknote deviation: the frozen protocol asks 1000/300, but banknote's
    minority class (610 rows total) cannot supply 500 train + 150 test per
    class. protocol.py is frozen and the loader is shared, so rather than
    tune a constant we clamp to the largest feasible equal-stratified budget
    (seed unchanged) and record the actual sizes here and in the README."""
    n_tr_req, n_te_req = SUBSETS[dataset]
    if dataset == "higgsml":
        n_tr, n_te = n_tr_req, n_te_req       # signal/background both ample
        Xtr, ytr, wtr, Xte, yte, wte = datasets.subset(
            dataset, n_tr, n_te, SEED, weights=True)
        Xtr, Xte = tasks.standardize(Xtr, Xte, missing=-999.0)
    else:
        try:
            n_tr, n_te = n_tr_req, n_te_req
            Xtr, ytr, Xte, yte = datasets.subset(dataset, n_tr, n_te, SEED)
        except ValueError:
            # A minority class cannot supply the requested budget (banknote).
            # Load once to size the largest feasible equal-stratified subset.
            _Xtr, ytr_full, _Xte, yte_full = datasets.load(dataset)
            n_tr = _feasible(ytr_full, n_tr_req)
            n_te = _feasible(yte_full, n_te_req)
            Xtr, ytr, Xte, yte = datasets.subset(dataset, n_tr, n_te, SEED)
        Xtr, Xte = tasks.standardize(Xtr, Xte)
        wte = None
    classes = int(max(int(ytr.max()), int(yte.max())) + 1)
    d = int(Xtr.shape[1])
    SUBSET_CACHE.mkdir(parents=True, exist_ok=True)
    cache = SUBSET_CACHE / f"{dataset}.npz"
    arrays = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    if wte is not None:
        arrays["wte"] = wte
    np.savez(cache, **arrays)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, wte=wte, d=d,
                classes=classes, n_train=int(len(ytr)), n_test=int(len(yte)),
                requested=(n_tr_req, n_te_req))


def _load_cached(dataset: str) -> dict:
    """The RSS worker's counterpart to _prepare: the cached standardized
    subset only (no full-CSV parse)."""
    z = np.load(SUBSET_CACHE / f"{dataset}.npz")
    d = int(z["Xtr"].shape[1])
    classes = int(max(int(z["ytr"].max()), int(z["yte"].max())) + 1)
    return dict(Xtr=z["Xtr"], ytr=z["ytr"], Xte=z["Xte"], yte=z["yte"],
                wte=z["wte"] if "wte" in z.files else None, d=d, classes=classes)


# --- metrics -----------------------------------------------------------------

def _ams_row(proba, yte, wte):
    """AMS at the fixed top-AMS_SELECT_FRAC-by-P(signal) operating point, plus
    the select-everything baseline it must beat."""
    score = np.asarray(proba)[:, 1]                 # P(class 1 = signal)
    k = max(1, int(round(AMS_SELECT_FRAC * len(score))))
    sel = np.argsort(score)[::-1][:k]
    y_pred = np.zeros(len(score), dtype=np.int32)
    y_pred[sel] = 1
    return {"value": round(tasks.ams(yte, y_pred, wte), 4),
            "select_all_ams": round(
                tasks.ams(yte, np.ones_like(yte), wte), 4),
            "select_frac": AMS_SELECT_FRAC}


# --- timing (inside the lock) ------------------------------------------------

def _time_dataset(dataset, data, reg):
    """Interleaved timing + metrics for one dataset. Returns
    (fit_s, predict_ms, test_acc, final_loss, ams) keyed by contender."""
    d, classes = data["d"], data["classes"]
    yte = data["yte"]
    native = {}
    for name, _factory, px, py in reg:
        native[name] = (px(data["Xtr"]), py(data["ytr"]), px(data["Xte"]))

    # One untimed warm-up fit per contender (WARMUP_N samples) — see WARMUP_N.
    for name, factory, _px, _py in reg:
        Xn, yn, _ = native[name]
        factory(dataset, d, classes).fit(Xn[:WARMUP_N], yn[:WARMUP_N])

    # FIT: outer loop repeats, inner loop contenders -> true round-robin.
    # Fresh estimator per repeat (fresh weights); construction is untimed.
    fit_samples = {name: [] for name, *_ in reg}
    fitted = {}
    for _ in range(REPEATS):
        for name, factory, _px, _py in reg:
            Xn, yn, _ = native[name]
            est = factory(dataset, d, classes)
            t0 = time.perf_counter()
            est.fit(Xn, yn)
            fit_samples[name].append(time.perf_counter() - t0)
            fitted[name] = est

    # PREDICT: batch predict over the test subset, round-robin.
    pred_samples = {name: [] for name, *_ in reg}
    for _ in range(PREDICT_CALLS):
        for name, *_rest in reg:
            Xtn = native[name][2]
            t0 = time.perf_counter()
            fitted[name].predict(Xtn)
            pred_samples[name].append((time.perf_counter() - t0) * 1000.0)

    # METRICS from the models the benchmark actually trained.
    test_acc, final_loss, ams = {}, {}, {}
    for name, *_rest in reg:
        Xtn = native[name][2]
        test_acc[name] = round(
            float(np.mean(fitted[name].predict(Xtn) == yte)), 4)
        final_loss[name] = round(float(fitted[name].final_loss_), 6)
        if dataset == "higgsml":
            ams[name] = _ams_row(fitted[name].predict_proba(Xtn), yte,
                                 data["wte"])

    fit_s = {n: {"median": median(s), "samples": s}
             for n, s in fit_samples.items()}
    predict_ms = {n: {"median": median(s), "samples": s}
                  for n, s in pred_samples.items()}
    return fit_s, predict_ms, test_acc, final_loss, ams


# --- RSS worker --------------------------------------------------------------

def _rss_mb() -> float:
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss: bytes on macOS, KiB on Linux.
    if sys.platform == "darwin":
        return maxrss / (1024.0 * 1024.0)
    return maxrss / 1024.0


def _run_worker(contender: str, dataset: str) -> int:
    """Fresh subprocess: import the contender's framework, load the cached
    subset, fit once under the full protocol, print peak RSS in MB. Import
    cost is included on purpose — it is what a user pays."""
    spec = {name: (factory, px, py)
            for name, factory, px, py in contenders()}.get(contender)
    if spec is None:
        print(f"unknown contender {contender!r}", file=sys.stderr)
        return 2
    factory, prep_X, prep_y = spec
    data = _load_cached(dataset)
    factory(dataset, data["d"], data["classes"]).fit(
        prep_X(data["Xtr"]), prep_y(data["ytr"]))
    print(f"{_rss_mb():.4f}")
    return 0


def _measure_rss(contender: str, dataset: str) -> float:
    proc = subprocess.run(
        [sys.executable, "-m", "bench.speed", "--worker", contender, dataset],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"RSS worker failed for {contender}/{dataset}:\n{proc.stderr}")
    return float(proc.stdout.strip().splitlines()[-1])


# --- environment -------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _env_block() -> dict:
    """Versions and thread settings — thread knobs are left at each
    framework's default and RECORDED, not equalized."""
    env = {
        "cpu": _cpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mantissa_threads": os.environ.get("MANTISSA_THREADS",
                                           f"default({os.cpu_count()})"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    try:
        from mantissa_cnn import MANTISSA_MIN_VERSION
        env["mantissa"] = f">={MANTISSA_MIN_VERSION} (f32 dense primitives)"
    except Exception:
        env["mantissa"] = "unknown"
    import torch
    env["torch"] = torch.__version__
    env["torch_threads"] = torch.get_num_threads()
    import tensorflow as tf
    import keras
    env["tensorflow"] = tf.__version__
    env["keras"] = keras.__version__
    env["tf_inter_op_threads"] = tf.config.threading.get_inter_op_parallelism_threads()
    env["tf_intra_op_threads"] = tf.config.threading.get_intra_op_parallelism_threads()
    env["tf_threads_note"] = "0 = TensorFlow default (runtime-chosen)"
    import sklearn
    env["sklearn"] = sklearn.__version__
    return env


# --- entrypoint --------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--worker":
        return _run_worker(argv[1], argv[2])

    reg = contenders()
    names = [n for n, *_ in reg]
    print(f"contenders: {', '.join(names)}")

    # PREP + parity, all outside the machine lock.
    print("structural parity ...", flush=True)
    params = check_parity(verbose=True)
    print("preparing subsets (load + standardize + cache) ...", flush=True)
    prepared = {ds: _prepare(ds) for ds in DATASETS}

    fit_s, predict_ms, test_acc, final_loss = {}, {}, {}, {}
    ams, peak_rss_mb = {}, {}

    print(f"acquiring {LOCK_DIR} for the timed region ...", flush=True)
    _acquire_lock()
    t_start = time.perf_counter()
    try:
        for dataset in DATASETS:
            print(f"\n[{dataset}] timing (R={REPEATS}, interleaved) ...",
                  flush=True)
            f, p, acc, fl, am = _time_dataset(dataset, prepared[dataset], reg)
            fit_s[dataset], predict_ms[dataset] = f, p
            test_acc[dataset], final_loss[dataset] = acc, fl
            if am:
                ams[dataset] = am
            for name in names:
                extra = (f"  AMS {am[name]['value']:5.2f}"
                         if am else "")
                print(f"  {name:14s} fit {f[name]['median']:8.3f} s   "
                      f"predict {p[name]['median']:8.2f} ms   "
                      f"acc {acc[name]:.4f}{extra}", flush=True)
            print(f"[{dataset}] peak RSS (fresh subprocess each) ...",
                  flush=True)
            peak_rss_mb[dataset] = {}
            for name in names:
                mb = _measure_rss(name, dataset)
                peak_rss_mb[dataset][name] = round(mb, 4)
                print(f"  {name:14s} {mb:8.1f} MB", flush=True)
        wall = time.perf_counter() - t_start
    finally:
        _release_lock()

    out = {
        "env": _env_block(),
        "protocol": {"datasets": list(DATASETS), "contenders": names,
                     "models": {ds: MODELS[ds] for ds in DATASETS},
                     "subsets_requested": {ds: list(SUBSETS[ds])
                                           for ds in DATASETS},
                     "subsets": {ds: [prepared[ds]["n_train"],
                                      prepared[ds]["n_test"]]
                                 for ds in DATASETS},
                     "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                     "seed": SEED, "repeats": REPEATS,
                     "predict_calls": PREDICT_CALLS, "warmup_n": WARMUP_N,
                     "ams_select_frac": AMS_SELECT_FRAC},
        "params": params,
        "fit_s": fit_s,
        "predict_ms": predict_ms,
        "test_acc": test_acc,
        "final_loss": final_loss,
        "ams": ams,
        "peak_rss_mb": peak_rss_mb,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO_ROOT)} "
          f"({wall:.0f}s timed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
