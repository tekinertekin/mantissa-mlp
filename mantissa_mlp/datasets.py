"""Six tabular classification datasets, loaded as (n, d) float32.

Nothing downloads implicitly. ``load(name)`` reads files from the data
directory; if any are missing it raises FileNotFoundError with the exact fix
command. The only code that touches the network is the explicit CLI::

    python -m mantissa_mlp.datasets download <name|all>
    python -m mantissa_mlp.datasets list

Data directory: ``./data/<name>/`` relative to the current working directory
(so ``mlp/data/`` when run from the repo root), or the ``MANTISSA_MLP_DATA``
environment variable. The directory is gitignored — datasets are never
committed. Two family exceptions, both documented below: ``mnist_flat``
reads mantissa-cnn's mnist files (one download for the whole family), and
``banknote`` also looks in a sibling mantissa-perceptron checkout's
``data/`` before asking you to download.

``load(name)`` -> ``(X_train, y_train, X_test, y_test)``: X float32 (n, d)
**unstandardized** (that is a train-statistics decision — see
:func:`mantissa_mlp.tasks.standardize`), y int32 class ids. higgsml
additionally carries physics event weights: ``load("higgsml",
weights=True)`` -> ``(X_train, y_train, w_train, X_test, y_test, w_test)``.

| name         | train/test      | d   | classes | source |
|--------------|-----------------|-----|---------|--------|
| higgsml      | 250000 / 550000 | 30  | 2 | ATLAS Higgs ML challenge 2014, CERN Open Data record 328, DOI 10.7483/OPENDATA.ATLAS.ZBP2.M5T8 (CC0) |
| dimuon       | 75/25 of 26911  | 16  | 3 | CMS dimuon events (McCauley, 2017), CERN Open Data record 545 (CC0) |
| mnist_flat   | 60000 / 10000   | 784 | 10 | LeCun et al. (1998), via mantissa-cnn's loader, flattened |
| covertype    | 75/25 of 581012 | 54  | 7 | UCI covtype (Blackard & Dean, 1999) |
| wine_quality | 75/25 of 6497   | 11  | 3 | UCI wine quality, red+white (Cortez et al., 2009) |
| banknote     | 75/25 of 1372   | 4   | 2 | UCI 00267 — the mantissa-perceptron protocol dataset |

Splits: higgsml uses the challenge's own ``KaggleSet`` column (t = the 250k
training set; b + v, the public and private leaderboards, are the 550k test
set; the 18238 unused rows are dropped). mnist_flat keeps the official
train/test files. The other four have no canonical split and get the family
protocol: stratified 75/25 holdout, seed 42 (mantissa-perceptron's split).

All URLs verified fetchable (curl -I, HTTP 200) 2026-07.
"""
from __future__ import annotations

import gzip
import os
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np

__all__ = ["DATASETS", "DIMUON_WINDOWS", "data_dir", "download",
           "download_command", "load", "subset"]

_DATA_ENV = "MANTISSA_MLP_DATA"

# mnist_flat reads mantissa-cnn's data directory; importing this module
# points MANTISSA_CNN_DATA at the data/ next to the installed mantissa_cnn
# package (the cnn checkout's data/ in the dev layout, where mnist already
# lives) unless the caller has set it — mantissa-autoencoder's pattern.
_CNN_DATA_ENV = "MANTISSA_CNN_DATA"


def _point_cnn_at_sibling_data() -> None:
    if _CNN_DATA_ENV in os.environ:
        return                     # the caller's choice stands
    import mantissa_cnn.datasets as _cnn_ds
    candidate = Path(_cnn_ds.__file__).resolve().parents[1] / "data"
    if candidate.is_dir():
        os.environ[_CNN_DATA_ENV] = str(candidate)


_point_cnn_at_sibling_data()


class _Spec(NamedTuple):
    files: Tuple[str, ...]        # filenames under data/<name>/
    urls: Tuple[str, ...]         # one per file ("" = not ours to download)
    magic: Tuple[bytes, ...]      # expected first bytes, one per file
    note: str


DATASETS = {
    "higgsml": _Spec(
        ("atlas-higgs-challenge-2014-v2.csv.gz",),
        ("http://opendata.cern.ch/record/328/files/"
         "atlas-higgs-challenge-2014-v2.csv.gz",),
        (b"\x1f\x8b",),
        "ATLAS Higgs->tautau signal vs background, 818238 simulated events "
        "(CERN Open Data 328, CC0)"),
    "dimuon": _Spec(
        ("Dimuon_DoubleMu.csv",),
        ("http://opendata.cern.ch/record/545/files/Dimuon_DoubleMu.csv",),
        (b"Run,Event",),
        "CMS dimuon resonances J/psi vs Upsilon vs Z, labels from invariant-"
        "mass windows (CERN Open Data 545, CC0)"),
    "mnist_flat": _Spec(
        (), (), (),
        "MNIST digits flattened to (n, 784) — mantissa-cnn's files, "
        "downloaded once for the whole family"),
    "covertype": _Spec(
        ("covtype.data.gz",),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
         "covtype/covtype.data.gz",),
        (b"\x1f\x8b",),
        "forest cover type from cartographic features, 581012 x 54, "
        "7 classes (UCI)"),
    "wine_quality": _Spec(
        ("winequality-red.csv", "winequality-white.csv"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
         "wine-quality/winequality-red.csv",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/"
         "wine-quality/winequality-white.csv"),
        (b'"fixed acidity"', b'"fixed acidity"'),
        "red+white vinho verde, quality binned low/mid/high (UCI, "
        "Cortez et al. 2009)"),
    "banknote": _Spec(
        ("data_banknote_authentication.txt",),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/"
         "data_banknote_authentication.txt",),
        (b"",),
        "genuine vs forged banknotes — the mantissa-perceptron protocol "
        "dataset, kept as the family-continuity sanity row"),
}

# Dimuon invariant-mass windows (GeV) -> class id. The J/psi window stops
# short of the psi(2S) at 3.686; the Upsilon window spans the 1S/2S/3S
# triplet (9.46/10.02/10.36); the Z window is the conventional 60-120 band
# around 91.19. Events outside every window are dropped.
DIMUON_WINDOWS = (
    ("jpsi", 2.8, 3.4),        # class 0
    ("upsilon", 9.0, 11.0),    # class 1
    ("z", 60.0, 120.0),        # class 2
)


def data_dir() -> Path:
    return Path(os.environ.get(_DATA_ENV, "data"))


def download_command(name: str) -> str:
    return f"python -m mantissa_mlp.datasets download {name}"


def _paths(name: str):
    d = data_dir() / name
    return [d / f for f in DATASETS[name].files]


def _require_files(name: str):
    paths = _paths(name)
    if name == "banknote" and not all(p.is_file() for p in paths):
        # Family continuity: the sibling mantissa-perceptron checkout keeps
        # this exact file in its data/ — reuse it rather than re-download.
        sibling = (Path(__file__).resolve().parents[2] / "perceptron"
                   / "data" / DATASETS[name].files[0])
        if sibling.is_file():
            return [sibling]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            f"dataset {name!r} not downloaded — run: {download_command(name)}")
    return paths


# -- per-dataset parsers --------------------------------------------------------

def _stratified_split(X, y, w=None, test_size: float = 0.25, seed: int = 42):
    """The family protocol split: seeded stratified holdout (perceptron's)."""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        test_mask[idx[:max(1, int(round(test_size * len(idx))))]] = True
    parts = (X[~test_mask], y[~test_mask], X[test_mask], y[test_mask])
    if w is None:
        return parts
    return parts + (w[~test_mask], w[test_mask])


def _load_higgsml(path: Path):
    """-> (Xtr, ytr, wtr, Xte, yte, wte). 30 float features (every DER_* and
    PRI_* column, located by header name), label s/b -> 1/0.

    The -999.0 entries are *kept*: they are the challenge's sentinel for
    "undefined for this event topology" (e.g. jet variables when the event
    has no jets — appendix B of Adam-Bourdarios et al. 2015), and handling
    them is a train-statistics decision that belongs next to
    standardization: ``tasks.standardize(Xtr, Xte, missing=-999.0)``.

    Split: the KaggleSet column — 't' (250k) is the train set, 'b' + 'v'
    (the 100k public + 450k private leaderboard sets, 550k) are the test
    set, 'u' (18238 unused) is dropped. Weights are the v2 file's 'Weight'
    column (normalized so the full 818238 events reproduce the expected
    signal/background counts), then renormalized per split so each side's
    per-class sums equal the full-set totals — the same convention as the
    challenge's own KaggleWeight, extended to this split, so AMS values
    are comparable to the leaderboard's. EventId and the Kaggle
    bookkeeping columns are dropped.
    """
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split(",")
        cols = {c: i for i, c in enumerate(header)}
        feat = [c for c in header if c.startswith(("DER_", "PRI_"))]
        for c in ("Weight", "Label", "KaggleSet"):
            if c not in cols:
                raise ValueError(f"{path}: no {c!r} column — not the v2 "
                                 f"challenge file")
        if len(feat) != 30:
            raise ValueError(f"{path}: expected 30 DER_*/PRI_* feature "
                             f"columns, found {len(feat)}")
        conv = {cols["Label"]: lambda s: 1.0 if s == "s" else 0.0,
                cols["KaggleSet"]: lambda s: float("tbvu".index(s))}
        # float32 rows halve the parse footprint; only the weights, whose
        # per-class SUMS drive the AMS renormalization, go back to float64.
        raw = np.loadtxt(f, delimiter=",", converters=conv, dtype=np.float32)
    X = np.ascontiguousarray(raw[:, [cols[c] for c in feat]])
    y = raw[:, cols["Label"]].astype(np.int32)
    w = raw[:, cols["Weight"]].astype(np.float64)
    kset = raw[:, cols["KaggleSet"]].astype(np.int32)

    s_tot, b_tot = w[y == 1].sum(), w[y == 0].sum()   # full-set totals
    out = []
    for mask in (kset == 0, (kset == 1) | (kset == 2)):   # t | b+v
        Xs, ys, ws = X[mask], y[mask], w[mask].copy()
        ws[ys == 1] *= s_tot / ws[ys == 1].sum()
        ws[ys == 0] *= b_tot / ws[ys == 0].sum()
        out.append((np.ascontiguousarray(Xs), ys, ws))
    (Xtr, ytr, wtr), (Xte, yte, wte) = out
    return Xtr, ytr, wtr, Xte, yte, wte


def _load_dimuon(path: Path):
    """-> stratified family split of the mass-window events.

    Features are the 16 muon kinematics columns — E, px, py, pz, pt, eta,
    phi, charge Q for each muon — **not** the invariant mass M, which is
    where the labels come from: J/psi / Upsilon / Z mass windows (see
    DIMUON_WINDOWS). The reconstruction-category columns (type1/type2, G/T
    strings) and the Run/Event bookkeeping are dropped.

    Honesty note, also in the README: the labels are *constructed* from M,
    and M is itself a function of the two 4-vectors (M^2 = (E1+E2)^2 -
    |p1+p2|^2), so the task is learnable by construction — what it measures
    is whether a small MLP can approximate that nonlinear function well
    enough to separate the three resonances, not any statistically hidden
    truth. That is a feature for a teaching dataset, and it is stated
    rather than implied.
    """
    with open(path) as f:
        header = f.readline().strip().split(",")
        cols = {c: i for i, c in enumerate(header)}
        feat = [p + s for s in ("1", "2")
                for p in ("E", "px", "py", "pz", "pt", "eta", "phi", "Q")]
        missing = [c for c in feat + ["M"] if c not in cols]
        if missing:
            raise ValueError(f"{path}: missing columns {missing} — not the "
                             f"record-545 dimuon file")
        raw = np.loadtxt(f, delimiter=",",
                         usecols=[cols[c] for c in feat + ["M"]],
                         dtype=np.float64)
    M = raw[:, -1]
    y = np.full(len(M), -1, dtype=np.int32)
    for cid, (_, lo, hi) in enumerate(DIMUON_WINDOWS):
        y[(M >= lo) & (M <= hi)] = cid
    keep = y >= 0
    X = np.ascontiguousarray(raw[keep, :-1], dtype=np.float32)
    return _stratified_split(X, y[keep])


def _load_mnist_flat():
    """mantissa-cnn's mnist, flattened NCHW (n,1,28,28) -> (n, 784)."""
    from mantissa_cnn import datasets as cnn_datasets
    try:
        Xtr, ytr, Xte, yte = cnn_datasets.load("mnist")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"dataset 'mnist_flat' not downloaded — run: "
            f"{download_command('mnist_flat')}") from None
    return (Xtr.reshape(len(Xtr), -1), ytr.astype(np.int32),
            Xte.reshape(len(Xte), -1), yte.astype(np.int32))


def _load_covertype(path: Path):
    """UCI covtype.data.gz: 54 features, cover types 1..7 -> class ids 0..6."""
    with gzip.open(path, "rb") as f:
        raw = np.loadtxt(f, delimiter=",", dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 55:
        raise ValueError(f"{path}: expected 55 columns (54 features + cover "
                         f"type), got shape {raw.shape}")
    X = np.ascontiguousarray(raw[:, :54])
    y = raw[:, 54].astype(np.int32) - 1
    return _stratified_split(X, y)


def _load_wine(red: Path, white: Path):
    """UCI wine quality, red + white merged: 11 features; the 0-10 expert
    score binned to 3 classes — low (<= 5), mid (= 6), high (>= 7). The
    binning is documented rather than canonical: 6 is the mode of both
    files, and the <=5 / 6 / >=7 cut is the roughly balanced split used
    throughout the literature that treats this as classification. The
    red/white origin is NOT appended as a feature — the 11 physicochemical
    columns are the advertised feature set (Cortez, Cerdeira, Almeida,
    Matos & Reis, 2009, "Modeling wine preferences by data mining from
    physicochemical properties", *Decision Support Systems* 47(4))."""
    parts = []
    for path in (red, white):
        with open(path) as f:
            header = f.readline()
            if "fixed acidity" not in header:
                raise ValueError(f"{path}: unexpected header {header[:40]!r}")
            parts.append(np.loadtxt(f, delimiter=";", dtype=np.float32))
    raw = np.concatenate(parts)
    if raw.shape[1] != 12:
        raise ValueError(f"expected 12 columns (11 features + quality), "
                         f"got {raw.shape[1]}")
    X = np.ascontiguousarray(raw[:, :11])
    q = raw[:, 11].astype(np.int32)
    y = np.where(q <= 5, 0, np.where(q == 6, 1, 2)).astype(np.int32)
    return _stratified_split(X, y)


def _load_banknote(path: Path):
    """UCI 00267: 4 wavelet features, genuine (0) vs forged (1) — the
    perceptron repo's protocol dataset, same parse, family split."""
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
    X = np.ascontiguousarray(raw[:, :4])
    y = raw[:, 4].astype(np.int32)
    return _stratified_split(X, y)


# -- public API ---------------------------------------------------------------

def load(name: str, weights: bool = False):
    """Load dataset ``name`` -> (X_train, y_train, X_test, y_test), float32
    features (unstandardized — see tasks.standardize), int32 class ids.

    ``weights=True`` (higgsml only) -> (X_train, y_train, w_train, X_test,
    y_test, w_test) with float64 physics event weights renormalized per
    split for the AMS metric. Never downloads: raises FileNotFoundError
    with the exact fix command if files are missing.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    if weights and name != "higgsml":
        raise ValueError(f"only 'higgsml' carries physics event weights, "
                         f"got weights=True for {name!r}")
    if name == "mnist_flat":
        return _load_mnist_flat()
    paths = _require_files(name)
    if name == "higgsml":
        Xtr, ytr, wtr, Xte, yte, wte = _load_higgsml(paths[0])
        if weights:
            return Xtr, ytr, wtr, Xte, yte, wte
        return Xtr, ytr, Xte, yte
    if name == "dimuon":
        return _load_dimuon(paths[0])
    if name == "covertype":
        return _load_covertype(paths[0])
    if name == "wine_quality":
        return _load_wine(paths[0], paths[1])
    return _load_banknote(paths[0])


def subset(name: str, n_train: int, n_test: int, seed: int = 0,
           weights: bool = False):
    """Seeded stratified subset -> same tuple shape as load(name, weights).

    Per-class quotas are as equal as the class counts allow (largest-
    remainder split of n over the classes present — mantissa-cnn's rule).
    The benchmark protocol uses 4000/2000 for higgsml and covertype,
    2000/1000 for the rest. higgsml subset weights are renormalized again
    so each side's per-class sums stay equal to the full-set totals —
    the challenge's own convention for its subsets, keeping AMS values
    comparable across sizes.
    """
    if weights:
        Xtr, ytr, wtr, Xte, yte, wte = load(name, weights=True)
    else:
        Xtr, ytr, Xte, yte = load(name)
    itr = _stratified_indices(ytr, n_train, np.random.default_rng(seed))
    ite = _stratified_indices(yte, n_test, np.random.default_rng(seed + 1))
    if not weights:
        return Xtr[itr], ytr[itr], Xte[ite], yte[ite]

    def rescale(w_all, y_all, idx):
        w = w_all[idx].copy()
        y_sub = y_all[idx]
        for c in (0, 1):
            w[y_sub == c] *= w_all[y_all == c].sum() / w[y_sub == c].sum()
        return w
    return (Xtr[itr], ytr[itr], rescale(wtr, ytr, itr),
            Xte[ite], yte[ite], rescale(wte, yte, ite))


def _stratified_indices(y, n, rng):
    classes = np.unique(y)
    base, extra = divmod(int(n), len(classes))
    picks = []
    for i, c in enumerate(classes):
        idx = np.flatnonzero(y == c)
        take = base + (1 if i < extra else 0)
        if take > len(idx):
            raise ValueError(f"class {c} has only {len(idx)} samples, need {take}")
        picks.append(rng.permutation(idx)[:take])
    return rng.permutation(np.concatenate(picks))


# -- explicit downloader (the only networking code) ----------------------------

def download(name: str) -> None:
    """Fetch every file of dataset ``name``, verified and atomic.

    The payload is checked before it can reach the load path: length against
    the server's Content-Length and each file's expected first bytes (gzip
    magic or the CSV header) — a truncated body or an HTML error page
    raises OSError instead of landing on disk. The verified body is written
    to a ``.part`` file and renamed into place, so ``load()`` never sees a
    partial download. ``mnist_flat`` delegates to mantissa-cnn's downloader
    (same files, one copy for the whole family).
    """
    if name == "mnist_flat":
        from mantissa_cnn import datasets as cnn_datasets
        cnn_datasets.download("mnist")
        return
    spec = DATASETS[name]
    d = data_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    for fname, url, magic in zip(spec.files, spec.urls, spec.magic):
        path = d / fname
        if path.is_file():
            print(f"{name}: {fname} already present")
            continue
        print(f"{name}: {url}\n  -> {path}")
        with urllib.request.urlopen(url, timeout=120) as r:
            length = r.headers.get("Content-Length")
            body = r.read()
        if length is not None and len(body) != int(length):
            raise OSError(f"{url}: truncated — got {len(body):,} of "
                          f"{int(length):,} announced bytes")
        if magic and not body.startswith(magic):
            raise OSError(f"{url}: unexpected content (starts {body[:12]!r}, "
                          f"want {magic!r}) — an error page or proxy "
                          f"response, not the dataset")
        if body.startswith(b"<"):
            raise OSError(f"{url}: got markup, not data — an error page")
        tmp = path.with_name(fname + ".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        print(f"  done ({len(body):,} bytes)")


def _main(argv) -> int:
    if len(argv) == 1 and argv[0] == "list":
        for name, spec in DATASETS.items():
            if name == "mnist_flat":
                try:
                    from mantissa_cnn import datasets as cnn_datasets
                    d = cnn_datasets.data_dir() / "mnist"
                    ok = all((d / f).is_file()
                             for f in cnn_datasets.DATASETS["mnist"].files)
                except Exception:
                    ok = False
            else:
                try:
                    _require_files(name)
                    ok = True
                except FileNotFoundError:
                    ok = False
            print(f"{name:13} {'present' if ok else 'missing':8} {spec.note}")
        return 0
    if len(argv) == 2 and argv[0] == "download":
        names = list(DATASETS) if argv[1] == "all" else [argv[1]]
        failed = []
        for name in names:
            if name not in DATASETS:
                print(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}",
                      file=sys.stderr)
                return 2
            try:
                download(name)
            except Exception as exc:            # keep fetching the rest
                print(f"{name}: FAILED — {exc}", file=sys.stderr)
                failed.append(name)
        if failed:
            print(f"download failed for: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0
    print("usage: python -m mantissa_mlp.datasets download <name|all>\n"
          "       python -m mantissa_mlp.datasets list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
