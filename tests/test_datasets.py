"""Dataset loaders on fabricated tiny files — fast, no network. The higgsml
and dimuon parsers are pinned in detail (they are the CERN centerpieces);
real files are touched only when already on disk and tiny (banknote via the
sibling perceptron checkout)."""
import gzip
from pathlib import Path

import numpy as np
import pytest

from mantissa_mlp import datasets as ds
from mantissa_mlp import tasks


@pytest.fixture()
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTISSA_MLP_DATA", str(tmp_path))
    return tmp_path


# -- fabricated higgsml ----------------------------------------------------------

HIGGS_HEADER = ("EventId,"
                + ",".join(f"DER_f{i}" for i in range(13)) + ","
                + ",".join(f"PRI_f{i}" for i in range(17))
                + ",Weight,Label,KaggleSet,KaggleWeight")


def write_higgs(tmp_path, rows, header=HIGGS_HEADER):
    d = tmp_path / "higgsml"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "atlas-higgs-challenge-2014-v2.csv.gz", "wt") as f:
        f.write(header + "\n")
        f.writelines(r + "\n" for r in rows)


def higgs_rows(n=20, sentinel_at=(0, 3)):
    """n rows: every 3rd is signal; first 10 are KaggleSet t, then 5 b, 5 v.
    Row 0 gets -999.0 sentinels in the first two DER features."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        feats = rng.normal(size=30).round(3).astype(object)
        if i == 0:
            for j in sentinel_at:
                feats[j] = -999.0
        lab = "s" if i % 3 == 0 else "b"
        kset = "t" if i < 10 else ("b" if i < 15 else "v")
        rows.append(f"{i}," + ",".join(str(v) for v in feats)
                    + f",{0.5 + 0.1 * i:.4f},{lab},{kset},1.0")
    return rows


def test_higgsml_loader_contract(data_home):
    write_higgs(data_home, higgs_rows())
    Xtr, ytr, wtr, Xte, yte, wte = ds.load("higgsml", weights=True)
    assert Xtr.shape == (10, 30) and Xte.shape == (10, 30)
    assert Xtr.dtype == np.float32 and ytr.dtype == np.int32
    # label map s/b -> 1/0: rows 0,3,6,9 of the t-set are signal
    assert ytr.tolist() == [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    # the -999.0 sentinel is KEPT by the loader (standardize handles it)
    assert Xtr[0, 0] == -999.0 and Xtr[0, 3] == -999.0
    # weights preserved up to the documented per-class renormalization:
    # within a class, relative weights must match the file exactly
    file_w_signal = np.array([0.5, 0.8, 1.1, 1.4])       # rows 0,3,6,9
    ratio = wtr[ytr == 1] / file_w_signal
    assert np.allclose(ratio, ratio[0])
    # per-class sums equal on both sides: the full-set totals
    assert wtr[ytr == 1].sum() == pytest.approx(wte[yte == 1].sum())
    assert wtr[ytr == 0].sum() == pytest.approx(wte[yte == 0].sum())


def test_higgsml_without_weights_is_family_tuple(data_home):
    write_higgs(data_home, higgs_rows())
    out = ds.load("higgsml")
    assert len(out) == 4
    assert out[0].shape == (10, 30)


def test_higgsml_sentinel_pins_standardize_contract(data_home):
    """End-to-end pin of the documented -999 approach on loader output."""
    write_higgs(data_home, higgs_rows())
    Xtr, ytr, Xte, yte = ds.load("higgsml")
    Str, Ste = tasks.standardize(Xtr, Xte, missing=-999.0)
    assert Str[0, 0] == 0.0                       # sentinel -> neutral 0
    assert np.abs(Str).max() < 50                 # no -999 leaked into stats
    col = Xtr[1:, 0]                              # defined entries of col 0
    assert np.allclose(Str[1:, 0], (col - col.mean()) / col.std(),
                       rtol=1e-4, atol=1e-4)


def test_higgsml_rejects_wrong_file(data_home):
    write_higgs(data_home, ["1,2,3"], header="a,b,c")
    with pytest.raises(ValueError, match="not the v2"):
        ds.load("higgsml")
    # right bookkeeping columns but a wrong feature count is also rejected
    write_higgs(data_home, ["1,2,0.1,s,t,1.0"],
                header="EventId,DER_only,Weight,Label,KaggleSet,KaggleWeight")
    with pytest.raises(ValueError, match="expected 30 DER_"):
        ds.load("higgsml")


def test_higgsml_subset_reweights_to_full_totals(data_home):
    write_higgs(data_home, higgs_rows(n=40))
    Xtr, ytr, wtr, Xte, yte, wte = ds.load("higgsml", weights=True)
    sXtr, sytr, swtr, sXte, syte, swte = ds.subset("higgsml", 6, 6, seed=0,
                                                   weights=True)
    assert sXtr.shape == (6, 30) and sXte.shape == (6, 30)
    assert swtr[sytr == 1].sum() == pytest.approx(wtr[ytr == 1].sum())
    assert swtr[sytr == 0].sum() == pytest.approx(wtr[ytr == 0].sum())
    assert swte[syte == 0].sum() == pytest.approx(wte[yte == 0].sum())


# -- fabricated dimuon -------------------------------------------------------------

DIMUON_HEADER = ("Run,Event,type1,E1,px1,py1,pz1,pt1,eta1,phi1,Q1,"
                 "type2,E2,px2,py2,pz2,pt2,eta2,phi2,Q2,M")


def write_dimuon(tmp_path, masses):
    d = tmp_path / "dimuon"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for i, m in enumerate(masses):
        k1 = ",".join(f"{v:.3f}" for v in rng.normal(size=7))
        k2 = ",".join(f"{v:.3f}" for v in rng.normal(size=7))
        rows.append(f"165617,{i},G,{k1},-1,T,{k2},1,{m}")
    (d / "Dimuon_DoubleMu.csv").write_text(
        DIMUON_HEADER + "\n" + "\n".join(rows) + "\n")


def test_dimuon_mass_windows_make_the_labels(data_home):
    # 4 J/psi, 3 Upsilon, 3 Z, and 2 out-of-window events that must vanish
    write_dimuon(data_home, [3.1, 3.0, 2.85, 3.35,      # J/psi window
                             9.5, 10.3, 9.05,           # Upsilon window
                             91.0, 60.5, 119.0,         # Z window
                             5.0, 130.0])               # dropped
    Xtr, ytr, Xte, yte = ds.load("dimuon")
    y = np.concatenate([ytr, yte])
    assert len(y) == 10                                 # 12 - 2 dropped
    assert np.bincount(y, minlength=3).tolist() == [4, 3, 3]
    # 16 kinematics features; the mass column must NOT be among them
    assert Xtr.shape[1] == 16
    all_X = np.concatenate([Xtr, Xte])
    assert not np.isin([3.1, 9.5, 91.0], all_X.round(4)).any()
    # charges survive as features (the +-1 columns)
    assert set(np.unique(all_X[:, 7])) == {-1.0} and \
        set(np.unique(all_X[:, 15])) == {1.0}


def test_dimuon_rejects_wrong_file(data_home):
    d = data_home / "dimuon"
    d.mkdir(parents=True)
    (d / "Dimuon_DoubleMu.csv").write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="record-545"):
        ds.load("dimuon")


# -- the other loaders ---------------------------------------------------------

def test_covertype_fabricated_and_missing_message(data_home):
    with pytest.raises(FileNotFoundError, match=r"dataset 'covertype' not "
                       r"downloaded — run: python -m mantissa_mlp.datasets "
                       r"download covertype"):
        ds.load("covertype")
    d = data_home / "covertype"
    d.mkdir(parents=True)
    rng = np.random.default_rng(2)
    rows = [",".join(str(v) for v in rng.integers(0, 100, size=54).tolist()
                     + [1 + i % 7]) for i in range(28)]
    with gzip.open(d / "covtype.data.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
    Xtr, ytr, Xte, yte = ds.load("covertype")
    assert Xtr.shape[1] == 54
    assert set(np.concatenate([ytr, yte]).tolist()) == set(range(7))


def test_wine_quality_three_class_binning(data_home):
    d = data_home / "wine_quality"
    d.mkdir(parents=True)
    header = ";".join(f'"{c}"' for c in ["fixed acidity"] + [f"c{i}" for i in
                                         range(10)] + ["quality"])
    rng = np.random.default_rng(3)
    for fn, qualities in (("winequality-red.csv", [3, 5, 6, 7, 6, 5, 8, 6]),
                          ("winequality-white.csv", [4, 6, 7, 5, 6, 6, 9, 5])):
        lines = [header] + [";".join(f"{v:.2f}" for v in rng.normal(size=11))
                            + f";{q}" for q in qualities]
        (d / fn).write_text("\n".join(lines) + "\n")
    Xtr, ytr, Xte, yte = ds.load("wine_quality")
    assert Xtr.shape[1] == 11
    y = np.concatenate([ytr, yte])
    # 16 rows: quality <=5 -> 0 (x6), ==6 -> 1 (x6), >=7 -> 2 (x4)
    assert np.bincount(y).tolist() == [6, 6, 4]


def test_banknote_from_sibling_perceptron_checkout(data_home):
    sibling = (Path(ds.__file__).resolve().parents[2] / "perceptron" / "data"
               / "data_banknote_authentication.txt")
    if not sibling.is_file():
        pytest.skip("sibling perceptron checkout not present")
    Xtr, ytr, Xte, yte = ds.load("banknote")
    assert len(Xtr) + len(Xte) == 1372                  # the UCI row count
    assert Xtr.shape[1] == 4
    assert set(np.unique(np.concatenate([ytr, yte]))) == {0, 1}


def test_mnist_flat_missing_uses_our_fix_command(data_home, monkeypatch):
    monkeypatch.setenv("MANTISSA_CNN_DATA", str(data_home / "nowhere"))
    with pytest.raises(FileNotFoundError, match=r"dataset 'mnist_flat' not "
                       r"downloaded — run: python -m mantissa_mlp.datasets "
                       r"download mnist_flat"):
        ds.load("mnist_flat")


# -- API contracts ----------------------------------------------------------------

def test_unknown_dataset():
    with pytest.raises(KeyError, match="unknown dataset"):
        ds.load("imagenet")


def test_weights_flag_is_higgsml_only(data_home):
    with pytest.raises(ValueError, match="only 'higgsml' carries physics"):
        ds.load("banknote", weights=True)


def test_subset_is_stratified_and_seeded(data_home):
    write_dimuon(data_home, [3.1] * 12 + [9.5] * 12 + [91.0] * 12)
    a = ds.subset("dimuon", 9, 6, seed=0)
    b = ds.subset("dimuon", 9, 6, seed=0)
    assert np.array_equal(a[0], b[0])
    assert np.bincount(a[1]).tolist() == [3, 3, 3]
    assert np.bincount(a[3]).tolist() == [2, 2, 2]
    c = ds.subset("dimuon", 9, 6, seed=1)
    assert not np.array_equal(a[0], c[0])
