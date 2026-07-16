"""AMS against a hand-computed case; standardization with train-only stats
and the -999.0 sentinel contract."""
import math

import numpy as np
import pytest

from mantissa_mlp import tasks


# -- ams -----------------------------------------------------------------------

def test_ams_hand_computed():
    # selected: rows 0,1 (signal, w 1+1 -> s=2) and 2,3 (background,
    # w 1+2 -> b=3); row 4 is unselected signal and must not count.
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1, 0])
    w = np.array([1.0, 1.0, 1.0, 2.0, 5.0])
    s, b = 2.0, 3.0
    expected = math.sqrt(2 * ((s + b + 10) * math.log(1 + s / (b + 10)) - s))
    assert tasks.ams(y_true, y_pred, w) == pytest.approx(expected)


def test_ams_empty_selection_is_zero():
    assert tasks.ams([1, 0], [0, 0], [3.0, 4.0]) == 0.0


def test_ams_b_reg_matters():
    # with b_reg=0 and pure-signal selection, AMS -> sqrt(2(s ln... )) form;
    # just pin that the regularizer changes the number the documented way.
    val10 = tasks.ams([1], [1], [5.0])
    val0 = tasks.ams([1], [1], [5.0], b_reg=1.0)
    assert val0 > val10 > 0


def test_ams_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        tasks.ams([1, 0], [1], [1.0, 1.0])


# -- standardize -----------------------------------------------------------------

def test_standardize_train_stats_only():
    Xtr = np.array([[0.0, 10.0], [2.0, 30.0]], dtype=np.float32)
    Xte = np.array([[1.0, 20.0]], dtype=np.float32)
    Str, Ste = tasks.standardize(Xtr, Xte)
    assert np.allclose(Str.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(Str.std(axis=0), 1.0, atol=1e-6)
    # the test row sits at the train mean -> exactly 0 under TRAIN stats
    assert np.allclose(Ste, 0.0, atol=1e-6)


def test_standardize_does_not_leak_test_stats():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(50, 3)).astype(np.float32)
    Xte = rng.normal(loc=100.0, size=(10, 3)).astype(np.float32)  # shifted
    _, Ste = tasks.standardize(Xtr, Xte)
    assert Ste.mean() > 50.0          # test mean must NOT be centred away


def test_standardize_missing_sentinel_contract():
    """The ONE documented -999.0 approach: sentinel excluded from the train
    stats, then imputed as 0.0 (the post-standardization mean) on both
    sides."""
    Xtr = np.array([[1.0, -999.0],
                    [3.0, 4.0],
                    [5.0, 6.0]], dtype=np.float32)
    Xte = np.array([[3.0, -999.0]], dtype=np.float32)
    Str, Ste = tasks.standardize(Xtr, Xte, missing=-999.0)
    # column 0 has no sentinels: plain standardization of [1, 3, 5]
    assert np.allclose(Str[:, 0], [-math.sqrt(1.5), 0.0, math.sqrt(1.5)],
                       atol=1e-6)
    # column 1: stats from the defined [4, 6] only -> mean 5, sd 1
    assert np.allclose(Str[1:, 1], [-1.0, 1.0], atol=1e-6)
    assert Str[0, 1] == 0.0                       # sentinel -> neutral 0
    assert Ste[0, 1] == 0.0                       # test sentinel too
    assert Ste[0, 0] == 0.0                       # 3.0 is the train mean


def test_standardize_constant_column_gets_unit_sd():
    Xtr = np.array([[2.0], [2.0]], dtype=np.float32)
    Str = tasks.standardize(Xtr)
    assert np.allclose(Str, 0.0)                  # no division by zero


def test_standardize_validation():
    with pytest.raises(ValueError, match="tabular"):
        tasks.standardize(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="X_test must be"):
        tasks.standardize(np.zeros((2, 3), dtype=np.float32),
                          np.zeros((2, 4), dtype=np.float32))


def test_standardize_returns_copies():
    Xtr = np.ones((2, 2), dtype=np.float32)
    out = tasks.standardize(Xtr)
    out[0, 0] = 99.0
    assert Xtr[0, 0] == 1.0
