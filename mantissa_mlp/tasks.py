"""Task helpers: the AMS physics metric and train-stats-only standardization.

Numpy only — nothing here touches the engine; metrics and preprocessing are
memory-bound array passes, the honest tool is numpy (the same call the
family makes about losses).
"""
from __future__ import annotations

import numpy as np

__all__ = ["ams", "standardize"]


def ams(y_true, y_pred, weights, b_reg: float = 10.0) -> float:
    """Approximate Median Significance of a selection, the HiggsML metric.

    ``y_pred`` marks the selected events (1 = selected as signal). With
    ``s`` the summed weights of selected true-signal events (weighted true
    positives) and ``b`` the summed weights of selected background events
    (weighted false positives),

        AMS = sqrt( 2 * ( (s + b + b_reg) * ln(1 + s / (b + b_reg)) - s ) )

    — the exact formula of Adam-Bourdarios, Cowan, Germain, Guyon, Kégl &
    Rousseau (2015), "The Higgs boson machine learning challenge", *JMLR
    W&CP* 42, section 3, with their regularization term ``b_reg = 10``
    (it damps the metric's variance when b is small). AMS approximates the
    median discovery significance, in Gaussian sigmas, that this selection
    would achieve — the physics answer to "how confidently would we claim
    the signal exists?", which is why the challenge scored it rather than
    accuracy: unweighted, the simulated signal is ~1/3 of events, but the
    weights renormalize to the real detector rates where the signal is
    vanishingly rare, so a high-accuracy classifier can still be a poor
    selector.

    Weights must be renormalized to the full dataset's per-class totals for
    numbers comparable across subsets — the loaders in
    :mod:`mantissa_mlp.datasets` do that (``weights=True``).

    Selecting nothing gives s = 0, hence AMS = 0.0.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if not (len(y_true) == len(y_pred) == len(w)):
        raise ValueError(f"y_true, y_pred and weights must have equal length, "
                         f"got {len(y_true)}, {len(y_pred)}, {len(w)}")
    sel = y_pred == 1
    s = float(w[sel & (y_true == 1)].sum())
    b = float(w[sel & (y_true == 0)].sum())
    return float(np.sqrt(2.0 * ((s + b + b_reg) * np.log1p(s / (b + b_reg)) - s)))


def standardize(X_train, X_test=None, missing=None):
    """Center/scale features using train-set statistics only — no test
    leakage (the family protocol since mantissa-perceptron's ``split``).

    ``missing``, if given, is a sentinel value marking entries that are
    *undefined*, not merely unmeasured — higgsml uses -999.0 for features
    that do not exist for an event's topology (e.g. jet variables when the
    event has no jets; see the challenge documentation, Adam-Bourdarios et
    al. 2015, appendix B). The documented approach here, chosen once for
    the whole package: mean/std are computed over the **defined** train
    entries only, defined entries are standardized with those statistics,
    and undefined entries become **0.0 — the post-standardization mean**,
    so they are exactly neutral to a Dense layer's weighted sum. (The
    alternative, missing-indicator columns, adds features and was not
    chosen; one approach, applied everywhere, keeps every model and
    benchmark comparable.)

    Constant (or all-missing) train columns get sd 1.0, so they standardize
    to a constant instead of dividing by zero.

    Returns float32 copies: ``X_train`` alone, or ``(X_train, X_test)``
    when a test set is passed (transformed with the *train* statistics).
    """
    Xtr = np.array(X_train, dtype=np.float32)          # copy: edited in place
    if Xtr.ndim != 2:
        raise ValueError(f"X_train must be tabular (n, d), got ndim={Xtr.ndim}")
    Xte = None
    if X_test is not None:
        Xte = np.array(X_test, dtype=np.float32)
        if Xte.ndim != 2 or Xte.shape[1] != Xtr.shape[1]:
            raise ValueError(f"X_test must be (n, {Xtr.shape[1]}), "
                             f"got shape {Xte.shape}")

    if missing is None:
        mu = Xtr.mean(axis=0, dtype=np.float64)
        sd = Xtr.std(axis=0, dtype=np.float64)
    else:
        defined = Xtr != np.float32(missing)
        cnt = defined.sum(axis=0)
        safe = np.maximum(cnt, 1)
        mu = np.where(defined, Xtr, 0.0).sum(axis=0, dtype=np.float64) / safe
        var = (np.where(defined, (Xtr - mu) ** 2, 0.0)
               .sum(axis=0, dtype=np.float64) / safe)
        sd = np.sqrt(var)
    sd[sd == 0.0] = 1.0
    mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)

    def apply(X):
        if missing is not None:
            undef = X == np.float32(missing)
        X -= mu32
        X /= sd32
        if missing is not None:
            X[undef] = 0.0                    # the post-standardization mean
        return np.ascontiguousarray(X, dtype=np.float32)

    Xtr = apply(Xtr)
    return Xtr if Xte is None else (Xtr, apply(Xte))
