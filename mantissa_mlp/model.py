"""MLP classifier for tabular data: Dense stacks + softmax cross-entropy + SGD.

Why this model owns its fit loop instead of wrapping mantissa_cnn.Sequential:
the family's Sequential is an image classifier — its data contract is NCHW
``(n, c, h, w)`` and its zoo assumes a Conv/Pool body. Tabular data is
``(n, d)`` float32, and pretending each row is a 1x1xd image would buy
nothing except reshapes in every user-facing call. The *loop itself* is
Sequential's, copied deliberately (same staging-buffer pattern, same loss,
same step order), so the two stay easy to diff.

Training loop design (mantissa_cnn.Sequential's):
- Shuffled mini-batches (seeded ``np.random.default_rng``), plain SGD
  (Robbins & Monro, 1951; the mini-batch form of LeCun et al., 1998).
- The head is ``Dense(classes)`` with the identity activation: it emits
  logits, and the loss applies softmax itself (fused softmax-cross-entropy,
  numerically stable via max-subtraction). Binary problems are 2-class
  softmax — one loss path, no separate sigmoid head.
- Memory: mini-batch input/label staging buffers are allocated once per fit
  (one set for full batches, one for the epoch tail) and refilled with
  ``np.take(..., out=...)``; Dense scratch is allocated once per batch shape
  (see mantissa_nn.layers). Steady-state training does no per-batch
  allocation.

Backends: the family's shared ones — ``backend="mantissa"`` (default) runs
every Dense/loss/SGD primitive in the C engine (via a per-model Session when
the engine offers one) and raises with the exact fix command when the engine
is missing; ``backend="numpy"`` is the pure-numpy reference oracle.

Data contract: X is (n, d) float32 (standardize it — see
:func:`mantissa_mlp.tasks.standardize`); y is integer class ids
0..classes-1.
"""
from __future__ import annotations

import numpy as np

from mantissa_nn import _numpy_backend
from mantissa_nn._engine import engine
from mantissa_nn.layers import Dense

__all__ = ["MLP"]

_HIDDEN_ACTS = ("relu", "tanh")


class MLP:
    """Fully connected classifier: hidden Dense stacks + a logits head.

    Parameters
    ----------
    hidden : tuple of int
        Hidden-layer widths, e.g. ``(64, 32)``. Must not be empty — with no
        hidden layer this would be a (multiclass) perceptron, and the family
        already has one of those.
    act : {"relu", "tanh"}
        Hidden activation. relu gets He-normal init, tanh Glorot-uniform
        (mantissa_nn.layers handles both). The head is always identity
        logits; softmax lives in the loss. Biases everywhere.
    classes : int or None
        Number of classes. None (default) infers ``max(y) + 1`` (at least 2)
        on the first fit.
    seed : int
        Seeds one rng stream for weight init and epoch shuffling — two
        models with the same seed and backend train identically.
    backend : {"mantissa", "numpy"}
        "mantissa" (default) requires the C engine and raises
        ImportError/RuntimeError with the exact fix otherwise.

    Fitted attributes
    -----------------
    history_ : dict with "loss" (per-epoch mean training loss) and "acc"
        (per-epoch training accuracy from each mini-batch's pre-update
        forward pass).
    n_features_, n_classes_ : set by build().
    """

    def __init__(self, hidden=(64, 32), act: str = "relu", classes=None,
                 seed: int = 0, backend: str = "mantissa"):
        if backend == "mantissa":
            tk = engine()                     # raises with the exact fix
            # mantissa >= 0.2.2: a per-model Session memoizes each buffer's
            # ctypes pointer by identity — our buffers are allocated once
            # and refilled in place, so pointer derivation becomes a dict
            # hit. Older engines just take the plain methods.
            self._backend = tk.session() if hasattr(tk, "session") else tk
        elif backend == "numpy":
            self._backend = _numpy_backend
        else:
            raise ValueError(f"backend must be 'mantissa' or 'numpy', got {backend!r}")
        self.backend = backend
        self.hidden = tuple(int(h) for h in hidden)
        if not self.hidden or any(h < 1 for h in self.hidden):
            raise ValueError(f"hidden must be a non-empty tuple of positive "
                             f"widths, got {hidden!r} — a zero-hidden-layer "
                             f"model is a perceptron, not an MLP")
        if act not in _HIDDEN_ACTS:
            raise ValueError(f"act must be one of {_HIDDEN_ACTS}, got {act!r}")
        self.act = act
        self.classes = None if classes is None else int(classes)
        if self.classes is not None and self.classes < 2:
            raise ValueError(f"classes must be >= 2, got {classes}")
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._built = False

    # -- construction ---------------------------------------------------------

    def build(self, n_features: int, classes=None):
        """Initialize parameters for ``n_features`` input columns. Called by
        fit() automatically; call it yourself to inspect summary() before
        training (needs ``classes`` from __init__ or here)."""
        classes = self.classes if classes is None else int(classes)
        if classes is None:
            raise ValueError("classes is unknown — pass classes= to MLP() "
                             "or build(), or let fit() infer it from y")
        self.layers = [Dense(h, act=self.act) for h in self.hidden]
        self.layers.append(Dense(classes))    # identity logits head
        shape = (int(n_features),)
        for layer in self.layers:
            shape = layer.build(shape, self._rng)
        self.n_features_ = int(n_features)
        self.n_classes_ = classes
        self._built = True
        return self

    def summary(self) -> str:
        """Per-layer output shapes and parameter counts (build() first)."""
        if not self._built:
            raise RuntimeError("summary() needs parameters — call "
                               "build(n_features) or fit() first")
        rows = [(type(l).__name__, str(l.out_shape), l.param_count())
                for l in self.layers]
        total = sum(r[2] for r in rows)
        w = max(len(r[0]) for r in rows)
        lines = [f"{'layer':<{w}}  {'out shape':<16}  params",
                 "-" * (w + 26)]
        lines += [f"{name:<{w}}  {shape:<16}  {p:,}" for name, shape, p in rows]
        lines.append(f"total params: {total:,}")
        return "\n".join(lines)

    # -- training -------------------------------------------------------------

    def fit(self, X, y, epochs: int = 10, batch_size: int = 32,
            lr: float = 0.01, verbose: bool = False):
        """Train with softmax cross-entropy on integer class ids (binary is
        the 2-class case)."""
        X = self._check_X(X)
        y = np.ascontiguousarray(y, dtype=np.int32).ravel()
        n = len(X)
        if len(y) != n:
            raise ValueError(f"X has {n} samples but y has {len(y)}")
        if y.min() < 0:
            raise ValueError(f"y must be non-negative class ids, got min {y.min()}")
        if not self._built:
            self.build(X.shape[1], classes=(self.classes if self.classes
                                            is not None
                                            else max(int(y.max()) + 1, 2)))
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, model was built "
                             f"for {self.n_features_}")
        if y.max() >= self.n_classes_:
            raise ValueError(f"y must be class ids in [0, {self.n_classes_}); "
                             f"got range [{y.min()}, {y.max()}]")

        backend = self._backend
        bs = min(int(batch_size), n)
        classes = self.n_classes_
        self.history_ = {"loss": [], "acc": []}

        # Per-fit staging buffers, refilled in place every batch: one set for
        # full batches, one for the epoch tail. No per-batch allocation.
        tail = n % bs
        Xb = np.empty((bs, self.n_features_), dtype=np.float32)
        yb = np.empty(bs, dtype=np.int32)
        dlog = np.empty((bs, classes), dtype=np.float32)
        Xt = np.empty((tail, self.n_features_), dtype=np.float32) if tail else None
        yt = np.empty(tail, dtype=np.int32) if tail else None
        dlogt = np.empty((tail, classes), dtype=np.float32) if tail else None

        for epoch in range(int(epochs)):
            order = self._rng.permutation(n)
            loss_sum = 0.0
            correct = 0
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                nb = len(idx)
                bx, by, bd = (Xb, yb, dlog) if nb == bs else (Xt, yt, dlogt)
                np.take(X, idx, axis=0, out=bx)
                np.take(y, idx, out=by)

                out = bx
                for layer in self.layers:
                    out = layer.forward(out, backend)

                loss = backend.softmax_xent(out, by, bd, nb, classes)
                loss_sum += loss * nb
                correct += int(np.count_nonzero(out.argmax(axis=1) == by))

                grad = bd
                for i in range(len(self.layers) - 1, -1, -1):
                    grad = self.layers[i].backward(grad, backend, need_dx=i > 0)
                for layer in self.layers:      # after ALL grads: dX of layer i
                    layer.step(backend, lr)    # depends on its pre-step params

            self.history_["loss"].append(loss_sum / n)
            self.history_["acc"].append(correct / n)
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}  "
                      f"loss {self.history_['loss'][-1]:.4f}  "
                      f"acc {self.history_['acc'][-1]:.4f}")
        return self

    # -- inference --------------------------------------------------------------

    def _logits(self, X, chunk: int = 1024):
        # chunk=1024 rather than the image models' 256: a tabular row is tens
        # of floats, not thousands, so bigger slices amortize the layer calls.
        if not self._built:
            raise RuntimeError("model has no parameters — call "
                               "build(n_features) or fit() first")
        X = self._check_X(X, expect=self.n_features_)
        out = np.empty((len(X), self.n_classes_), dtype=np.float32)
        for s in range(0, len(X), chunk):
            h = X[s:s + chunk]                 # contiguous slice view, no copy
            for layer in self.layers:
                h = layer.forward(h, self._backend)
            out[s:s + chunk] = h               # copy out: layer scratch is reused
        return out

    def predict_proba(self, X):
        """Softmax class probabilities, shape (n, classes)."""
        z = self._logits(X)
        z -= z.max(axis=1, keepdims=True)
        np.exp(z, out=z)
        z /= z.sum(axis=1, keepdims=True)
        return z

    def predict(self, X):
        """Predicted class ids, shape (n,)."""
        return self._logits(X).argmax(axis=1)

    def score(self, X, y) -> float:
        """Mean accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    # -- internals ---------------------------------------------------------------

    def _check_X(self, X, expect=None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be tabular (n, d) float32, got ndim={X.ndim}")
        if expect is not None and X.shape[1] != expect:
            raise ValueError(f"X has {X.shape[1]} features, model was built "
                             f"for {expect}")
        return X
