"""The two zoo architectures re-expressed layer-for-layer in each framework,
plus the data prep — everything bench/speed.py needs that is not measurement.

Per dataset the protocol pins one zoo shape (bench.protocol.MODELS):
``higgs_mlp`` is 30->300->200->100->2 relu; ``tabular_mlp`` is d->64->32->C
relu. Each is rebuilt identically in torch, tensorflow and scikit-learn.

Estimator surface (uniform across contenders; construction is untimed,
``fit`` is the timed region):

- ``factory(dataset, d, classes)`` -> fresh estimator, weights initialized
  with the same init family as ours (He normal before relu, Glorot uniform
  on the identity-activation logits head, zero biases). torch's and keras'
  own defaults differ; matching the init keeps the comparison about the
  frameworks, not the initializer (the cnn/autoencoder benchmarks' policy).
  scikit-learn draws its own init and does not expose a hook — recorded as a
  fairness caveat rather than papered over.
- ``fit(X, y)`` trains under the fixed protocol (bench.protocol: plain SGD,
  lr LR, batch BATCH_SIZE, EPOCHS epochs, softmax cross-entropy, seeded
  shuffles, CPU) and sets ``final_loss_``.
- ``predict(X)`` -> integer class ids (numpy).
- ``predict_proba(X)`` -> (n, classes) numpy probabilities — the AMS metric
  needs P(signal) on higgsml.
- ``param_count()`` -> trainable parameter count.

``X``/``y`` arrive in the contender's native form via its ``prep_X``/``prep_y``
(float32 numpy for ours/tf/sklearn, float32/int64 tensors for torch) —
conversion happens once, outside the timed region.

Structural parity (``python -m bench.contenders``): the softmax family (ours,
its numpy backend by construction, torch, tensorflow) carries an *identical*
parameter count for every dataset; scikit-learn matches exactly on the four
multiclass datasets, and on the two binary ones (higgsml, banknote) its head
is a single logistic unit rather than a 2-logit softmax — a genuine
sigmoid-vs-softmax structural difference of ``last_hidden + 1`` fewer
parameters, asserted in that exact form rather than hidden.
"""
from __future__ import annotations

import os

# Keep TensorFlow's C++ banner out of benchmark output (set before any TF
# import anywhere in the process).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

import numpy as np

# numpy 2.x on Apple Accelerate emits spurious FPE RuntimeWarnings from the
# BLAS matmul kernel even on finite inputs (the cnn/perceptron benchmarks
# document this: contender weights stay bounded). They fire from sklearn's
# and the vanilla-numpy backend's matmuls.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

from mantissa_mlp import models

from .protocol import BATCH_SIZE, CONTENDERS, EPOCHS, LR, MODELS, SEED

__all__ = ["OursMLP", "TorchMLP", "KerasMLP", "SklearnMLP",
           "arch_spec", "contenders", "check_parity"]

# JSON keys, in protocol.CONTENDERS display order. protocol pins the names;
# these are the stable machine keys the results JSON and plots use.
KEYS = ("ours", "vanilla_numpy", "torch", "tensorflow", "sklearn")
assert len(KEYS) == len(CONTENDERS)


def arch_spec(dataset: str):
    """(zoo_model_name, hidden_widths) for a dataset — the shape every
    contender rebuilds. Reads bench.protocol.MODELS so the benchmark cannot
    drift from the pinned zoo."""
    model = MODELS[dataset]
    if model == "higgs_mlp":
        return "higgs_mlp", (300, 200, 100)
    if model == "tabular_mlp":
        return "tabular_mlp", (64, 32)
    raise ValueError(f"unmapped zoo model {model!r} for dataset {dataset!r}")


# --- ours (both backends) ----------------------------------------------------

class OursMLP:
    """mantissa_mlp.models.<zoo> on the chosen backend."""

    def __init__(self, dataset, d, classes, backend):
        model, _hidden = arch_spec(dataset)
        if model == "higgs_mlp":
            self._net = models.higgs_mlp(seed=SEED, backend=backend)
        else:
            self._net = models.tabular_mlp(d, classes, seed=SEED,
                                           backend=backend)

    def fit(self, X, y):
        self._net.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
        self.final_loss_ = float(self._net.history_["loss"][-1])
        return self

    def predict(self, X):
        return self._net.predict(X)

    def predict_proba(self, X):
        return self._net.predict_proba(X)

    def param_count(self):
        return sum(l.param_count() for l in self._net.layers)


# --- torch -------------------------------------------------------------------

def _torch_layers(d, hidden, classes):
    """Linear/ReLU stack mirroring the zoo: identity-logits head, softmax in
    the loss (CrossEntropyLoss)."""
    import torch.nn as nn
    seq, prev = [], d
    for h in hidden:
        seq += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    seq.append(nn.Linear(prev, classes))    # identity logits head
    return nn.Sequential(*seq)


class TorchMLP:
    """torch.nn.Sequential, eager, explicit seeded mini-batch SGD loop with
    the same shuffle-stream construction as ours (np rng permutation), and
    the same init family (He normal on relu layers, Glorot uniform on the
    logits head, zero biases)."""

    def __init__(self, dataset, d, classes):
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        _model, hidden = arch_spec(dataset)
        self._m = _torch_layers(d, hidden, classes)
        g = torch.Generator().manual_seed(SEED)
        lin = [m for m in self._m if isinstance(m, nn.Linear)]
        for m in lin:
            nn.init.zeros_(m.bias)
        for m in lin[:-1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu", generator=g)
        nn.init.xavier_uniform_(lin[-1].weight, generator=g)
        self._rng = np.random.default_rng(SEED)   # seeded epoch shuffle

    def fit(self, X, y):
        import torch
        m = self._m
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.0)
        loss_fn = torch.nn.CrossEntropyLoss()
        n = len(X)
        for _ in range(EPOCHS):
            order = self._rng.permutation(n)
            loss_sum = 0.0
            for s in range(0, n, BATCH_SIZE):
                idx = torch.from_numpy(order[s:s + BATCH_SIZE])
                opt.zero_grad()
                loss = loss_fn(m(X[idx]), y[idx])
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(idx)
            self.final_loss_ = loss_sum / n
        return self

    def predict(self, X):
        import torch
        self._m.eval()
        with torch.no_grad():
            return self._m(X).argmax(1).numpy()

    def predict_proba(self, X):
        import torch
        self._m.eval()
        with torch.no_grad():
            return torch.softmax(self._m(X), 1).numpy()

    def param_count(self):
        return sum(p.numel() for p in self._m.parameters())


# --- tensorflow / keras ------------------------------------------------------

class KerasMLP:
    """tf.keras.Sequential, same layers and init family as ours; built +
    compiled in the constructor — outside the timed region, like any one-time
    setup. Each repeat still gets a fresh model, so weight init is per repeat.
    Softmax lives in the loss (from_logits=True)."""

    def __init__(self, dataset, d, classes):
        import keras
        keras.utils.set_random_seed(SEED)   # init + fit(shuffle=True) shuffling
        _model, hidden = arch_spec(dataset)
        L = keras.layers
        he = keras.initializers.HeNormal(seed=SEED)
        glorot = keras.initializers.GlorotUniform(seed=SEED)
        layers = [L.Dense(h, activation="relu", kernel_initializer=he)
                  for h in hidden]
        layers.append(L.Dense(classes, kernel_initializer=glorot))   # logits
        m = keras.Sequential([keras.Input((d,))] + layers)
        m.compile(optimizer=keras.optimizers.SGD(learning_rate=LR, momentum=0.0),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self._m = m

    def fit(self, X, y):
        h = self._m.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        shuffle=True, verbose=0)
        self.final_loss_ = float(h.history["loss"][-1])
        return self

    def predict(self, X):
        return self._m.predict(X, verbose=0).argmax(1)

    def predict_proba(self, X):
        import keras
        return np.asarray(keras.activations.softmax(
            keras.ops.convert_to_tensor(self._m.predict(X, verbose=0))))

    def param_count(self):
        return int(self._m.count_params())


# --- scikit-learn ------------------------------------------------------------

class SklearnMLP:
    """sklearn.neural_network.MLPClassifier pinned to the protocol: solver
    'sgd', constant learning rate, momentum 0, batch BATCH_SIZE, max_iter
    EPOCHS, alpha 0 (no L2 — everyone else trains undecayed). A REAL MLP with
    Cython SGD underneath, so a full contender here (unlike the cnn repo).

    Two honest, unavoidable differences, recorded not hidden: sklearn draws
    its own (Glorot-style) init and exposes no hook to match ours; and for a
    binary problem it uses ONE logistic output unit (log-loss) rather than a
    2-logit softmax head, so higgsml/banknote have last_hidden+1 fewer
    parameters. Both are properties of the library, not tuning."""

    def __init__(self, dataset, d, classes):
        from sklearn.neural_network import MLPClassifier
        _model, hidden = arch_spec(dataset)
        # max_iter caps epochs at EPOCHS; n_iter_no_change kept above it so
        # early stopping never fires before the cap (all EPOCHS run, matching
        # every other contender). tol irrelevant then, left default.
        self._clf = MLPClassifier(
            hidden_layer_sizes=tuple(hidden), activation="relu",
            solver="sgd", alpha=0.0, batch_size=BATCH_SIZE,
            learning_rate="constant", learning_rate_init=LR,
            momentum=0.0, nesterovs_momentum=False, shuffle=True,
            max_iter=EPOCHS, n_iter_no_change=EPOCHS + 1, random_state=SEED)
        self._classes = classes

    def fit(self, X, y):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            # max_iter=EPOCHS is the protocol epoch budget, not a convergence
            # target — the "did not converge" warning is expected and noise.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._clf.fit(X, y)
        self.final_loss_ = float(self._clf.loss_)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def param_count(self):
        return int(sum(c.size for c in self._clf.coefs_)
                   + sum(b.size for b in self._clf.intercepts_))


# --- registry ----------------------------------------------------------------
# (key, factory, prep_X, prep_y). prep maps float32 numpy into the contender's
# native form ONCE, outside the timed region, so fit() measures training only.
# Heavy imports live inside the classes so an RSS worker pays only for the
# framework it actually uses.

def _to_f32(X):
    return np.ascontiguousarray(X, dtype=np.float32)


def _to_i32(y):
    return np.ascontiguousarray(y, dtype=np.int32)


def _prep_torch_X(X):
    import torch
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))


def _prep_torch_y(y):
    import torch
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))


def contenders():
    reg = [
        ("ours", lambda ds, d, c: OursMLP(ds, d, c, "mantissa"),
         _to_f32, _to_i32),
        ("vanilla_numpy", lambda ds, d, c: OursMLP(ds, d, c, "numpy"),
         _to_f32, _to_i32),
        ("torch", TorchMLP, _prep_torch_X, _prep_torch_y),
        ("tensorflow", KerasMLP, _to_f32, _to_i32),
        ("sklearn", SklearnMLP, _to_f32, _to_i32),
    ]
    assert tuple(n for n, *_ in reg) == KEYS
    return reg


# --- structural parity -------------------------------------------------------

def check_parity(verbose: bool = True):
    """Assert parameter-count parity per dataset. The softmax family (ours,
    torch, tensorflow) must be byte-identical; vanilla_numpy shares ours'
    layer objects, so it is covered by 'ours'. scikit-learn must match exactly
    on multiclass and, on binary, be exactly ``last_hidden + 1`` short (its
    single logistic head)."""
    rows = {}
    for dataset in MODELS:
        _model, hidden = arch_spec(dataset)
        d, classes = _shape_for(dataset)
        soft = {"ours": OursMLP(dataset, d, classes, "mantissa").param_count(),
                "torch": TorchMLP(dataset, d, classes).param_count(),
                "tensorflow": KerasMLP(dataset, d, classes).param_count()}
        assert len(set(soft.values())) == 1, \
            f"softmax-family parameter mismatch for {dataset}: {soft}"
        sk = SklearnMLP(dataset, d, classes)
        # Small standardized-scale fit so coefs_ exist; >= one full batch of
        # every class so sklearn does not clip batch_size (that path is
        # exercised for real in the benchmark, not here).
        rng = np.random.default_rng(SEED)
        n_tiny = max(BATCH_SIZE * 2, classes * 8)
        y_tiny = np.tile(np.arange(classes), n_tiny // classes + 1)[:n_tiny]
        sk.fit(rng.standard_normal((n_tiny, d)).astype(np.float32), y_tiny)
        sk_params = sk.param_count()
        base = soft["ours"]
        if classes == 2:
            expected = base - (hidden[-1] + 1)     # one logistic unit, not 2
            note = f"binary sigmoid head (softmax {base} - {hidden[-1] + 1})"
        else:
            expected = base
            note = "exact"
        assert sk_params == expected, \
            f"sklearn parameter mismatch for {dataset}: {sk_params} != {expected}"
        counts = dict(soft, vanilla_numpy=base, sklearn=sk_params)
        rows[dataset] = counts
        if verbose:
            print(f"{dataset:13s} softmax-family={base:>7,}  "
                  f"sklearn={sk_params:>7,}  ({note})  OK")
    return rows


def _shape_for(dataset):
    """(d, classes) for a dataset without loading it — the fixed feature/class
    counts documented in mantissa_mlp.datasets. Used only by the parity check;
    the benchmark itself reads d/classes from the loaded subset."""
    return {"higgsml": (30, 2), "covertype": (54, 7), "dimuon": (16, 3),
            "mnist_flat": (784, 10), "wine_quality": (11, 3),
            "banknote": (4, 2)}[dataset]


if __name__ == "__main__":
    check_parity()
