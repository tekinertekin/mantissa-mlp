"""MLP end-to-end: the XOR signature test, fit/predict contracts, chunking
edges, multiclass, and mantissa-vs-numpy parity (skipif-guarded, mirroring
the family's guard)."""
import numpy as np
import pytest

import mantissa_cnn._engine as eng

from mantissa_mlp import MLP, models

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0, 1, 1, 0], dtype=np.int32)


def _engine_ready() -> bool:
    try:
        eng.cnn_engine()
        return True
    except Exception:
        return False


def blobs(n=90, d=5, classes=3, seed=0, spread=4.0):
    """Well-separated Gaussian blobs — learnable in a few epochs."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=spread, size=(classes, d))
    y = np.arange(n, dtype=np.int32) % classes
    X = (centers[y] + rng.normal(size=(n, d))).astype(np.float32)
    return X, y


# -- the signature test ------------------------------------------------------

def test_xor_learns():
    """THE test of the repo: the perceptron family's Minsky-Papert
    cliffhanger, resolved by one hidden layer + backprop. The recipe is
    xor_net's documented one (per-pattern updates, as in Rumelhart, Hinton
    & Williams 1986), pinned to the numpy oracle backend and seed 0."""
    net = models.xor_net(seed=0, backend="numpy")
    net.fit(XOR_X, XOR_Y, epochs=500, batch_size=1, lr=0.3)
    assert net.score(XOR_X, XOR_Y) > 0.99                # all four corners
    assert net.history_["loss"][-1] < 0.05               # confidently right


# -- learning behavior --------------------------------------------------------

def test_multiclass_blobs_learn():
    X, y = blobs()
    net = MLP(hidden=(16,), classes=3, seed=0, backend="numpy")
    net.fit(X, y, epochs=30, batch_size=16, lr=0.05)
    assert net.score(X, y) > 0.95
    assert net.history_["loss"][-1] < net.history_["loss"][0]
    assert len(net.history_["loss"]) == len(net.history_["acc"]) == 30


def test_same_seed_same_run():
    X, y = blobs()
    runs = [MLP(hidden=(8, 4), seed=3, backend="numpy")
            .fit(X, y, epochs=3, batch_size=16, lr=0.05) for _ in range(2)]
    a, b = runs
    assert a.history_["loss"] == b.history_["loss"]
    for la, lb in zip(a.layers, b.layers):
        assert np.array_equal(la.W, lb.W)
        assert np.array_equal(la.b, lb.b)


def test_partial_tail_batch_is_handled():
    X, y = blobs(n=40)                       # bs 16 -> tail 8
    net = MLP(hidden=(8,), seed=0, backend="numpy")
    net.fit(X, y, epochs=2, batch_size=16, lr=0.01)
    assert len(net.history_["loss"]) == 2


def test_classes_inferred_from_y():
    X, y = blobs(n=60, classes=5)
    net = MLP(hidden=(8,), backend="numpy").fit(X, y, epochs=1)
    assert net.n_classes_ == 5
    net2 = MLP(hidden=(8,), backend="numpy").fit(X[:20], (y[:20] > 0).astype(np.int32),
                                                 epochs=1)
    assert net2.n_classes_ == 2              # binary floor: max(y)+1 >= 2


# -- inference ---------------------------------------------------------------

def test_predict_chunking_edges():
    """1030 rows crosses the 1024-row chunk boundary; results must equal
    the single-chunk pass and predict_proba must be a proper softmax."""
    X, y = blobs(n=90)
    net = MLP(hidden=(8,), classes=3, seed=0, backend="numpy")
    net.fit(X, y, epochs=2, batch_size=16)
    Xbig = np.tile(X, (12, 1))[:1030]
    logits_chunked = net._logits(Xbig)
    logits_whole = net._logits(Xbig, chunk=len(Xbig))
    assert logits_chunked.shape == (1030, 3)
    # allclose, not equal: BLAS picks different reduction orders for the
    # 1024-row and 1030-row GEMMs, so the last bits may differ
    assert np.allclose(logits_chunked, logits_whole, rtol=1e-5, atol=1e-6)
    # deterministic: the same chunked pass repeats bit-identically
    assert np.array_equal(logits_chunked, net._logits(Xbig))
    p = net.predict_proba(Xbig)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(net.predict(Xbig), p.argmax(axis=1))


def test_score_is_mean_accuracy():
    X, y = blobs(n=30)
    net = MLP(hidden=(8,), classes=3, seed=0, backend="numpy").fit(X, y, epochs=5)
    assert net.score(X, y) == pytest.approx(float(np.mean(net.predict(X) == y)))


# -- contracts ----------------------------------------------------------------

def test_constructor_validation():
    with pytest.raises(ValueError, match="backend must be 'mantissa' or 'numpy'"):
        MLP(backend="torch")
    with pytest.raises(ValueError, match="hidden must be a non-empty tuple"):
        MLP(hidden=(), backend="numpy")
    with pytest.raises(ValueError, match="act must be one of"):
        MLP(act="gelu", backend="numpy")
    with pytest.raises(ValueError, match="classes must be >= 2"):
        MLP(classes=1, backend="numpy")


def test_fit_validation():
    X, y = blobs(n=30)
    net = MLP(hidden=(8,), backend="numpy")
    with pytest.raises(ValueError, match="X must be tabular"):
        net.fit(X.reshape(30, 5, 1), y)
    with pytest.raises(ValueError, match="X has 30 samples but y has 10"):
        net.fit(X, y[:10])
    with pytest.raises(ValueError, match="non-negative class ids"):
        net.fit(X, y - 1)
    net_fixed = MLP(hidden=(8,), classes=2, backend="numpy")
    with pytest.raises(ValueError, match=r"y must be class ids in \[0, 2\)"):
        net_fixed.fit(X, y)                  # y has a 2, classes=2
    net.fit(X, y, epochs=1)
    with pytest.raises(ValueError, match="X has 4 features, model was built for 5"):
        net.fit(X[:, :4], y)
    with pytest.raises(ValueError, match="X has 4 features, model was built for 5"):
        net.predict(X[:, :4])


def test_unbuilt_model_raises():
    net = MLP(backend="numpy")
    with pytest.raises(RuntimeError, match="model has no parameters"):
        net.predict(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="summary"):
        net.summary()
    with pytest.raises(ValueError, match="classes is unknown"):
        net.build(3)                         # classes=None and none passed


def test_summary_lists_all_layers():
    net = MLP(hidden=(8, 4), classes=3, backend="numpy").build(5)
    s = net.summary()
    assert s.count("Dense") == 3
    # 5*8+8 + 8*4+4 + 4*3+3 = 48 + 36 + 15
    assert "total params: 99" in s


# -- backend parity ------------------------------------------------------------

@pytest.mark.skipif(not _engine_ready(),
                    reason="mantissa engine with CNN primitives not available")
def test_backend_parity_two_training_steps():
    """Same seed, same data, 2 SGD steps: C engine == numpy oracle."""
    X, y = blobs(n=32)

    def train(backend):
        net = MLP(hidden=(8, 4), classes=3, seed=1, backend=backend)
        net.fit(X, y, epochs=2, batch_size=32, lr=0.05)  # 1 batch = 1 step/epoch
        return net

    a, b = train("mantissa"), train("numpy")
    assert np.allclose(a.history_["loss"], b.history_["loss"], rtol=1e-4)
    for la, lb in zip(a.layers, b.layers):
        assert np.allclose(la.W, lb.W, rtol=1e-4, atol=1e-5)
        assert np.allclose(la.b, lb.b, rtol=1e-4, atol=1e-5)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X),
                       rtol=1e-4, atol=1e-5)
