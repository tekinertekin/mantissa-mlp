"""Backprop through two hidden layers vs central finite differences.

The numpy backend is the correctness oracle for the C engine, so the full
backward chain this package trains with — softmax-cross-entropy through
hidden Dense layers — must itself be verified against f(x+h) - f(x-h).
Buffers are float64 (the backend is dtype-agnostic; production is float32)
so tolerances can be tight. mantissa-cnn's own tests gradcheck each layer
in isolation; this checks the *composition* the MLP actually runs: the
gradient that reaches W1 has flowed through the loss and two activations.
"""
import numpy as np
import pytest

from mantissa_cnn import _numpy_backend as B

EPS = 1e-6
RTOL = 1e-5
ATOL = 1e-8

N, D, H1, H2, C = 4, 5, 4, 3, 3


def _fd(f, x, eps=EPS):
    g = np.empty_like(x)
    flat, gf = x.reshape(-1), g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        hi = f()
        flat[i] = old - eps
        lo = f()
        flat[i] = old
        gf[i] = (hi - lo) / (2 * eps)
    return g


@pytest.mark.parametrize("act", [B.RELU, B.TANH])
def test_mlp_backprop_two_hidden_layers(act):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(N, D))
    # relu's z==0 kink breaks finite differences; nudge biases off it the
    # way mantissa's own gradcheck suite does.
    params = {
        "W1": rng.normal(size=(H1, D)), "b1": rng.normal(size=H1) + 0.1,
        "W2": rng.normal(size=(H2, H1)), "b2": rng.normal(size=H2) + 0.1,
        "W3": rng.normal(size=(C, H2)), "b3": rng.normal(size=C),
    }
    labels = rng.integers(0, C, size=N).astype(np.int32)

    Z1, Y1 = np.empty((N, H1)), np.empty((N, H1))
    Z2, Y2 = np.empty((N, H2)), np.empty((N, H2))
    Z3, Y3 = np.empty((N, C)), np.empty((N, C))
    dlog = np.empty((N, C))

    def forward_loss():
        B.linear_forward_batch(params["W1"], X, params["b1"], Z1, Y1, N, H1, D, act)
        B.linear_forward_batch(params["W2"], Y1, params["b2"], Z2, Y2, N, H2, H1, act)
        B.linear_forward_batch(params["W3"], Y2, params["b3"], Z3, Y3,
                               N, C, H2, B.IDENTITY)
        return B.softmax_xent(Y3, labels, dlog, N, C)

    forward_loss()
    grads = {k: np.empty_like(v) for k, v in params.items()}
    dX2, dX1, dX0 = np.empty((N, H2)), np.empty((N, H1)), np.empty((N, D))
    B.linear_backward_batch(params["W3"], Y2, Z3, dlog, grads["W3"], grads["b3"],
                            dX2, N, C, H2, B.IDENTITY)
    B.linear_backward_batch(params["W2"], Y1, Z2, dX2, grads["W2"], grads["b2"],
                            dX1, N, H2, H1, act)
    B.linear_backward_batch(params["W1"], X, Z1, dX1, grads["W1"], grads["b1"],
                            dX0, N, H1, D, act)

    for name, value in params.items():
        fd = _fd(forward_loss, value)
        assert np.allclose(grads[name], fd, rtol=RTOL, atol=ATOL), \
            f"analytic {name} disagrees with finite differences"
    # and the gradient w.r.t. the input, through the whole stack
    assert np.allclose(dX0, _fd(forward_loss, X), rtol=RTOL, atol=ATOL)
