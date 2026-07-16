"""Zoo contracts: built models, honest shapes, documented parameter counts."""
import numpy as np

from mantissa_mlp import models


def test_xor_net_is_the_minimal_net():
    net = models.xor_net(backend="numpy")
    assert net.hidden == (2,)
    assert net.act == "tanh"
    assert net.n_features_ == 2 and net.n_classes_ == 2
    # 2*2+2 hidden + 2*2+2 head — twelve numbers to end the XOR story
    assert sum(l.param_count() for l in net.layers) == 12
    assert "Dense" in net.summary()


def test_tabular_mlp_shapes():
    net = models.tabular_mlp(11, 3, backend="numpy")
    assert net.hidden == (64, 32)
    assert net.n_features_ == 11 and net.n_classes_ == 3
    custom = models.tabular_mlp(4, 2, hidden=(8,), backend="numpy")
    assert [l.units for l in custom.layers] == [8, 2]


def test_higgs_mlp_is_600_hidden_units_on_30_features():
    net = models.higgs_mlp(backend="numpy")
    assert net.n_features_ == 30 and net.n_classes_ == 2
    assert sum(net.hidden) == 600                    # the HiggsML-winner scale
    # 30*300+300 + 300*200+200 + 200*100+100 + 100*2+2
    assert sum(l.param_count() for l in net.layers) == 89802


def test_zoo_models_are_built_and_seeded():
    a = models.tabular_mlp(5, 2, seed=7, backend="numpy")
    b = models.tabular_mlp(5, 2, seed=7, backend="numpy")
    assert np.array_equal(a.layers[0].W, b.layers[0].W)
    assert a.summary() == b.summary()
