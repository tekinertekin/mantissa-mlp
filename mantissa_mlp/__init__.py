"""mantissa-mlp: multilayer perceptrons for tabular data on the mantissa
C engine, built on mantissa-cnn.

The family story: mantissa-perceptron's README ends at Minsky & Papert's
XOR limit — one neuron, one line, no XOR. This package is the resolution:
a hidden layer of mantissa-cnn Dense layers trained with backpropagation
(Rumelhart, Hinton & Williams, 1986). Everything dense comes from
mantissa-cnn (layers, C-engine and numpy backends); this package adds the
tabular training loop, a small cited zoo, six tabular datasets (two of
them published by CERN), and the AMS physics metric.
"""
try:
    import mantissa_cnn  # noqa: F401  (the base package: layers + backends)
except ImportError:
    raise ImportError(
        "mantissa-cnn is not installed — run: pip install mantissa-cnn"
    ) from None

from .model import MLP
from . import models, tasks

def __getattr__(name):
    # PEP 562 lazy import (mantissa-autoencoder's pattern): importing
    # .datasets points MANTISSA_CNN_DATA at the sibling cnn data/ for
    # mnist_flat — only do that side effect when datasets are actually used.
    if name == "datasets":
        import importlib
        return importlib.import_module(".datasets", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.1"
__all__ = ["MLP", "models", "tasks", "datasets"]
