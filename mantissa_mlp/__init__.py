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

__version__ = "0.1.0"
__all__ = ["MLP"]
