"""Model zoo: three cited MLP recipes, honestly named.

Each function returns a built :class:`~mantissa_mlp.model.MLP` (parameters
initialized, so ``summary()`` works immediately). All of them are plain
Dense stacks trained with softmax cross-entropy — the head is always
identity logits, softmax lives in the loss.
"""
from __future__ import annotations

from .model import MLP

__all__ = ["xor_net", "tabular_mlp", "higgs_mlp"]


def xor_net(seed: int = 0, backend: str = "mantissa") -> MLP:
    """The minimal net that resolves the family's cliffhanger: 2 inputs ->
    2 hidden tanh units -> output. A single neuron cannot represent XOR
    (Minsky & Papert, *Perceptrons*, 1969 — where the mantissa-perceptron
    README ends); one hidden layer of two units can, and backpropagation
    finds the weights (Rumelhart, Hinton & Williams, 1986, "Learning
    representations by back-propagating errors", *Nature* 323 — XOR is
    the paper's first worked example). One hidden layer also suffices in
    general: Cybenko (1989), "Approximation by Superpositions of a
    Sigmoidal Function", *MCSS* 2(4).

    Deviations from the paper's 2-2-1 net, flagged: the family head is
    2-logit softmax rather than a single sigmoid unit (same decision
    boundary, one loss path for the whole package), and the hidden units
    are tanh rather than logistic sigmoid.

    The recipe that reliably learns XOR from this seed::

        net = models.xor_net(backend="numpy")
        net.fit(X_xor, y_xor, epochs=500, batch_size=1, lr=0.3)

    ``batch_size=1`` is per-pattern updating, as in the 1986 paper. XOR at
    exactly two hidden units has genuine bad minima — Rumelhart et al.
    themselves report sticking on rare runs, and most (seed, lr) pairs
    under full-batch descent land in the classic half-solved plateau
    (loss ln(2)/2: two patterns confidently right, two at 0.5). Measured
    here across seeds 0-7: per-pattern lr=0.3 from seed 0 converges;
    that pinned combination is the package's signature test.
    """
    return MLP(hidden=(2,), act="tanh", classes=2, seed=seed,
               backend=backend).build(2)


def tabular_mlp(d: int, classes: int, hidden=(64, 32), seed: int = 0,
                backend: str = "mantissa") -> MLP:
    """The workhorse: d -> 64 -> 32 -> classes, relu hidden layers. No
    single citation because this *is* the generic multilayer perceptron
    (Rumelhart, Hinton & Williams, 1986); the default two-hidden-layer
    relu shape is the modern textbook baseline for tabular data (Goodfellow,
    Bengio & Courville, *Deep Learning*, 2016, ch. 6). Standardize the
    features first — see :func:`mantissa_mlp.tasks.standardize`."""
    return MLP(hidden=hidden, classes=int(classes), seed=seed,
               backend=backend).build(int(d))


def higgs_mlp(seed: int = 0, backend: str = "mantissa") -> MLP:
    """30 -> 300 -> 200 -> 100 -> 2, relu: a 600-hidden-unit MLP in the
    spirit of the HiggsML winners. The challenge's top entries were neural
    networks of a few hundred hidden units on the 30 features — Gábor
    Melis's winning entry ensembled 3-hidden-layer nets of 600 units each
    (Adam-Bourdarios, Cowan, Germain, Guyon, Kégl & Rousseau, 2015, "The
    Higgs boson machine learning challenge", *JMLR W&CP* 42; Melis's
    model is described in the same volume). Honest deviations, flagged:
    no ensemble, no local weight sharing, plain SGD instead of momentum +
    dropout — this is the architecture's spirit at this package's budget,
    not a reproduction of the winning pipeline.

    Feed it the higgsml features standardized with the -999.0 sentinel
    masked (``tasks.standardize(Xtr, Xte, missing=-999.0)``) and score
    with :func:`mantissa_mlp.tasks.ams`, the challenge's own metric.
    """
    return MLP(hidden=(300, 200, 100), classes=2, seed=seed,
               backend=backend).build(30)
