# Changelog

Release notes for **mantissa-mlp**, newest first — in the family's style
(see mantissa/RELEASES.md): what shipped, what was measured, what was
deliberately not done.

---

## v0.1.1 — 2026-07-16

The benchmark harness the frozen protocol was waiting for.
`bench/contenders.py` re-expresses the two zoo shapes layer-for-layer in
torch, tensorflow/keras and scikit-learn's `MLPClassifier` (a full
contender here — a real MLP, pinned to `solver="sgd"`, constant lr,
momentum 0), with a structural parity check; `bench/speed.py` runs
interleaved-repeat fit timing, test accuracy, higgsml AMS and
fresh-subprocess peak RSS under a `/tmp/mantissa-bench.lock` machine lock;
`bench/plots.py` draws the three figures. torch, tensorflow/keras,
scikit-learn and matplotlib added to the dev environment; covertype and
wine_quality downloaded — all six datasets now present.

**Measured** (Apple M4, CPU, medians over 5 interleaved repeats × 5
contenders × 6 datasets; 47 s timed region):
- **higgsml, the CERN centerpiece** (~90k-param `higgs_mlp`, 4000/2000):
  ours reaches **AMS 2.61** at the top-15%-by-P(signal) operating point
  against the select-everything baseline's 1.08 (selecting nothing scores
  0), test accuracy 0.773. The numpy backend is numerically identical
  (2.61 / 0.773); torch 2.59 / 0.772; tensorflow 2.13 / 0.772;
  scikit-learn 1.74 / 0.747. Accuracy and AMS disagree by design — the
  renormalized weights make the signal rare, so AMS, not accuracy, is the
  physics answer.
- **Memory is the clean win**: ours and the numpy backend peak at ~30–40 MB
  RSS across every dataset, vs scikit-learn ~93–104, torch ~242–250,
  tensorflow ~484–500 (import + one fit, fresh process).
- **Fit time splits by layer width**: ours is fastest on the four
  `tabular_mlp` sets (64→32 hidden), but slower than its *own* numpy
  backend on the two widest layers (higgsml 300-wide 0.219 vs 0.109 s;
  mnist_flat 784-wide 0.053 vs 0.034 s) — where fused Accelerate BLAS
  already saturates and the Session's per-primitive dispatch is pure
  overhead. An honest loss, left visible as an optimization target.
- Accuracy is a near-tie across the softmax family (ours ≡ numpy exactly);
  torch edges dimuon/banknote/mnist_flat, scikit-learn trails on the two
  binary sets. Structural parity holds exactly for the softmax family on
  all six datasets and for scikit-learn on the four multiclass sets.

**Deviations, recorded not tuned**: the frozen protocol's banknote
(1000/300) and wine_quality (2000/1000) budgets exceed a minority class,
so the harness clamped to the largest feasible equal-stratified sizes
(**916/300** and **2000/957**, seed unchanged) — `protocol.py` stays
frozen and both requested and actual sizes are in the results JSON.
scikit-learn draws its own initializer (no hook to match ours/torch/tf's
He/Glorot) and uses a single logistic output unit on binary problems
(`last_hidden+1` fewer parameters than the 2-logit softmax head); peak RSS
loads a cached standardized subset, not the 818k-row higgsml CSV, so it
measures the framework's footprint and not a 386 MB `np.loadtxt` transient.

**Deliberately not done**: no optimizer/momentum sweep, no per-framework
tuning, no GPU; the wide-layer dispatch overhead is measured and named, not
hidden.

---

## v0.1.0 — 2026-07-16

Initial release: multilayer perceptrons for tabular data (`MLP` with
softmax cross-entropy + mini-batch SGD, hidden relu/tanh Dense stacks and
an identity logits head) on top of mantissa-cnn's Dense layer, backends
and loss; a cited three-model zoo (`xor_net` — Rumelhart, Hinton &
Williams 1986, the family's Minsky-Papert resolution; `tabular_mlp`;
`higgs_mlp` — 600 hidden units in the spirit of the HiggsML winners,
Adam-Bourdarios et al. 2015); six tabular datasets with the
explicit-download discipline, two of them CERN-published (ATLAS higgsml,
record 328; CMS dimuon, record 545); and the AMS metric with
sentinel-aware, train-stats-only standardization.

**Measured** (M4, C engine, 42-test suite green in 0.23 s):
- **XOR, the signature number**: `xor_net` (12 parameters) reaches 4/4
  corners with the documented per-pattern recipe (500 epochs, lr 0.3,
  seed 0, numpy oracle backend), final loss < 0.05. Most (seed, lr)
  pairs under full-batch descent land in the classic ln(2)/2 plateau —
  measured across seeds 0–7 and recorded in the docstring rather than
  hidden.
- **higgsml verified after download**: 818,238 rows exactly (250,000
  KaggleSet t / 100,000 b / 450,000 v / 18,238 u), 279,560 signal vs
  538,678 background (34.2% unweighted — but renormalized weights sum to
  692 signal vs 411,000 background, which is why AMS, not accuracy, is
  the metric); 21.1% of feature entries are −999.0 sentinels; 65,630,848
  bytes; loader parses it in 1.6 s.
- **dimuon verified after download**: 26,911 of 100,000 events fall in
  the three resonance windows — 8,628 J/ψ, 12,159 Υ, 6,124 Z.
- **higgs_mlp 2-epoch smoke** (4000/2000 stratified subset, seed 0,
  batch 32, lr 0.01, standardized with the −999 mask): loss 0.604 →
  0.509, fit 0.10 s, test accuracy 0.762, **AMS 2.07** against the
  select-everything baseline's 1.08 (selecting nothing scores 0 by
  construction).
- **mnist_flat smoke** (2000/1000 via the sibling cnn checkout's data):
  5-epoch loss 2.17 → 1.19 monotone, test accuracy 0.702, fit 0.02
  s/epoch scale.

**Deliberately not done**: the benchmark harness (the protocol is frozen
in `bench/protocol.py` first, family rule — sklearn's MLPClassifier joins
as a full contender when it runs); no optimizer zoo, no early stopping,
no sparse inputs; covertype and wine_quality ship loaders + verified URLs
but were not downloaded in this cycle — their loaders raise the exact fix
command.
