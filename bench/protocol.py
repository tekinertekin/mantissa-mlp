"""Benchmark protocol constants — fixed BEFORE any benchmark code runs, so
the numbers cannot be tuned after the fact (the family rule: measure, don't
assume). The harness (speed/accuracy/plots, mantissa-cnn's bench layout) is
a later phase; nothing here executes a benchmark.

Contenders: ours (mantissa engine), ours (numpy backend), torch,
tensorflow, and scikit-learn's MLPClassifier — sklearn is a REAL MLP, so
unlike in the cnn repo (where it cannot express a convolution and was
removed) it is a full contender here. Same architecture re-expressed
layer-for-layer in each framework, identical hyperparameters (sklearn
pinned to solver="sgd", constant learning rate, momentum 0, matching
hidden_layer_sizes/batch/lr/epochs), CPU only.

Metrics per (dataset, contender): fit wall-time (median of interleaved
repeats), test accuracy, AMS for higgsml (the physics metric — weighted,
tasks.ams), and peak RSS in a fresh subprocess with import cost included.

Datasets take the family subset budgets: the two big sets get 4000/2000,
the rest 2000/1000, stratified, seed 0 (higgsml subsets carry renormalized
weights so the AMS column is comparable to full-set numbers). Features
standardized on train statistics only (missing=-999.0 for higgsml).
"""
SEED = 0
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.01
REPEATS = 5             # interleaved A/B/C/A/B/C..., median reported

# dataset -> (n_train, n_test) for mantissa_mlp.datasets.subset
SUBSETS = {
    "higgsml": (4000, 2000),
    "covertype": (4000, 2000),
    "dimuon": (2000, 1000),
    "mnist_flat": (2000, 1000),
    "wine_quality": (2000, 1000),
    "banknote": (1000, 300),     # the whole set is only 1372 rows
}

# dataset -> zoo model (mantissa_mlp.models); tabular_mlp gets (d, classes)
MODELS = {
    "higgsml": "higgs_mlp",
    "covertype": "tabular_mlp",
    "dimuon": "tabular_mlp",
    "mnist_flat": "tabular_mlp",
    "wine_quality": "tabular_mlp",
    "banknote": "tabular_mlp",
}

CONTENDERS = ("ours (mantissa)", "vanilla numpy", "torch", "tensorflow",
              "scikit-learn")
METRICS = ("fit_s", "test_acc", "ams", "peak_rss_mb")   # ams: higgsml only
