"""Thresholds of the fidelity report (synthetic vs real distributions and relationships).

The baseline is measured on REAL_GROUPS disjoint groups of real people (the holdout + TRAIN_SLICES train slices,
the train is 4 x the holdout): mean and spread of the distance over every pair of groups.

A warning needs two things (calibrated on the six sample datasets x 3 seeds):
- the difference is **real**: z = (synthetic distance - mean) / spread above MIN_FIDELITY_Z. Real groups left out
  of the baseline: z about a standard normal (mean 0.07, deviation 1.17), 1 of 72 cases above 4 (113 rows per
  group); the weakest broken model: 4.46;
- the difference is **large**: synthetic distance - mean (the "excess") above a limit, because with 10,000 rows
  even a tiny difference is many spreads away. Default VAE, worst excess: numeric 0.023, associations 0.018;
  broken models (100 steps, or every column shuffled), smallest on the measure they break: 0.031, 0.023.
"""

FIDELITY_MAX_ROWS = 10_000     # rows per group at most (the holdout has at most MAX_HOLDOUT_ROWS = 10,000)
TRAIN_SLICES = 4               # train slices of the baseline: the train is 4 x the holdout (HOLDOUT_FRACTION 0.2)
REAL_GROUPS = 1 + TRAIN_SLICES # the holdout and the train slices: 10 pairs
SYNTHETIC_SLICES = 4           # synthetic slices compared with the holdout (the synthetic rows are as many as train)
HISTOGRAM_BINS = 20            # bins of a numeric or date histogram, on the train range
MIN_BASELINE = 1e-3            # floor of a spread, so z stays finite when the real groups agree
MIN_FIDELITY_Z = 4.0           # below: the difference may be chance
MAX_NUMERIC_EXCESS = 0.03      # mean W1 beyond chance, on the [0, 1] train range (3% of the range)
MAX_CATEGORICAL_EXCESS = 0.025 # mean total variation beyond chance (2.5 percentage points of the shares)
MAX_ASSOCIATION_EXCESS = 0.02  # mean association difference beyond chance
MAX_LISTED_COLUMNS = 3         # columns named in a warning
