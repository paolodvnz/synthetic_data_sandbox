"""Hyperparameters of the generative core: losses, architecture, training and sampling."""

BETA = 1.0                  # weight of the KL term once warmed up: 1 is the true ELBO
KL_WARMUP_FRACTION = 0.1    # share of the training steps over which the KL weight grows from 0 to BETA
ACTIVE_DIM_KL = 0.01        # a latent dimension with a lower mean KL carries no information (collapse)

# Learned sigma of the Gaussian NLL ("mixed_sigma" loss). Numeric columns live in [0, 1]:
# without a floor, sigma -> 0 rewards a perfect copy of the rows and the loss goes to minus infinity.
MIN_SIGMA = 0.01
MAX_SIGMA = 1.0
INITIAL_SIGMA = 0.1         # starting sigma: between the bounds, near the typical early error (as TVAE)

# Architecture: sizes follow the width W of the encoded vector, within named bounds.
N_HIDDEN_LAYERS = 2         # dense layers in the encoder (the decoder mirrors them)
HIDDEN_PER_FEATURE = 2      # hidden width = 2 W ...
MIN_HIDDEN = 64             # ... but never narrower than this
MAX_HIDDEN = 512            # ... nor wider than this
FEATURES_PER_LATENT = 4     # latent size = W / 4, so the bottleneck stays narrower than the input ...
MIN_LATENT = 8
MAX_LATENT = 32

# Training length: set in epochs, so the KL warm-up and early stopping scale with the data (with fixed steps,
# 450 rows got ~1,000 epochs of warm-up and over-fitted before early stopping could act); capped in steps.
VALIDATION_FRACTION = 0.1   # cut from the train rows (the holdout is kept for the privacy check)
BATCH_SIZE = 256            # fixed: GPU memory does not grow with the number of rows
MAX_EPOCHS = 1_000          # capacity experiment: lowest over-fitting without losing fidelity against the holdout
MIN_STEPS = 200             # floor for tiny datasets, so the warm-up and the evaluations still have some steps
MAX_STEPS = 20_000          # cap: datasets above ~4,500 rows train as many steps as before
EVALUATIONS = 100           # evaluations on the validation rows per training (every max_steps // 100 steps)
PATIENCE = 10               # evaluations without improvement before early stopping
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5         # light L2 penalty, a mild brake on memorisation (as TVAE)

# Sampling: rows the coherence repair of decode could not fix (Adult: 0.1%) are replaced by new rows.
MAX_GENERATION_ROUNDS = 10    # rounds before keeping the incoherent rows left (the table keeps its size)
REGENERATION_OVERSAMPLE = 1.5 # rows drawn per missing row in a round, so one round is usually enough
