"""Thresholds of the privacy check (exact copies + DCR)."""

DCR_PERCENTILE = 5             # low tail of the distances: where copied or near-copied rows show up
MIN_DCR_RATIO = 0.8            # synthetic percentile must reach this share of the holdout one (a perfect
                               # synthetic sample of 1,000 rows gives 0.83 to 1.12; confirmed on 6 datasets)
EXACT_DISTANCE = 1e-6          # a Gower distance at or below this counts as an exact copy
EXACT_SHARE_TOLERANCE = 0.01   # copies allowed beyond the genuine duplicates of the real data
NOISE_Z = 3.0                  # standard errors of sampling noise tolerated on the share of copies
DCR_BATCH_SIZE = 256           # query rows per block: memory is batch_size x reference block
DCR_REFERENCE_BLOCK = 8192     # reference rows per block, so memory does not grow with the rows
DCR_MAX_ROWS = 10_000          # synthetic rows compared at most (a seeded random subset beyond)
