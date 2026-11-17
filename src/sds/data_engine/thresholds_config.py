"""Configuration of the Data Engine: the thresholds of the schema contract.

Calibrated on the six benchmark datasets of ``sds.samples`` (``sds-download-samples``).
"""

# Type inference
MAX_DISCRETE_LEVELS = 20      # integer column with at most this many values -> categorical
MAX_CATEGORIES = 50           # text column with more levels -> dropped (IDs, free text, codes)
YES_NO_LEVELS = 2             # integer column with at most this many values: codes and quantities
                              # decode to the same column, so the reading cannot be changed
ID_UNIQUENESS_RATIO = 0.95    # integer/text column at least this unique -> suspected ID
MAX_NULL_FRACTION = 0.30      # column with more nulls -> dropped (else dropna loses too many rows)
NUMERIC_MAJORITY = 0.95       # share of parseable values for a "mostly numeric" column
MAX_NULL_MARKERS = 3          # distinct non-numeric words still read as missing-value markers
                              # ("n.d.", "-"); more -> they are labels (e.g. ICD-9 "V57", "E878")

# Rare values
MIN_LEVEL_ROWS = 10           # fewer rows: a level is merged into "Other", a near-constant column
                              # (rows differing from the most frequent value) is proposed for removal

# Point masses of the numeric columns (train rows; measured on the six samples: columns without a mass have a
# ratio of 1.0-1.4, columns with one at least 3.4)
MIN_MASS_SHARE = 0.05          # the most frequent value covers at least this share of the rows...
MIN_MASS_RATIO = 3             # ...and is at least this many times as frequent as the second value

# Preparation
HOLDOUT_FRACTION = 0.20        # rows kept apart for the privacy check (DCR), never trained on
MAX_HOLDOUT_ROWS = 10_000      # a few thousand reference rows are enough for the DCR
MIN_ROWS = 1_000               # fewer training rows -> reliability warning (455-614 rows still pass the
                               # privacy gate, with more over-fitting and noise: a warning, not a block)
MIN_USABLE_ROWS = 50           # fewer rows after removing missing values -> error
HIGH_ROW_LOSS_FRACTION = 0.50  # more rows removed -> stronger warning naming the columns
MAX_LISTED_COLUMNS = 3         # columns named as responsible for the rows removed

# Decoding
MAX_DECIMALS = 10              # decimals tried for the values of a numeric column; more (computed values such
                               # as 1/3) -> no rounding of the synthetic numbers
DECIMALS_TOLERANCE = 1e-12     # relative difference between a value and its rounding still read as float noise
                               # (relative: an absolute one would read 1e-10 as 0 with no decimals)
MAX_REPAIR_PASSES = 5          # passes of the coherence repair over the rows with an unseen pair of
                               # categorical values (Adult: 25% of the rows before, 0.1% left after)

# Memory
ENCODE_BLOCK_ROWS = 100_000    # rows encoded at a time: transform needs the float32 output + one block

# File reading
MAX_BAD_LINES_FRACTION = 0.05  # more malformed lines -> the file is probably misread
SNIFF_BYTES = 64 * 1024        # sample used to detect the separator
BINARY_PROBE_BYTES = 8 * 1024  # a NUL byte in this prefix -> not a text file
CANDIDATE_SEPARATORS = ",;\t|"

# Dates: ISO, or day/month/year and month/day/year with one of three separators and a four-digit year. The shape
# (DATE_SHAPE) says a value looks like such a date; the formats decide which one it is.
ISO_DATE_FORMAT = "%Y-%m-%d"
DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y")
DATE_SHAPE = r"\d{1,2}([/.-])\d{1,2}\1\d{4}"   # the same separator twice

# Compared lower-case after stripping spaces.
NULL_TOKENS: frozenset[str] = frozenset({"", "?", "na", "n/a", "null", "nan"})

# Messages
MAX_LISTED_VALUES = 5          # values quoted in a warning
