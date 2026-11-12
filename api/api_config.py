"""Configuration of the API."""

from sds.config import MAX_UPLOAD_MB

# Developer mode: read once, when the app is created. Off unless the variable holds one of these words.
DEV_MODE_ENV = "SDS_DEV_MODE"
TRUE_WORDS = frozenset({"1", "true", "yes", "on"})

# Upload: the CSV arrives as the raw request body, read in chunks into memory (multipart would spool it to disk).
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
DEFAULT_SOURCE_NAME = "dataset.csv"
MAX_SOURCE_NAME_CHARS = 255         # longer names keep their end (where the extension is)
N_EXAMPLES = 3                      # distinct values shown per column (values, never rows)

# Training job: /train/stop waits for the thread at most this long (the privacy check cannot be interrupted and
# takes up to ~2 min on 200 MB); past it the job still says "running" with stop_requested and the UI keeps polling.
STOP_TIMEOUT_SECONDS = 300

# Validation results and download.
PREVIEW_ROWS = 10                   # synthetic rows shown in the Validate step (only once the gate has passed)
SYNTHETIC_SUFFIX = "_synthetic.csv"
DEFAULT_DOWNLOAD_STEM = "dataset"   # for an uploaded name with nothing before the extension
# What to do in the app when the privacy gate fails: the engine's message gives the cause, this the remedy.
# Training is deterministic and has no user levers, so only the data can change the outcome.
GATE_ADVICE = ("To fix it, go back to step 1 (Upload) and untick the columns that identify a single person "
               "(codes, exact dates, rare values), or upload a file with more rows; then train again. "
               "Training again without changing anything gives the same result.")

# Developer views (/dev/*).
DECODING_CHECK_ROWS = 10_000        # train rows encoded and decoded back at most (a seeded sample beyond)
DECODING_CHECK_TOLERANCE = 1e-6     # share of the train range within which a decoded number counts as identical
N_RANDOM_ROWS = 5                   # rows decoded from random vectors: what a model that learned nothing makes

# HTTP statuses of the error format {"code", "message"}.
STATUS_FORBIDDEN = 403              # the privacy gate did not pass
STATUS_NOT_FOUND = 404
STATUS_CONFLICT = 409               # the request is fine but the session is not at the right step
STATUS_TOO_LARGE = 413              # the upload is over MAX_UPLOAD_BYTES
STATUS_UNPROCESSABLE = 422          # the data cannot be used (DataEngineError, VaeError)
STATUS_INTERNAL = 500               # a bug: the details stay in the log

# /shutdown without sds-app: nobody would stop the UI, so the API does not end itself.
NOT_CLOSABLE_MESSAGE = ("The app was not started with sds-app, so it cannot be closed from the page: "
                        "stop it from the terminal (Ctrl+C).")

INTERNAL_ERROR_MESSAGE = "Something went wrong on our side. Please try again, or start over with Reset."
