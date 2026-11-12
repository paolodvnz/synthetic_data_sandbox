"""Project-wide configuration: only what every block shares.

Block-specific settings live next to the block that owns them, in a ``*_config.py`` module
(e.g. ``sds.data_engine.thresholds_config``). This module imports nothing heavy, so that any block
can read the seed without loading PyTorch or the environment file.
"""

from pathlib import Path

RANDOM_STATE = 42

# The folder holding pyproject.toml, api/ and ui/: found from this file's position, which is reliable because the
# app is installed in editable mode (pip install -e .), the only supported way. src/sds/config.py -> sds, src, root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Largest CSV accepted by the app (API and UI): the practical limit that replaced a row cap (no sampling of
# the train rows, so rare cases are kept). About 5.4 GB of RAM at the peak for a 200 MB file.
MAX_UPLOAD_MB = 200

# Closing the app from the page (the "Close" button). sds-app tells the API it may close (CLOSABLE_ENV = "1");
# the API answers, then ends with CLOSED_BY_USER_EXIT_CODE, which sds-app reads as a wanted stop: it stops the UI
# and exits with 0. An exit code, not a signal, so it works the same on every system (SIGTERM on Windows ends a
# process without its cleanup). Not one of the usual codes: 0, 1, 2, 3 (uvicorn startup failure), 126 and above.
CLOSABLE_ENV = "SDS_CLOSABLE"
CLOSED_BY_USER_EXIT_CODE = 86
