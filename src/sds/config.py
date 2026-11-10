"""Project-wide configuration: only what every block shares.

Block-specific settings live next to the block that owns them, in a ``*_config.py`` module
(e.g. ``sds.data_engine.thresholds_config``). This module imports nothing heavy, so that any block
can read the seed without loading PyTorch or the environment file.
"""

RANDOM_STATE = 42
