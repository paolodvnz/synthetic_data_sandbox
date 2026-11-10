"""Import hygiene: light blocks must not load heavy dependencies or read secrets.

Each check runs in a fresh interpreter, because ``sys.modules`` of the test process already holds
whatever other tests imported.
"""

import subprocess
import sys

import pytest

CHECK = (
    "import os, sys\n"
    "os.environ.pop('OPENAI_API_KEY', None)\n"
    "import {module}\n"
    "assert 'torch' not in sys.modules, 'torch was imported'\n"
    "assert 'dotenv' not in sys.modules, 'dotenv was imported'\n"
    "assert 'OPENAI_API_KEY' not in os.environ, 'the environment file was read'\n"
)


@pytest.mark.parametrize("module", ["sds", "sds.config", "sds.data_engine", "sds.samples"])
def test_import_loads_neither_torch_nor_secrets(module):
    result = subprocess.run(
        [sys.executable, "-c", CHECK.format(module=module)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
