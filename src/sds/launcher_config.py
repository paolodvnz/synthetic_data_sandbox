"""Configuration of ``sds-app``: where the API and the UI listen and how long the launcher waits for them."""

API_HOST = "127.0.0.1"          # local only (privacy model): never 0.0.0.0
UI_HOST = "localhost"
DEFAULT_API_PORT = 8000
DEFAULT_UI_PORT = 8501
MAX_PORT = 65_535

API_SCRIPT = "api.main:app"     # run from the project root, like the tests
UI_SCRIPT = "ui/app.py"
API_HEALTH_PATH = "/health"
UI_HEALTH_PATH = "/_stcore/health"   # Streamlit's own liveness endpoint
API_URL_ENV = "SDS_API_URL"     # read by the UI to find the API

API_START_SECONDS = 120         # the first import of PyTorch takes a few seconds, more on a cold disk
UI_START_SECONDS = 60
STOP_SECONDS = 10               # after terminate, then kill
POLL_SECONDS = 0.2
HEALTH_REQUEST_SECONDS = 2      # timeout of one health request
