"""Configuration of the Streamlit UI."""

API_URL_ENV = "SDS_API_URL"                 # set by sds-app
DEFAULT_API_URL = "http://127.0.0.1:8000"   # when the UI is started by hand (streamlit run ui/app.py)
REQUEST_SECONDS = 10                        # timeout of a quick API call
APP_TITLE = "Synthetic Data Sandbox"
CLOSED_MESSAGE = ("**The app is closed.** The data in memory has been discarded. You can close this tab; "
                  "to use the app again, start it with sds-app.")
API_DOWN_MESSAGE = "The app's engine is not running. Close this page and start the app again with sds-app."
STOP_REQUEST_SECONDS = 330                  # /reset and /train/stop wait for the job: up to 300 s in the API
WORK_REQUEST_SECONDS = 300                  # /upload, /choices, samples: reading a 200 MB file takes ~1 min
UPLOAD_TYPES = ["csv", "txt"]
POLL_SECONDS = 1                            # how often the progress bar asks /status
