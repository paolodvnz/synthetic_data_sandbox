"""Keys of ``st.session_state`` shared by the page and the steps (what the UI remembers between reruns)."""

STEP = "step"                             # the step being looked at
UPLOADER_GENERATION = "uploader_generation"   # part of the file uploader's key: a new value empties it
DEV_PANELS = "dev_panels"                 # the "Show developer panels" toggle
CONFIRMING_START_OVER = "confirming_start_over"
CONFIRMING_CLOSE = "confirming_close"
CLOSED = "closed"                         # the app was closed: only the goodbye message is left
CLOSE_REQUESTED = "close_requested"       # the API was asked to close (once)
CLOSE_ERROR = "close_error"               # why the API refused to close, shown after the rerun
DATASET = "dataset"                       # the last dataset view received from the API
EDITOR_TOKEN = "editor_token"             # part of the keys of the table and of Advanced: a new value resets them
UPLOAD_ERROR = "upload_error"             # the message of a refused file, shown after the rerun
