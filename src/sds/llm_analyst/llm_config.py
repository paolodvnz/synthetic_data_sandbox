"""Limits of the calls to the LLM provider."""

ANSWER_TOKENS = 8000           # max_completion_tokens: reasoning models spend part of it before answering;
                               # the longest of 40 test answers (reasoning included) used 885. It is a ceiling, not
                               # a cost (only used tokens count, also against Groq's 8,000 per minute): the room is
                               # for models that reason longer
REQUEST_TIMEOUT_SECONDS = 120  # one call took about 1.5 s on Groq; room for 8,000 tokens on a slower provider
MAX_HISTORY_TURNS = 3          # question/answer pairs sent back: follow-ups understood 4/4 in the tests,
                               # +145...+260 prompt tokens per pair
MAX_QUESTION_CHARS = 500       # a question from a person, not a document: longer text is refused before the call
MAX_ERROR_CHARS = 200          # error sent back with the fix request: type and message are enough to fix the
                               # code (3/3 provoked fixes in the tests), a long message only costs tokens
