"""A scripted stand-in for the provider client: no test calls a real model."""

import threading
from types import SimpleNamespace


def reply(content, finish_reason="stop", prompt_tokens=100, completion_tokens=50):
    """A response shaped like the one of the openai library."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeClient:
    """Gives the scripted replies in order (an exception is raised) and records every request.

    With ``hold=True`` every call waits for ``release()``; ``entered`` is set as soon as a call arrives.
    """

    def __init__(self, *script, hold: bool = False):
        self.script = list(script)
        self.requests = []
        self.entered = threading.Event()
        self.released = threading.Event()
        if not hold:
            self.released.set()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def release(self):
        self.released.set()

    def create(self, **request):
        self.requests.append(request)
        self.entered.set()
        self.released.wait()
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return reply(item) if isinstance(item, str) else item
