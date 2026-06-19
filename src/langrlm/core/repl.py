import io
import contextlib
from typing import Callable

from langrlm.core.context_store import BaseContextStore
from langrlm.utils.builtins_set import SAFE_BUILTINS


class REPL:
    _namespace: dict
    _final_answer: None | str

    def __init__(
            self,
            context_store: BaseContextStore,
            llm_query_fn: Callable,
        ):
        self._namespace = {
            "__builtins__": SAFE_BUILTINS,

            "context_slice": context_store.slice,
            "context_size": context_store.size,

            "llm_query": llm_query_fn,
            "SUBMIT": self._submit,
        }
        self._final_answer = None

    def _submit(self, answer: str) -> None:
        self._final_answer = answer

    def execute(self, code: str) -> tuple[str, bool]:
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, self._namespace)
        except Exception as e:
            return f"Error executing code: {e}", False

        output = stdout.getvalue()

        if self._final_answer is not None:
            return self._final_answer, True

        return output or "", False
