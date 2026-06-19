import functools
from abc import ABC, abstractmethod

from langrlm.utils.enums import ContextStoreType

def _check_context_is_not_none(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.context is None:
            raise ValueError("Context is not loaded. Please load the context before invoking this method.")
        return func(self, *args, **kwargs)
    return wrapper

class BaseContextStore(ABC):
    context: None | str

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def slice(self, start: int, end: int) -> str:
        pass

    @abstractmethod
    def size(self) -> int:
        pass


class FileContextStore(BaseContextStore):
    context: None | str
    path: str

    def __init__(self, path: str):
        self.context = None
        self.path = path

    def load(self) -> None:
        if self.context is not None:
            return
        try:
            with open(self.path, "r") as f:
                self.context = f.read()
        except Exception as e:
            raise ValueError(f"Failed to load context from file: {e}")

    @_check_context_is_not_none
    def slice(self, start: int, end: int) -> str:
        return self.context[start:end]

    @_check_context_is_not_none
    def size(self) -> int:
        return len(self.context)


def create_context_store(
        context_store_type: ContextStoreType, **kwargs
    ) -> BaseContextStore:
    if context_store_type == ContextStoreType.FILE:
        return FileContextStore(**kwargs)
    else:
        raise ValueError(f"Unsupported context store type: {context_store_type}")