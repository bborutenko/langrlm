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


class StringContextStore(BaseContextStore):
    """Context store backed by an in-memory string.

    Used for nested/recursive RLM calls, where a sub-task's context is text
    that already lives in memory rather than a file on disk.
    """

    context: None | str

    def __init__(self, text: str):
        """Create the store.

        Args:
            text: The context contents. Available immediately; ``load()`` is a
                no-op.
        """
        self.context = text

    def load(self) -> None:
        return

    @_check_context_is_not_none
    def slice(self, start: int, end: int) -> str:
        return self.context[start:end]

    @_check_context_is_not_none
    def size(self) -> int:
        return len(self.context)


def to_string_context_store(
        store: BaseContextStore, start: int, end: int
    ) -> StringContextStore:
    """Convert any context store into a StringContextStore over a slice.

    Used when spawning a sub-call: the parent's context is sliced down to the
    chunk the sub-call should work on, and that chunk is wrapped back into a
    :class:`StringContextStore` so the nested engine gets the same interface.
    Works uniformly for any store type — if ``store`` is already a
    :class:`StringContextStore`, this simply returns a new one holding the
    requested slice.

    The ``store`` must already be loaded (``store.load()`` called); this
    function does not load it. In normal use the parent store is loaded by the
    engine before any sub-call is spawned.

    Args:
        store: The context store to convert (any :class:`BaseContextStore`),
            already loaded.
        start: Start index of the slice.
        end: End index of the slice.

    Returns:
        A new :class:`StringContextStore` holding ``store[start:end]``.

    Raises:
        ValueError: If ``store`` has not been loaded yet.
    """
    chunk = store.slice(start, end)
    return StringContextStore(chunk)


def create_context_store(
        context_store_type: ContextStoreType, **kwargs
    ) -> BaseContextStore:
    """Factory that builds a context store of the requested type.

    Selects the concrete :class:`BaseContextStore` implementation for the given
    ``context_store_type`` and constructs it with the extra keyword arguments.

    Args:
        context_store_type: Which kind of store to create. See
            :class:`~langrlm.utils.enums.ContextStoreType`.
        **kwargs: Constructor arguments forwarded to the chosen store.
            ``ContextStoreType.FILE`` takes ``path`` — the file whose contents
            become the context; ``ContextStoreType.STRING`` takes ``text`` — the
            context contents directly.

    Returns:
        A ready-to-use :class:`BaseContextStore` instance.

    Raises:
        ValueError: If ``context_store_type`` is not supported.
    """
    if context_store_type == ContextStoreType.FILE:
        return FileContextStore(**kwargs)
    elif context_store_type == ContextStoreType.STRING:
        return StringContextStore(**kwargs)
    else:
        raise ValueError(f"Unsupported context store type: {context_store_type}")