from enum import Enum

class ContextStoreType(str, Enum):
    """Supported context store backends.

    Passed to :func:`~langrlm.core.context_store.create_context_store` to select
    where the agent's context comes from.

    Members:
        FILE: Load the context from a local file (requires a ``path`` argument).
    """

    FILE = "file"

