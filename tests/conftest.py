import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langrlm.core.context_store import FileContextStore


def code_block(code: str) -> str:
    """Wrap code in a python markdown fence, as the model would emit it."""
    return f"```python\n{code}\n```"


@pytest.fixture
def make_code_model():
    """Build a fake model that emits the given code snippets, one per turn."""
    def _make(*code_snippets: str) -> FakeListChatModel:
        return FakeListChatModel(responses=[code_block(c) for c in code_snippets])
    return _make


@pytest.fixture
def make_text_model():
    """Build a fake model that replies with the given plain-text strings."""
    def _make(*texts: str) -> FakeListChatModel:
        return FakeListChatModel(responses=list(texts))
    return _make


@pytest.fixture
def make_file_context(tmp_path):
    """Build a FileContextStore over a temp file with the given contents."""
    def _make(text: str) -> FileContextStore:
        path = tmp_path / "ctx.txt"
        path.write_text(text)
        return FileContextStore(str(path))
    return _make
