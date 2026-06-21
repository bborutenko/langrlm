import pytest

from langrlm.core.context_store import (
    FileContextStore,
    StringContextStore,
    to_string_context_store,
    create_context_store,
)
from langrlm.utils.enums import ContextStoreType


class TestFileContextStore:
    def test_load_slice_size(self, make_file_context):
        store = make_file_context("hello world")
        store.load()
        assert store.size() == 11
        assert store.slice(0, 5) == "hello"

    def test_slice_before_load_raises(self, make_file_context):
        store = make_file_context("data")
        with pytest.raises(ValueError):
            store.slice(0, 1)

    def test_load_is_idempotent(self, make_file_context):
        store = make_file_context("abc")
        store.load()
        store.context = "changed"  # second load must not overwrite
        store.load()
        assert store.context == "changed"

    def test_load_missing_file_raises(self):
        with pytest.raises(ValueError):
            FileContextStore("/no/such/file.txt").load()


class TestStringContextStore:
    def test_available_immediately(self):
        store = StringContextStore("abcde")
        store.load()  # no-op
        assert store.size() == 5
        assert store.slice(1, 3) == "bc"


class TestToStringContextStore:
    def test_from_file_store(self, make_file_context):
        store = make_file_context("abcdefghij")
        store.load()
        sliced = to_string_context_store(store, 2, 6)
        assert isinstance(sliced, StringContextStore)
        assert sliced.slice(0, sliced.size()) == "cdef"

    def test_from_string_store(self):
        base = StringContextStore("hello world")
        sliced = to_string_context_store(base, 0, 5)
        assert sliced.slice(0, sliced.size()) == "hello"

    def test_requires_loaded_store(self, make_file_context):
        store = make_file_context("data")  # not loaded
        with pytest.raises(ValueError):
            to_string_context_store(store, 0, 1)


class TestCreateContextStore:
    def test_file(self, tmp_path):
        path = tmp_path / "c.txt"
        path.write_text("x")
        store = create_context_store(ContextStoreType.FILE, path=str(path))
        assert isinstance(store, FileContextStore)

    def test_string(self):
        store = create_context_store(ContextStoreType.STRING, text="x")
        assert isinstance(store, StringContextStore)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            create_context_store("nope")  # type: ignore[arg-type]
