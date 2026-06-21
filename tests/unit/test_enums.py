from langrlm.utils.enums import ContextStoreType


def test_is_str_enum():
    assert ContextStoreType.FILE == "file"
    assert ContextStoreType.STRING == "string"


def test_members():
    assert {m.value for m in ContextStoreType} == {"file", "string"}
