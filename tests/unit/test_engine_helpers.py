from langrlm.core.engine import Engine


extract = Engine._extract_code
to_text = Engine._to_text


class TestExtractCode:
    def test_python_fence(self):
        assert extract("```python\nx = 1\n```") == "x = 1\n"

    def test_py_fence(self):
        assert extract("```py\nx = 1\n```") == "x = 1\n"

    def test_bare_fence(self):
        assert extract("```\nx = 1\n```") == "x = 1\n"

    def test_no_fence_returns_none(self):
        assert extract("just some prose") is None


class TestToText:
    def test_plain_string(self):
        assert to_text("hello") == "hello"

    def test_list_of_strings(self):
        assert to_text(["a", "b"]) == "ab"

    def test_list_of_blocks(self):
        assert to_text([{"text": "a"}, {"text": "b"}]) == "ab"

    def test_other_coerced_to_str(self):
        assert to_text(123) == "123"
