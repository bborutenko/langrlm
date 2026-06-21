from langrlm.core.repl import REPL
from langrlm.core.context_store import StringContextStore


def make_repl(context: str = "abcdefghij", llm_query_fn=None):
    if llm_query_fn is None:
        llm_query_fn = lambda prompt, text: f"answer({prompt})"
    return REPL(
        context_store=StringContextStore(context),
        llm_query_fn=llm_query_fn,
    )


def test_print_captured():
    repl = make_repl()
    output, done = repl.execute("print('hi')")
    assert output == "hi\n"
    assert done is False


def test_submit_returns_final():
    repl = make_repl()
    output, done = repl.execute("SUBMIT('the answer')")
    assert output == "the answer"
    assert done is True


def test_context_helpers_exposed():
    repl = make_repl("abcde")
    output, done = repl.execute("print(context_size(), context_slice(0, 2))")
    assert output == "5 ab\n"


def test_llm_query_exposed():
    repl = make_repl(llm_query_fn=lambda prompt, text: f"[{prompt}|{text}]")
    output, _ = repl.execute("print(llm_query('q', 'data'))")
    assert output == "[q|data]\n"


def test_error_is_returned_not_raised():
    repl = make_repl()
    output, done = repl.execute("1 / 0")
    assert "Error executing code" in output
    assert done is False


def test_namespace_persists_between_executes():
    repl = make_repl()
    repl.execute("x = 41")
    output, _ = repl.execute("print(x + 1)")
    assert output == "42\n"


class TestSandbox:
    def test_open_is_blocked(self):
        repl = make_repl()
        output, done = repl.execute("open('/etc/passwd')")
        assert "Error executing code" in output
        assert done is False

    def test_import_is_blocked(self):
        repl = make_repl()
        output, _ = repl.execute("import os")
        assert "Error executing code" in output

    def test_safe_builtins_available(self):
        repl = make_repl()
        output, _ = repl.execute("print(len('abc'), sorted([3, 1, 2]))")
        assert output == "3 [1, 2, 3]\n"
