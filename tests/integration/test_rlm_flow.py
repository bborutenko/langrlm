from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langrlm import RLM
from langrlm.core.context_store import StringContextStore


def test_submit_returns_answer(make_code_model):
    model = make_code_model('SUBMIT("size: " + str(context_size()))')
    rlm = RLM(model, context=StringContextStore("abcde"))
    assert rlm.invoke("go") == "size: 5"


def test_plain_text_answer_is_final(make_text_model):
    model = make_text_model("just an answer")
    rlm = RLM(model, context=StringContextStore("x"))
    assert rlm.invoke("go") == "just an answer"


def test_multi_step_then_submit(make_code_model):
    # First step prints (no SUBMIT), second step submits.
    model = make_code_model("print('looking')", 'SUBMIT("found")')
    rlm = RLM(model, context=StringContextStore("data"), max_depth=5)
    assert rlm.invoke("go") == "found"


def test_forced_answer_when_depth_exhausted():
    # Two code steps without SUBMIT, then the forced-answer turn returns text.
    model = FakeListChatModel(responses=[
        "```python\nprint(1)\n```",
        "```python\nprint(2)\n```",
        "FORCED FINAL",
    ])
    rlm = RLM(model, context=StringContextStore("ctx"), max_depth=2)
    assert rlm.invoke("go") == "FORCED FINAL"


def test_auto_loads_file_context(make_file_context, make_code_model):
    store = make_file_context("abcde")  # not manually loaded
    model = make_code_model('SUBMIT(str(context_size()))')
    rlm = RLM(model, context=store)
    assert rlm.invoke("go") == "5"


def test_state_reset_between_invokes(make_code_model):
    model = make_code_model('SUBMIT("first")', 'SUBMIT("second")')
    rlm = RLM(model, context=StringContextStore("x"))
    assert rlm.invoke("a") == "first"
    assert rlm.invoke("b") == "second"  # not the stale "first"


class TestStructuredDetection:
    def test_fake_model_has_no_structured_output(self, make_code_model):
        rlm = RLM(make_code_model('SUBMIT("x")'), context=StringContextStore("a"))
        assert rlm._has_structured_output is False
        assert rlm.code_schema is None
        assert rlm.sub_schema is None
