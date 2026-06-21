from langrlm.core.engine import Engine
from langrlm.core.context_store import StringContextStore


def test_recursive_sub_call_nests(make_code_model, make_text_model):
    # Root slices the context and queries the sub-model; with max_depth=2 the
    # query runs a nested Engine, whose model answers in plain text.
    root = make_code_model('r = llm_query("echo", context_slice(0, 4))\nSUBMIT(r)')
    sub = make_text_model("ECHOED")
    engine = Engine(
        rlm_model=root,
        sub_model=sub,
        context=StringContextStore("abcdefghij"),
        max_depth=2,
    )
    assert engine.complete("go") == "ECHOED"


def test_flat_base_case(make_code_model, make_text_model):
    # With max_depth=1, llm_query falls back to a flat sub-model call.
    root = make_code_model('SUBMIT(llm_query("echo", context_slice(0, 4)))')
    sub = make_text_model("FLAT")
    engine = Engine(
        rlm_model=root,
        sub_model=sub,
        context=StringContextStore("abcdefghij"),
        max_depth=1,
    )
    assert engine.complete("go") == "FLAT"
