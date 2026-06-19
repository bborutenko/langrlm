# langrlm

**Recursive Language Models (RLM) built on top of LangChain.**

`langrlm` is an open-source library that turns any LangChain chat model into a
**Recursive Language Model (RLM)** agent. You bring your own model — created the
usual LangChain way — and pass it into `langrlm`. The library wraps that model
and drives it as an RLM, letting the model recursively decompose a problem,
call itself (or sub-models) on smaller sub-problems, and combine the
intermediate results into a final answer.

## Idea

The core idea is simple: **the model is a parameter.**

```python
from langchain_openai import ChatOpenAI
from langrlm import RLM, create_context_store, ContextStoreType

# 1. Build a model the normal LangChain way
model = ChatOpenAI(model="gpt-4o")

# 2. Point langrlm at the large context the model should reason over
context = create_context_store(ContextStoreType.FILE, path="big_document.txt")

# 3. Hand the model + context to langrlm, which uses them as an RLM agent
rlm = RLM(model, context=context)

answer = rlm.invoke("Summarize every dollar amount mentioned in the document")
```

By default the same `model` is reused for the recursive sub-calls. You can pass
a cheaper/faster `sub_model` and cap the recursion with `max_depth`:

```python
rlm = RLM(model, context=context, sub_model=cheap_model, max_depth=8)
```

Because the input is a standard LangChain model, `langrlm` works with **any
provider LangChain supports** (OpenAI, Anthropic, local models, etc.) without
being tied to a specific API.

## What is an RLM?

A Recursive Language Model treats the language model not as a single
question-in / answer-out call, but as a recursive reasoning engine:

- **Decompose** — break a large or complex task into smaller sub-tasks.
- **Recurse** — invoke the model on each sub-task (potentially several levels
  deep).
- **Aggregate** — merge the sub-answers back up into a coherent final result.

This recursion lets models tackle problems that exceed a single context window
or a single reasoning step.

## Goals

- Provide a thin, model-agnostic layer between LangChain and RLM-style inference.
- Make recursive reasoning a drop-in capability for existing LangChain models.
- Stay open and extensible so users can customize decomposition, recursion
  depth, and aggregation strategies.

> ⚠️ Early stage — the API is still evolving.
