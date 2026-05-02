# Claude 4.7 model patch

Minimal intended changes in `app.py`:

```python
LLM_MODEL_DEFAULT = sget("LLM_MODEL", "claude-sonnet-4-7")
```

```python
claude_candidates = [
    "claude-sonnet-4-7",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]
```

No changes are required in `build_llm()` because it already passes `model=model_name` into `ChatAnthropic`.
