# Tool calling with validation, timeouts, and bounded retries — example

Companion code for the article at `/blog/tool-calling-validation-timeouts-retries/`.

Python 3.11+, standard library only. No LLM is called: proposals are hand-written strings
shaped like a model's tool call (a name plus a JSON arguments string).

## Files

- `toolcall.py` — strict proposal validation, allowlist/authorization, per-attempt timeout and
  overall budget (`asyncio.timeout_at`), bounded retries for read-only tools, output validation,
  structured outcomes, and a per-request `TurnBudget`.
- `demo.py` — runs six canned proposals through the pipeline and prints log events and outcomes.
- `tests/test_toolcall.py` — `unittest` tests, including sanitized surrogate/nesting rejection through `handle_proposal`.
- `pyproject.toml` — Python version and dependency-free project configuration.
- `README.md` — these run instructions and limitations.

For the article's individual downloads, save `toolcall.py`, `demo.py`, `pyproject.toml`,
and `README.md` in one directory, and `test_toolcall.py` in its `tests/` subdirectory.
Run commands from that root directory. The commands below do not require
`tests/__init__.py`.

## Run

```bash
python -m unittest discover -s tests -v
python demo.py
```

or, with uv (used for verification in October 2026; not a statement about 2024 tooling):

```bash
uv run --python 3.11 --isolated python -m unittest discover -s tests -v
uv run --python 3.11 --isolated python demo.py
```

## Known limitations

- Application configuration is trusted: registry, authenticated scopes, schemas and defaults
  must be valid. Defaults must already satisfy their schemas because omitted defaults are not
  revalidated. Attempt timeout and overall budget must be finite positive numbers,
  `max_attempts` a positive integer, and `backoff_s` a nonempty sequence of finite nonnegative
  numbers. `TurnBudget` limits and counters must be valid finite nonnegative integers.
- A logging callback must be nonthrowing and nonblocking. Normal handled execution paths emit
  events; external cancellation and `TurnBudget` exhaustion can leave no event. The caller
  must log `LoopLimitExceeded` and `CancelledError`, and re-raise cancellation.
- Direct `execute()` calls require an already validated and authorized `ToolCall`.
  `handle_proposal()` performs validation and authorization before execution.

- `max_age_s` is validated but ignored; no freshness logic exists.
- Timeouts rely on cooperative asyncio cancellation; blocking or cancellation-swallowing handlers
  can exceed the budget. A timeout says nothing about remote side effects.
- No production write tool is implemented. Writes are never retried; only timeout/transient
  failures map to `unknown_effect`. Unexpected errors (`tool_error`) and invalid results
  (`invalid_output`) also do not prove that no effect occurred. A write adapter and caller
  must reconcile every ambiguous outcome; idempotency and reconciliation are described,
  not implemented.
- Timing tests use real short sleeps and may be flaky on a heavily loaded machine.
