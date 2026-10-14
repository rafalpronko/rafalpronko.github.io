# Model routing demo: allowlist, capability guards, staged commit and rollback

Companion code for the article "Model routing with Jev: allowlists, capability guards, and rollback".

This is a **synthetic, serial, single-threaded simulation** using only the Python standard library.
It does not call Jev or any other model API, it has no network code, and it does not tokenize text.
The classifier proposal is a hand-written fixture shaped like a choice result (a label and a
confidence). Route aliases (`fast-text`, `mid-tools`, `full-tools`), adapters and model ids are
fictional. Token counts are configured numbers, not tokenizer output and not a proof that a real
request fits a real context window.

The router is a **switch-admission** check only. The catalog, policy and turn token counts are
trusted host facts; proposal fields are assumed to be ordinary parsed values. A candidate that is
already the live route is kept without a size check, so every outbound request still needs its own
validation. `persisted=False` means the write was not acknowledged, not that nothing was written.

## Files

- `routing.py` — the router: policy, confidence and alternatives-container validation, allowlist, capability guards,
  copy-on-write staging, full-request size check, commit point, best-effort persistence.
- `catalog.py` — the fictional host catalog.
- `demo.py` — makes seven routing calls across six turn ids and prints each decision.
- `tests/test_routing.py` — pytest suite.

## Run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -v
uv run python demo.py
```

Without uv: `python -m venv .venv && . .venv/bin/activate && pip install pytest && pytest -v`.

## Deliberately left out

Concurrency and locking, real tokenizers, network adapters and credentials, cooldowns,
telemetry, configuration loading, and releasing real resources on rollback. In this demo a
rollback only discards a staged copy; a real host must also release whatever staging acquired.
