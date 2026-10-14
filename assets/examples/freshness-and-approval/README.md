# Freshness and approval: example

Companion code for the article "Freshness and approval: why a valid command can become unsafe".

This is a synthetic teaching example. The target, the approval ledger and the
dispatcher live in one process's memory and are serialized with in-process
locks. It is not a production design and it does not talk to any real device,
service or database.

## Files

- `freshness_gate.py` — the pure gate: `Policy`, `Command`, `Snapshot`,
  `Approval`, `binding_digest()`, `command_fingerprint()`, `issue_approval()`
  and `evaluate()`.
  No clock, no I/O; every decision is a function of its arguments.
- `fake_target.py` — a fake target with a conditional write and idempotency
  keys, an approval ledger with revoke and an atomic authorize-and-consume,
  test clocks and a dispatcher that ties them together.
- `tests/test_gate.py` — the gate's decision table.
- `tests/test_dispatch.py` — scenarios: human delay, TOCTOU, two dispatcher
  objects in one process standing in for separate actors, restart, target
  routing, policy changes, the authorization cutoff (expiry, stale state and
  revocation just before it; the clock read under the ledger lock), lost
  response, idempotency key reuse.

## Running the tests

Python 3.12, standard library only; pytest for the tests. These commands use
current tooling (uv and pytest 9.1.1) and were last verified on 2026-10-13.
They are not a record of the tools available on the article's editorial date
(22 August 2024). Run from this directory:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python 'pytest==9.1.1'
.venv/bin/python -m pytest
```

The numbers in the tests (5 s state age, 60 s approval TTL) are illustrative.
They are not recommended values for any real system.

Known limits, on purpose: everything runs in one process and in memory;
`FakeTarget.read()` is assumed to be a synchronous, authoritative read;
the authorization cutoff is the clock reading taken inside
`ApprovalLedger.authorize_and_consume()`, where the whole gate runs and the
approval is spent in one step; neither revocation nor expiry after that reading
stops the write.
