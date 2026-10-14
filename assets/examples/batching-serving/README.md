# Batching and serving simulator

Companion code for "Batching and serving: measure throughput, latency, and failures under load".

A deterministic, dependency-free discrete-event simulation of one worker with dynamic batching. Time is a virtual clock in integer microseconds, and service time is a formula (`overhead_us + per_item_us * batch_size`). Nothing in it measures an endpoint, a GPU, or a framework, so its output describes the algorithm under the chosen parameters, not anyone's capacity.

## Run

Place the four downloaded files (`simulate.py`, `test_simulate.py`, `pyproject.toml`, and `README.md`) together in one directory, then run these commands from that directory:

```sh
uv venv --python 3.12 .venv
.venv/bin/python simulate.py            # table
.venv/bin/python simulate.py --json     # same rows as JSON
.venv/bin/python -m unittest -v
```

No packages are installed. The virtual environment only pins the interpreter. Any Python 3.12 works without uv too, for example `python3.12 simulate.py`.

## Model

- Open-loop arrivals: request `i` arrives at `i * 1_000_000 // rate_rps` µs during a 10 s injection window, independent of completions.
- Bounded waiting queue (`queue_capacity`, default 64). An arrival that finds it full is rejected and never gets a latency.
- Single worker. A batch leaves when `max_batch_size` requests are waiting or the oldest has waited `max_wait_us`, but never before the worker is free. A dispatch due at time t happens before an arrival at time t.
- Client deadline (`deadline_us`, default 100 ms). A request finishing later counts as `deadline_missed`. The worker still completes it, and its latency stays in the percentiles.
- After the injection window, the queue drains, so every admitted request completes.

## Output columns

- `offered`: scheduled arrivals. `rejected`: arrivals that found the queue full. `admitted = offered - rejected`.
- `completed = admitted` (because of the drain), split into `within_deadline` and `deadline_missed`.
- `completed_throughput_rps` and `goodput_rps` divide by the observation window: the longer of the injection window and the last completion time.
- `rejection_rate = rejected / offered`. `deadline_miss_rate = deadline_missed / admitted`.
- Latency (`finish - arrival`) and queue delay (`dispatch - arrival`) percentiles use nearest rank over completed requests only. An empty set prints `-`.

## Scenarios

`SCENARIOS` holds synthetic rates of 20, 100, 380, and 600 requests per second, each run with `max_batch_size` 1 and 8. With the default parameters, the formula's ceilings are about 105 req/s at batch size 1 and 400 req/s with full batches of 8. To explore other settings, edit `SCENARIOS`, `BATCH_SIZES`, or the `Config` defaults.
