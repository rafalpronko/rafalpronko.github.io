---
layout: post
lang: en
title: "Batching and serving: measure throughput, latency, and failures under load"
description: "A deterministic dynamic-batching simulator that keeps rejections, deadline misses, goodput, and tail latency separate, plus a load-test plan for a real endpoint."
author: Rafał Prońko
date: 2024-07-18 09:00:00 +0200
permalink: /blog/batching-serving-under-load/
image: /assets/images/batching-serving.svg
---

> **Dates and evidence.** The date above, 2024-07-18, is an editorial label for this slot in the series. It fits the material the article relies on. Python 3.12.4 was released on June 6, 2024, and the Orca paper on iteration-level scheduling appeared at USENIX OSDI 2022. The label does not mean that anything here was written, run, or measured in July 2024. The simulator was executed on 2026-10-12 with Python 3.12.11 in a virtual environment created by uv 0.7.8. Those are the current tools used for this run. I do not claim they existed in 2024. Test status: EXECUTED_PASS, 17 unittest tests passed. Only the simulator was run. No endpoint, model server, or GPU was measured. Sources were checked on 2026-10-12. Every number below comes from a virtual clock and a service-time formula I chose.

When each model execution has a large fixed cost and traffic is heavy enough to fill batches, batching can raise throughput on the same hardware without changing the model. That makes it one of the first serving knobs people turn. It is also one of the easiest to misreport. The usual sequence is: raise the maximum batch size, rerun the benchmark, see requests per second go up, ship. That single number hides two things: where the waiting moved, and which requests quietly stopped being counted.

In this post I build a small deterministic simulator of one worker with dynamic batching. I use it to make the accounting explicit: what was offered, what got in, what finished, what finished in time, and what was turned away. Then I describe how I would measure a real endpoint. A simulator that enforces its own algorithm is not evidence of anyone's capacity.

## What batching buys, and what it costs

Some of the cost of a model execution grows much more slowly than the number of inputs it carries. Examples are framework dispatch, kernel launches, setting up transfers, and, for many models, moving weights through the memory hierarchy. Grouping requests amortizes that shared part whenever one more item costs less than the shared part.

Real hardware is less tidy than that sentence. Weight traffic is often shared across a batch, but it is not a constant per batch. Kernel selection and tiling can change how often weight tiles are reread, and activations and other per-item memory traffic grow with batch size. Throughput usually scales nonlinearly with batch size:

- when execution is memory-bound and shared weight traffic dominates, extra items can add relatively little time;
- once it becomes compute-bound, the cost becomes roughly proportional;
- there are steps where kernels change or memory runs out.

The real overhead depends on the model, framework, hardware, and input shapes, and has to be measured. The simulator deliberately uses the simplest linear model instead.

Fragment (pseudocode for the simulator's service-time assumption, not runnable code):

```
service_time(n) = overhead + per_item × n
```

Take a made-up overhead of 8 ms and 1.5 ms per item. One request takes 9.5 ms and a batch of eight takes 20 ms. Under this formula, the ceiling is about 105 requests per second at batch size 1. If every batch is full at eight, it is 400 per second. Those ceilings come from two numbers I typed in, not from a GPU.

The cost is waiting. A request in a batch waits for other requests, and how long depends on traffic:

- **Sparse traffic.** The batch never fills. The oldest request waits the full `max_wait` and then goes alone. It pays for batching and amortizes nothing.
- **Heavy traffic.** Batches fill almost immediately, and most of the waiting is queueing behind a busy worker. Here batching reduces waiting, because the worker drains the queue faster.

Configuration names tend to blur one distinction, so I'll state it early. `max_wait` is measured from the oldest waiting request's arrival. It limits how long the batcher deliberately holds a dispatch that the worker is free to run. It does not bound queueing behind a busy worker, and under load a request can wait many multiples of `max_wait`. The client deadline covers everything: queueing, batch assembly, and service. Setting `max_wait` to 5 ms says nothing about whether p99 meets a 100 ms deadline.

## Fixed, dynamic, and continuous batching

**Fixed batching** waits for exactly N requests, or pads to N. It suits offline jobs. Online, latency at low load is unbounded unless something forces a flush.

**Dynamic batching** dispatches when N requests are waiting or when the oldest has waited a bounded time, whichever comes first. My simulator implements this. When inputs have variable shapes, a batch is usually padded to its longest member, and padded compute is still compute. In a naive implementation, one long input plus seven short ones costs roughly as much as eight long ones.

**Continuous (iteration-level) batching** is used by some LLM generation servers, not all. It exists because a generation request is not one unit of work:

- **Prefill** processes the prompt tokens, and its cost scales with prompt length.
- **Decode** produces one token per step for each active sequence.

An iteration-level scheduler admits and retires sequences between decode steps, so a short answer can free its slot without waiting for a long one to finish. The Orca paper (USENIX OSDI 2022) describes this idea as iteration-level scheduling: the engine runs one model iteration at a time on the batch instead of a whole request. Servers that adopt the idea differ in how they schedule prefill, preemption, and memory, so check what a given server actually does.

The metrics change accordingly:

- **time to first token (TTFT)** is measured from the request to the moment the first output token reaches the client. It includes request preprocessing such as tokenization, queueing, prefill, and delivery of the first token;
- **inter-token latency** is the interval between consecutive output tokens. It depends on decode step time, on how many sequences share a step, on prefills interleaved with decoding, and on how tokens are streamed to the client;
- **total latency** ≈ TTFT + (output_tokens − 1) × mean inter-token interval.

Requests per second is not comparable between two LLM runs unless the prompt and output length distributions match. At minimum, report input and output tokens per second together with those distributions. I did not run an LLM experiment for this post. The simulator has no tokens, and every request in it is one item of equal cost.

## The simulator

![Open-loop arrivals enter a bounded queue or are rejected; a batch leaves when it is full or the oldest request has waited max_wait; the worker spends overhead plus per-item time; completed requests are classified against the client deadline]({{ '/assets/images/batching-serving.svg' | relative_url }})

The model is deliberately small and explicit:

- **Virtual clock.** Time is an integer number of microseconds. Nothing sleeps and nothing is timed. The same configuration always produces the same output.
- **Open-loop arrivals.** Request `i` arrives at `i * 1_000_000 // rate` microseconds, for every slot inside a 10-second injection window. The schedule is fixed before the run and does not depend on completions. This matters because a client that waits for each response before sending the next request slows down whenever the server slows down. Such a client hides the queue it should be measuring.
- **Bounded waiting queue.** If `queue_capacity` requests (default 64) are already waiting, a new arrival is rejected immediately. Requests inside the worker do not count against the limit.
- **Batch trigger.** The next batch leaves at `max(free time, trigger)`. The trigger is the earlier of two moments: the oldest waiting request's arrival plus `max_wait_us`, and the arrival of the `max_batch_size`-th waiting request. A freed worker therefore does not dispatch unconditionally. It dispatches immediately only if a full batch is already waiting or the oldest request has already waited `max_wait_us`. Otherwise it stays idle until one of those conditions is met. A batch takes up to `max_batch_size` of the oldest requests. A dispatch due at time *t* happens before an arrival at time *t*.
- **Deadline classification.** Every request has a client-visible deadline of 100 ms. The simulator does not cancel work. If `finish - arrival` exceeds the deadline, the request counts as a deadline miss. The worker still completes it, and its latency stays in the distribution. This is a common real failure mode: the client gives up and the server keeps computing for nobody. Propagating cancellation is a separate design decision, and this model represents its absence.
- **Drain.** After the injection window, no new requests arrive and the worker runs until the queue is empty. Every admitted request completes, so nothing is silently left in flight.

The core loop is short enough to read in full. `next_launch` answers one question: when would the next batch leave if nothing else arrived? The loop compares that time with the next scheduled arrival.

Fragment of `simulate.py` (not runnable on its own; `Config`, `Request`, `arrivals`, and `dispatch` are defined in the full file):

```python
def next_launch(queue: deque[Request], free_us: int, config: Config) -> int | None:
    """When the next batch leaves if no further requests arrive."""
    if not queue:
        return None
    trigger = queue[0].arrival_us + config.max_wait_us
    if len(queue) >= config.max_batch_size:
        trigger = min(trigger, queue[config.max_batch_size - 1].arrival_us)
    return max(free_us, trigger)


def simulate(config: Config) -> list[Request]:
    """Run arrivals through a bounded queue and one batching worker, then drain."""
    config.validate()
    requests = [Request(t) for t in arrivals(config)]
    queue: deque[Request] = deque()
    free_us, index = 0, 0
    while index < len(requests) or queue:
        launch_us = next_launch(queue, free_us, config)
        upcoming = requests[index] if index < len(requests) else None
        if launch_us is not None and (upcoming is None or launch_us <= upcoming.arrival_us):
            free_us = dispatch(queue, launch_us, config)
        elif len(queue) >= config.queue_capacity:
            upcoming.outcome = "rejected"
            index += 1
        else:
            queue.append(upcoming)
            index += 1
    return requests
```

There is no event heap, because there are only two kinds of events and one worker. The loop processes whichever comes first, a dispatch or an arrival. If an arrival fills the batch, the next iteration simply computes an earlier launch time.

## Counts and denominators

Most misleading serving numbers put an honest numerator over the wrong denominator. These are the definitions the simulator uses:

- **offered**: requests scheduled inside the injection window;
- **rejected**: offered requests that found the queue full;
- **admitted** = offered − rejected;
- **completed**: admitted requests that finished. Because of the drain, this equals admitted;
- **within_deadline** and **deadline_missed** partition completed;
- **offered rate** = offered / injection window;
- **completed throughput** = completed / observation window. The observation window is the longer of the injection window and the time of the last completion;
- **goodput** = within_deadline / the same observation window;
- **rejection rate** = rejected / offered; **deadline-miss rate** = deadline_missed / admitted;
- **latency** = finish − arrival and **queue delay** = dispatch − arrival, both over completed requests only;
- **percentiles** use nearest rank: sort the values and take the ⌈q·n⌉-th one, with no interpolation. An empty set gives no value, not zero.

Two of these choices are deliberate:

- **Rejections stay out of the latency distribution.** A rejection returns almost instantly. Pooling it with successes would make p50 look better exactly as the system gets worse.
- **Deadline misses stay in, at their full completion time.** Dropping them would cut off the tail, which is where the problem is.

The tests assert the conservation identities, so a later edit cannot quietly break them.

## Scenarios

The CLI runs `max_batch_size` 1 and 8 against four synthetic rates:

- 20/s (low);
- 100/s (near the batch-1 ceiling of about 105/s);
- 380/s (near the batch-8 ceiling of 400/s);
- 600/s (beyond both).

Overhead, per-item time, queue capacity, deadline, and injection window are the same in every run.

Some consequences follow from the design before anything runs:

- **At 20/s**, arrivals are 50 ms apart and `max_wait` is 5 ms. With batch size 8, every request waits the full 5 ms and goes alone. A test checks exactly that.
- **At 380/s**, batch size 1 is past its ceiling, so its queue must fill and reject.
- **At 600/s**, both configurations are overloaded. The bounded queue turns the excess into rejections plus queueing delay of up to roughly `queue_capacity / batch_size` service times.

## Running it

The three commands below create the environment, run the simulator, and run the tests. Run them from the directory that contains the example files:

```sh
uv venv --python 3.12 .venv
.venv/bin/python simulate.py
.venv/bin/python -m unittest -v test_simulate
```

The simulator has no dependencies, so the virtual environment only pins the interpreter. On 2026-10-12, uv 0.7.8 resolved Python 3.12.11, and all 17 unittest tests passed. `.venv/bin/python simulate.py --json` prints the same rows as JSON.

## Results

This is the unedited output of `.venv/bin/python simulate.py` from that run (2026-10-12, Python 3.12.11, uv 0.7.8). It is simulator output only, not a measurement of any endpoint or GPU.

```
scenario  rate/s  batch  offered  rejected  completed  missed  goodput/s  done/s   p50 ms   p95 ms   p99 ms  queue p99 ms  mean batch
     low      20      1      200         0        200       0       20.0    20.0      9.5      9.5      9.5           0.0         1.0
     low      20      8      200         0        200       0       20.0    20.0     14.5     14.5     14.5           5.0         1.0
 near_b1     100      1     1000         0       1000       0      100.0   100.0      9.5      9.5      9.5           0.0         1.0
 near_b1     100      8     1000         0       1000       0      99.96   99.96     14.5     14.5     14.5           5.0         1.0
 near_b8     380      1     3800      2683       1117    1103       1.32  105.26  616.079  617.369  617.474       607.974         1.0
 near_b8     380      8     3800         0       3800       0     379.05  379.05   28.474   36.606   36.922        18.422       7.037
overload     600      1     6000      4883       1117    1105       1.13  105.26  616.667    617.5    617.5         608.0         1.0
overload     600      8     6000      1940       4060    3965       9.34  399.33    172.5  179.167  179.167       159.167      7.9764
```

The columns map to the definitions above:

- `rate/s` is the offered rate and `batch` is `max_batch_size`;
- `missed` is deadline_missed;
- `done/s` is completed throughput, and both it and `goodput/s` are divided by the observation window;
- the p-columns are completion latency, and `queue p99 ms` is queue delay;
- `mean batch` is the average number of requests per dispatched batch.

I read the table in a fixed order:

1. **Conservation.**
2. **Goodput against completed throughput.** The gap is work the worker did for clients who had already given up.
3. **Rejection rate.**
4. **p99 against the deadline**, and only then.

**Conservation.** In every row, offered = rejected + completed, for example 3800 = 2683 + 1117 and 6000 = 1940 + 4060. Because of the drain, every admitted request completes, so no request is missing from the accounting.

**Drain and the two windows.** The offered rate uses the 10-second injection window. `done/s` and `goodput/s` use the observation window, which extends to the last completion and therefore includes the drain. Two rows show why the distinction matters:

- **100/s, batch 8.** This row reports 99.96/s rather than 100.0/s. The last request arrives at 9.990 s and finishes 14.5 ms later, at 10.0045 s, so 1000 completions are divided by 10.0045 s. At batch size 1, the last request finishes at 9.9995 s, inside the injection window, so the window stays at 10 s and the row shows 100.0.
- **Saturated batch-1 rows.** The worker never idles, so 1117 requests × 9.5 ms puts the last completion at about 10.61 s. That gives 105.26/s, the formula ceiling. Dividing by the injection window instead would report 111.7/s, more than the worker can do. The extra 0.61 s is the drain of the full 64-request queue.

**Low and moderate load.** At 20/s and 100/s, neither configuration accumulates a backlog. Batch size 1 answers in the 9.5 ms service time. With batch size 8, every request starts 5 ms after arrival and finishes 14.5 ms after arrival. At 20/s, that delay is the full 5 ms batching hold. At 100/s, the second request arrives at 10 ms, waits 4.5 ms behind the worker and another 0.5 ms for the batching trigger, then starts 5 ms after arrival and finishes 14.5 ms after arrival. The mean batch is 1.0 in both rows, so batching amortizes nothing. Both configurations still meet the deadline for every request.

**380/s.** The two batch sizes split sharply here.

- **Batch size 1** is past its ceiling. It rejects 2683 of 3800 requests (70.6%) and completes 1117 at 105.26/s. Only 14 of those finish within 100 ms; 1103 miss, a 98.7% miss rate. p50 is 616 ms, which is about 64 queued requests × 9.5 ms plus the request's own service time. `done/s` looks like a busy, healthy worker, but goodput is 1.32/s.
- **Batch size 8** admits everything. It averages 7.04 requests per batch, keeps p99 at 36.9 ms, and delivers 379.05/s of goodput. Its queue p99 is 18.4 ms, well above the 5 ms `max_wait`. That is queueing behind a busy worker, not batching hold, which is the distinction from the beginning of the post.

**600/s.** Both configurations are overloaded.

- **Batch size 1** completes the same 1117 requests as at 380/s, because the worker ceiling has not changed. The extra offered load turns entirely into rejections: 4883 of 6000, or 81.4%. Twelve requests meet the deadline, and goodput is 1.13/s.
- **Batch size 8** is the most instructive row. `done/s` reaches 399.33, essentially the 400/s ceiling of the formula, with batches nearly full at 7.98. Yet 3965 of 4060 completed requests miss the deadline (97.7%), and goodput is 9.34/s. Going from 380/s to 600/s raised completed throughput from 379 to 399/s but cut goodput from 379 to 9.3/s. Queue p99 is 159 ms, about 64 / 8 batches × 20 ms. Once the queue fills, most admitted requests wait behind it for longer than the 100 ms deadline. Across the whole run, 95 requests still finish within the deadline. The run also rejects 1940 requests (32.3%).

A configuration is better only if it improves the envelope: more goodput at an acceptable tail and failure rate. Moving a single requests-per-second number does not count. Within this model and these made-up costs, batch size 1 holds the deadline up to 100/s, batch size 8 holds it at 380/s, and neither holds it at 600/s. That statement is about the formula, not about any hardware.

## Measuring a real endpoint

The simulator shows what its own algorithm does with its own inputs. To learn what a real server does, I would run a separate experiment:

1. **Pin the workload.** Record the model version, batching configuration, hardware, replica count, and a request mix with known input and output length distributions. Without lengths, the results cannot be compared with anything.
2. **Warm up and discard.** Model loading, graph capture or compilation, caches, and autoscaling all distort the first minutes. Run a warmup phase and keep it out of the results.
3. **Generate load open-loop.** Compute intended send times from the target rate, either constant or Poisson, and send whether or not earlier requests have returned. Measure latency from the *intended* send time.

   A closed-loop generator with N virtual users stops sending when the server stalls. The stall then shows up as a handful of samples instead of every request that should have been sent. That is coordinated omission. Closed loop is legitimate when you deliberately model a fixed population of users; if so, say so in the report.
4. **Measure the generator.** Log intended and actual send times, and report the issued rate and the lag distribution next to the target rate. If the generator fell behind, the run tested a lower rate than its label, so rerun with more generator capacity.

   In Python, asyncio tasks give concurrency inside one event-loop thread. They give neither parallel compute nor GPU capacity, and a loop busy parsing responses delays its own sends. Use `time.perf_counter()` or `time.monotonic()` for intervals; `time.time()` follows the system clock and can jump.
5. **Classify every outcome.** Use separate classes for success within SLO, success late, client timeout, rejection or overload status, other server error, and connection error. Count each class, and keep every request in the denominator.
6. **Step the rate past saturation.** Increase offered load in steps, each long enough for the queue to settle. Repeat each step at least three times and report the spread. Saturation shows up as goodput flattening or falling while offered load rises and the tail grows.
7. **Report an SLO envelope.** For each batching configuration and length mix, report the highest offered rate that still meets the targets. For example: p99 latency under the limit, and rejections plus errors within budget. For generation, set TTFT and inter-token latency targets separately.
8. **Correlate with server metrics.** Queue depth, batch size distribution, memory, and accelerator utilization help explain client-side results. They do not replace them.

If the bounded queue lives in a Python gateway, `asyncio.Queue(maxsize=N)` gives the shape directly: `put_nowait()` raises `QueueFull`, and that is the rejection path. asyncio queues are not thread-safe, so use them only from the event loop that owns them.

## Limits

The model has one worker and one FIFO queue. All items cost the same. There is no padding, no tokens, no cancellation, no retries, and no network.

Retries deserve a note: they would turn rejections into extra offered load and make overload worse. The model leaves them out on purpose.

Arrivals are perfectly regular. Real traffic is burstier, and bursts raise the tail at the same mean rate. The service-time formula is linear by assumption and was not fitted to any measurement. As discussed above, real fixed costs and batch scaling are hardware- and model-dependent and usually nonlinear.

The simulator is useful for reasoning about accounting and trade-offs. It is not capacity evidence for any server, model, or GPU.

## Files and sources

Example files (Python 3.12, no dependencies; executed on 2026-10-12 with Python 3.12.11 and uv 0.7.8):

- [`simulate.py`]({{ '/assets/examples/batching-serving/simulate.py' | relative_url }})
- [`test_simulate.py`]({{ '/assets/examples/batching-serving/test_simulate.py' | relative_url }})
- [`pyproject.toml`]({{ '/assets/examples/batching-serving/pyproject.toml' | relative_url }})
- [`README.md`]({{ '/assets/examples/batching-serving/README.md' | relative_url }})

Sources (all checked on 2026-10-12):

- Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* USENIX OSDI 2022: <https://www.usenix.org/conference/osdi22/presentation/yu>. Cited here only as the conceptual source for iteration-level scheduling, based on the published abstract. This article makes no claims about the paper's full text or its benchmark results.
- Python 3.12.4 release, June 6, 2024: <https://www.python.org/downloads/release/python-3124/>
- `time` module, Python 3.12.4 docs: <https://docs.python.org/release/3.12.4/library/time.html>
- asyncio queues (bounded `maxsize`, `QueueFull`, not thread-safe): <https://docs.python.org/release/3.12.4/library/asyncio-queue.html>
- asyncio coroutines and tasks: <https://docs.python.org/release/3.12.4/library/asyncio-task.html>
