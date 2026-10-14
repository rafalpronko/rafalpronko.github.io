---
layout: post
lang: en
title: "Keeping a RAG corpus current: dlt, dbt and index reconciliation, deletions included"
date: 2024-01-04 09:00:00 +0100
permalink: /blog/rag-corpus-dlt-dbt-index-reconciliation/
author: Rafał Prońko
categories: [data-engineering, rag]
description: "How to keep a retrieval index equal to its source: complete snapshots, desired-state diffs, index generations, safe promotion and deleting what the source no longer has."
image: /assets/images/rag-corpus-dlt-dbt-index-reconciliation.svg
---

# Keeping a RAG corpus current: dlt, dbt and index reconciliation, deletions included

![Diagram: a complete source listing is landed by dlt, dbt builds the desired documents from the latest complete snapshot, a per-tenant diff against the serving index plans the changes, and a new index generation is validated and promoted together with the checkpoint. Incomplete snapshots are never promoted; the latest complete snapshot remains eligible.]({{ '/assets/images/rag-corpus-dlt-dbt-index-reconciliation.svg' | relative_url }})

<time datetime="2024-01-04">4 January 2024</time>

**This is a historical editorial label, not an actual historical publication date.** The prepared example was executed and verified on 12 October 2026.

Adding documents to a retrieval index is the easy part. The first load works, the assistant answers questions, and everyone moves on.

The trouble starts a few weeks later. A policy page is shortened, a holiday notice is deleted, a document is moved to a group with fewer readers. The pipeline still runs green every night. The index still returns the old refund period, the deleted notice and a chunk that half of the company should no longer see.

None of this is a model problem. It is a data synchronisation problem: the index has to end up equal to what the source says *now*, including the things the source no longer says. The prepared example uses [dlt](https://dlthub.com/) for loading, [dbt](https://www.getdbt.com/) for the transformation and a small local index to show where the obvious incremental setup quietly keeps stale data.

> **Versions and verification.** The example pins releases that existed before the editorial label: dlt 0.4.2, dbt-core 1.7.4, dbt-duckdb 1.7.0 and DuckDB 0.9.2 on Python 3.11. Their APIs were checked against tagged documentation and installed packages. Reproduction uses [uv](https://docs.astral.sh/uv/), a present-day tool; the dependency lock uses `exclude-newer = "2024-01-04T00:00:00Z"`. This constrains package availability, not the date of execution: Python 3.11.12 and SQLite 3.47.1 are the modern verification runtime. The demo and tests were executed on 12 October 2026. The full runnable source, pinned dependencies, lockfile, dbt project and execution instructions are in [the example archive]({{ '/assets/examples/rag-corpus-dlt-dbt-index-reconciliation.zip' | relative_url }}).

Every Python and SQL block below is an illustrative **FRAGMENT**, with its context described beside it. Run the archive rather than copying these fragments into a standalone script.

## The index is a derived copy, so it needs a contract

A retrieval index is a materialised view of a document source. It has the same problem as any copy: it can drift. Four kinds of change have to reach it:

1. **Content change.** A document's text changes. Its old chunks must disappear, not just get new neighbours.
2. **Re-chunking.** The text is the same, but the chunker configuration changed. The number of chunks and their boundaries change, so every chunk ID changes too.
3. **Deletion.** The document no longer exists at the source.
4. **Visibility change.** The document still exists, but the set of people who may read it got smaller.

The first two are easy to see in a diff. The last two are the dangerous ones, because they arrive as *absence*: a document that is not in the latest response, a group that is not in the ACL any more. An incremental pipeline is built to ignore absence. That is its whole point.

Before writing any code, the contract needs to answer one question: **how do we know that something is gone?** There are only two honest answers.

- **An authoritative full snapshot.** The source can list *every* document of a scope (here: one tenant) as of a single version. If the listing is complete, anything not in it is gone.
- **A change feed with tombstones.** The source tells me explicitly: document `c` was deleted at time 30. Absence from a page means nothing; only the tombstone does.

What does not work is the middle ground: an incremental page of "recently updated" documents, from which I infer that everything not on the page was deleted. That page was never meant to describe the whole corpus.

## Why the default incremental tools do not delete for you

This is worth stating carefully, because the configuration names suggest otherwise.

**dlt `merge` with a `primary_key`** upserts rows. The [dlt 0.4.2 incremental loading docs](https://github.com/dlt-hub/dlt/blob/0.4.2/docs/website/docs/general-usage/incremental-loading.md) describe it as loading into a staging dataset, deduplicating by primary key, deleting the matching keys from the destination and inserting the new rows. In the installed 0.4.2 package's merge SQL, the `DELETE` is restricted to matching staged keys. A document that does not arrive in the batch is not touched. A tombstone, unless you filter it, is just another row with `deleted = true`.

**dbt `unique_key` with this adapter's `delete+insert` incremental strategy** decides which existing rows are replaced by incoming rows. The [dbt incremental models](https://docs.getdbt.com/docs/build/incremental-models) and [`unique_key`](https://docs.getdbt.com/reference/resource-configs/unique_key) documentation describe it as a match key for updates, not as a uniqueness constraint the warehouse enforces, and not as a rule that removes keys that stopped arriving. Other strategies and adapters have different semantics. An incremental model processes the rows your `is_incremental()` filter lets through. Rows that never come through stay in this example.

The demo includes a deliberately naive incremental model to show this:

**FRAGMENT — dbt model; source declarations and project configuration are in the archive.**

```sql
{% raw %}{{ config(
    materialized='incremental',
    unique_key='doc_key',
    incremental_strategy='delete+insert',
    tags=['corpus'],
) }}
-- Deliberately naive: only rows loaded since the last run are processed.
-- unique_key replaces matching keys; it never removes keys that stop arriving.
select
    tenant || '/' || doc_id as doc_key,
    tenant, doc_id, revision, title, body, snapshot_id, _dlt_load_id
from {{ source('landing', 'snapshot_rows') }}
{% if is_incremental() %}
where _dlt_load_id > (select max(_dlt_load_id) from {{ this }})
{% endif %}{% endraw %}
```

After the holiday notice is deleted at the source, `acme/holiday-2023` is still in this table. Nothing is wrong with dbt here; the model does exactly what it says.

Neither tool is at fault. They give you append, replace and upsert. Deletion is a contract you have to design.

## The design in one picture

The demo uses the snapshot contract, because it is the one most document APIs can offer (list everything, page by page). It works in five steps:

1. **Land** the full listing of one tenant with dlt as append-only rows, then append a **manifest** saying how many rows the listing declared and how many arrived.
2. **Transform** with dbt into full-table models: the latest *complete* snapshot per tenant, and the desired documents built from it.
3. **Diff** the desired documents against what the serving index contains for that tenant.
4. **Build** a new index generation next to the serving one, reusing unchanged chunks, then **validate** it against the desired state.
5. **Promote** the new generation and commit the checkpoint in one transaction. Only then delete the old generation.

If anything fails before step 5, the old generation keeps serving and the next run starts over from the durable desired state.

The index is [SQLite FTS5](https://www.sqlite.org/fts5.html): a lexical full-text index with BM25 ranking. **It is not a vector store and the demo computes no embeddings.** It keeps the example offline, with no paid API calls, and lets the tests compare exact chunk text. The per-chunk `encode` function stands in for the expensive step (an embedding call) so the example can count encoded versus reused chunks and inject a failure. These are work counts, not latency or scalability benchmarks. The reconciliation logic does not depend on the representation; the consistency guarantees do, as discussed below.

## Step 1: a snapshot is complete only if the source says so

A loop that ends is not proof that the listing is complete. A buggy source can return `next_page_token = null` early, and a source that changes between pages can skip or repeat documents. The synthetic API in the demo has an explicit contract: every page of one listing carries the same `as_of` version and the same `total`.

The dlt resource checks that contract while it pages, and the manifest records the result:

**FRAGMENT — extractor body; imports, API, pipeline, manifest and snapshot ID setup, and the `_load_now` guard are in `corpus_sync/landing.py`.**

```python
@dlt.resource(name="snapshot_rows", write_disposition="append")
def snapshot_rows():
    token, pages, seen, as_of, total = None, 0, set(), None, None
    while True:
        page = api.list_page(tenant, token)
        pages += 1
        if as_of is None:
            as_of, total = page["as_of"], page["total"]
        elif page["as_of"] != as_of or page["total"] != total:
            raise SnapshotError(f"{tenant}: listing changed between pages")
        for item in page["items"]:
            if item["doc_id"] in seen:
                raise SnapshotError(f"{tenant}: duplicate {item['doc_id']}")
            seen.add(item["doc_id"])
            yield {**item, "tenant": tenant, "snapshot_id": snapshot_id}
        token = page["next_page_token"]
        if token is None:
            break
    manifest.update(
        tenant=tenant, snapshot_id=snapshot_id, as_of=as_of, pages=pages,
        declared_total=total, row_count=len(seen),
        complete=len(seen) == total,
    )

_load_now(pipeline, snapshot_rows())
if not manifest:
    raise SnapshotError(f"{tenant}: extractor did not run; no manifest written")
_load_now(pipeline, [manifest], table_name="snapshot_manifests",
          write_disposition="append")
```

Three details matter.

- **Rows land before the manifest.** If the row load fails, there is no manifest, and rows without a manifest are never treated as a snapshot.
- **The scope is one tenant.** "Missing from the listing" only means "deleted" within the scope the listing covers. If a listing for tenant `acme` is used to compute deletions for tenant `beta`, the pipeline will happily delete all of `beta`. The tenant is part of every key, every diff and every index row.
- **`total` is a claim, not a proof.** It comes from the source. If your source cannot give you a consistent version and a total, you cannot use absence to delete. Use tombstones, or a periodic full export with a known boundary. A source that wrongly declares `total = 0` with an empty listing would empty the tenant; a cheap extra safeguard is a deletion threshold that holds back automatic promotion when a run would remove more than an agreed share of a tenant's documents. The demo does not implement one.

The snapshot rows are appended, never replaced. That keeps every snapshot inspectable and makes a failed run harmless: it adds rows nobody selects.

**Recovering from a crashed landing.** dlt keeps work between steps as *load packages* in the pipeline's working directory. If a process dies after extraction or normalisation but before the load finishes, those packages stay pending. In the installed dlt 0.4.2, `pipeline.run(data)` first normalises and loads pending packages and, if there were any, returns *without extracting* `data` (it only logs a warning). A naive retry therefore loads the leftovers of the crashed attempt, never calls the source, and leaves the manifest empty. The example handles this explicitly. Before each landing attempt, `sync_tenant` calls `drop_pending_packages(with_partial_loads=True)` and reports the discarded load IDs. Leftovers are dropped rather than loaded: a partly loaded package may leave rows behind, but if the crash came before the manifest load committed, that attempt has no manifest and is never selected. Dropping only removes local packages; it does not roll back anything already committed in DuckDB. If the process dies after the manifest load committed but before `land_snapshot` returns, that manifest stays, and it is a valid snapshot: its rows completed first and the manifest records whether the listing was complete. The next dbt build treats it like any other. Each `run` is then guarded: it refuses to start while packages are pending, calls `LoadInfo.raise_on_failed_jobs()` because 0.4.2 does not raise when a load job fails terminally, and checks that nothing is pending afterwards, so the manifest load cannot be silently replaced by a leftover either. If the extractor never filled the manifest, landing fails before anything is appended. An integration test crashes the extract → normalise → load chain at three points and checks that the retry calls the source again and that only the fresh attempt gets a manifest; a fourth test crashes right after the manifest load committed and checks that this complete snapshot is the one recovery serves. The cost of discarding is one extra listing.

Each landing attempt needs a fresh `snapshot_id`, and landing rejects an ID that already has rows or a manifest; an index retry reads the already landed desired state without re-appending that snapshot. The source version must be monotonic and identify an immutable listing. Equal totals cannot prove that the source told the truth, or compensate for an API that reuses a version for different contents. The demo infers the landing schema from document rows, so an initially empty warehouse needs schema bootstrapping before dbt can build it; a missing table causes the build to fail rather than authorise deletion.

## Step 2: dbt builds the desired state as a full table

The dbt side has no incremental logic at all. One model keeps only snapshots whose manifest says complete *and* whose landed rows match it:

**FRAGMENT — completeness predicate; the `landed` aggregation CTE and model configuration are omitted here.**

```sql
{% raw %}select m.tenant, m.snapshot_id, m.as_of, m.row_count
from {{ source('landing', 'snapshot_manifests') }} as m
left join landed as l
    on l.tenant = m.tenant and l.snapshot_id = m.snapshot_id
where m.complete
  and m.row_count = m.declared_total
  and coalesce(l.landed_rows, 0) = m.row_count
  and coalesce(l.landed_docs, 0) = m.row_count{% endraw %}
```

`latest_complete_snapshots` picks the newest one per tenant by `as_of`, and `desired_documents` is rebuilt from it on every run. Two complete snapshots with the same `as_of` are ordered by `snapshot_id`, a plain string comparison in which `acme-s10` sorts before `acme-s9`. That is harmless only because of the contract above: one `as_of` identifies one immutable listing, so either snapshot yields the same desired documents.

**FRAGMENT — desired documents model; referenced models and source declarations are in the archive.**

```sql
{% raw %}select
    r.tenant || '/' || r.doc_id as doc_key,
    r.tenant, r.doc_id, r.revision, r.title, r.body, r.acl,
    md5(strlen(r.title)::varchar || ':' || r.title || r.body) as content_hash,
    s.snapshot_id, s.as_of
from {{ source('landing', 'snapshot_rows') }} as r
join {{ ref('latest_complete_snapshots') }} as s
    on s.tenant = r.tenant and s.snapshot_id = r.snapshot_id
where r.status = 'published' and r.acl <> ''{% endraw %}
```

A document missing from the latest complete snapshot is simply absent from this table. An archived document is absent too, and so is a document nobody may read. No delete statement is needed, because nothing is carried over from the previous run.

This is a full rebuild per run. The tiny synthetic corpus provides no evidence of capacity at larger scales; measure listing cost, transformation time, memory and index build time on representative data. Partitioning complete rebuilds by tenant can reduce the work while retaining the deletion contract. `dbt build` also runs `unique` and `not_null` tests on `doc_key` and on one snapshot per tenant; if they fail, the sync stops before the index is touched. The readable `tenant/doc_id` key assumes the demo's slash-free identifiers; production keys should use an unambiguous tuple or encoding. The content hash prefixes the title with its byte length, so moving a line break from the title into the body changes the hash; a plain `title || newline || body` would give `("A\nB", "C")` and `("A", "B\nC")` the same hash and let the index reuse chunks with the wrong title. The MD5 content hash is a change detector for trusted synthetic input, not a security primitive.

## Step 3: diff desired state against the serving index

The index stores, for every chunk, the tenant, document ID, revision, content hash and ACL. Grouping the serving generation by document gives the current state, and the plan is a plain comparison:

**FRAGMENT — planner; callers supply dictionaries scoped to one tenant.**

```python
def plan(desired: dict, current: dict) -> dict:
    out = {"added": [], "content_changed": [], "metadata_changed": [],
           "removed": sorted(set(current) - set(desired)), "unchanged": []}
    for doc_id in sorted(desired):
        want, have = desired[doc_id], current.get(doc_id)
        if have is None:
            out["added"].append(doc_id)
        elif want["content_hash"] != have["content_hash"]:
            out["content_changed"].append(doc_id)
        elif (want["acl"], want["revision"]) != (have["acl"], have["revision"]):
            out["metadata_changed"].append(doc_id)
        else:
            out["unchanged"].append(doc_id)
    return out
```

`removed` is computed by subtraction. That is only safe because `desired` came from a snapshot that passed the completeness check in step 2. The same function fed with an incremental page would delete most of the corpus.

Content and metadata are separate on purpose. A changed ACL must reach the index, but it does not require re-encoding the text.

## Step 4: stable IDs and a new generation

Chunk IDs are derived, not random:

**FRAGMENT — chunk identity; imports and `spec_key` are in `corpus_sync/index.py`.**

```python
def chunk_id(tenant, doc_id, content_hash, spec, ordinal) -> str:
    raw = json.dumps([tenant, doc_id, content_hash, spec_key(spec), ordinal],
                     ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
```

The same tenant, document content hash, chunker settings and position produce the same ID. JSON preserves field boundaries, including identifiers containing separators, and the full SHA256 digest avoids deliberately truncating the ID. A changed input produces a different digest subject to the usual hash-collision assumption. Two consequences follow:

- When a document is shortened from four chunks to two, the new chunks have new IDs. The old four are not "updated"; they are simply not in the new generation.
- When the chunker changes (`spec`: chunker name, chunk size, representation), every ID changes and nothing is reused. A chunk produced by chunker v1 never survives into a generation built with chunker v2. The same goes for a new embedding model in a real system: put its name and version into the spec.

Instead of patching the serving index in place, `build_generation` writes a complete new generation next to it, with status `building`. Unchanged documents (same content hash, same spec) copy their chunks from the serving generation; everything else is chunked and encoded again. Each document is committed separately, which mimics a remote store without multi-document transactions. A crash halfway leaves a `building` generation that nobody queries and that the next run deletes.

Then `validate_generation` derives the expected chunks again and compares **every row with content**, not just counts: tenant, document ID, revision, hash, ACL, ordinal, exact chunk text and expected chunk ID. The FTS table must have exactly the same IDs and text, with no missing or duplicate rows. A shortened sequence or changed text fails validation even if its remaining ordinals are gap-free.

## Step 5: promotion and checkpoint in one transaction

**FRAGMENT — promotion core; the archive also checks candidate tenant, snapshot, version and `building` status before these writes, and defines `with_transaction`.**

```python
def promote(con, tenant, gid, snapshot, expected_gid) -> None:
    def work():
        row = con.execute("select generation_id from serving where tenant = ?",
                          (tenant,)).fetchone()
        if (row[0] if row else None) != expected_gid:
            raise PromotionConflict(f"{tenant}: serving generation moved")
        con.execute(
            "insert into serving values (?, ?, ?, ?) on conflict (tenant) do update "
            "set generation_id = excluded.generation_id, "
            "snapshot_id = excluded.snapshot_id, as_of = excluded.as_of",
            (tenant, gid, snapshot["snapshot_id"], snapshot["as_of"]))
        con.execute("update generations set status = 'retired' "
                    "where tenant = ? and status = 'serving'", (tenant,))
        con.execute("update generations set status = 'serving' "
                    "where generation_id = ?", (gid,))
    with_transaction(con, work)
```

The `serving` row is both the alias that queries join through and the checkpoint that says which snapshot the index reflects. Because they live in one row and change in one transaction, there is no moment in which the checkpoint has moved but the index has not, or the other way round. Queries see the old generation or the new one, never a mix.

The checkpoint moves **only** after a complete snapshot was landed, transformed, built and validated. A run that fails before promotion never advances it. An incomplete snapshot is never promoted, but it does not freeze the tenant either: the run still reconciles against the latest *complete* snapshot, so if an earlier complete snapshot's index build had failed, a later run with an incomplete listing can promote that earlier snapshot. That is what makes retries boring: the next run reads the same durable desired state from DuckDB and tries again. Deleting the old generation happens after the promotion has committed, so it cannot undo it: if that cleanup fails, the result is still `promoted`, with `cleanup: pending`, and the next run deletes the leftover generations before doing anything else.

A few more guards:

- A snapshot older than the checkpoint (`as_of` lower) is ignored, so a delayed job cannot roll the index back.
- If nothing changed, only the checkpoint moves; the same generation keeps serving.
- `expected_gid` is a compare-and-swap. It catches a second writer that promoted in between, but it is not a concurrency design. **The demo assumes a single writer per tenant**; concurrent writers were not tested. In production, put a lock or a lease around the whole per-tenant sync. The shared dlt pipeline and dbt rebuild also need serial execution in this example.

The search query joins through `serving` and applies the ACL filter in SQL, **before** `LIMIT`. Filtering after retrieving the top five is a classic leak: the five best chunks may all be forbidden, and the user gets nothing, or a fallback path returns them anyway.

## What the demo runs

`demo.py` runs the whole chain (dlt → DuckDB → `dbt build` → SQLite FTS5) for two tenants through six steps:

| Step | Input | What should happen |
|---|---|---|
| s1 | first complete snapshots for `acme` and `beta` | both tenants indexed; archived draft excluded |
| s2 | `acme`: refund policy shortened, holiday notice deleted, VPN guide no longer readable by `support` | new generation: refund has 2 chunks with new IDs, holiday gone, VPN chunks reused with new ACL; `beta` unchanged, checkpoint only |
| s3 | the same input again | same generation keeps serving, same chunks; only the checkpoint ID moves |
| s4 | `acme` listing truncated by the source, `beta` times out on page 2 | nothing promoted; serving state identical to s3 |
| s5 | complete `acme` snapshot with a new SSO page, encoder fails | build aborted; serving state identical to s4 |
| s6 | retry | `acme` serves the s5 snapshot including the SSO page; `beta` gets its edit |

A second, smaller scenario loads three pages of a change feed with dlt `merge` and shows the other contract: the page in which only `b` changed leaves `a` and `c` alone, a tombstone for `c` is stored as a row with `deleted = true`, and a dbt model filters tombstones out. That scenario also shows a cursor pitfall: an edit of `a` with `updated_at = 15` arrives after the cursor already reached 20, and `dlt.sources.incremental` drops it as already seen. Late changes are a property of your source and your cursor, not something merge can repair. Asking the API for overlapping pages is not enough, because the stored cursor still filters the late rows out. Either re-read a bounded window with `initial_value` and `end_value`, which in dlt 0.4.2 runs without changing the incremental state, and rely on idempotent merges, or periodically reconcile against a full snapshot.

## What actually happened

The prepared example was executed on 12 October 2026 with `env UV_CACHE_DIR=.work/uv-cache uv run --frozen python demo.py`; it exited with code 0. The full test command, `env UV_CACHE_DIR=.work/uv-cache uv run --frozen pytest -v`, also exited with code 0: **31 passed, 0 skipped**. Both commands were run with `--offline` added, against the already synced environment. Below is an abridged excerpt from the demo output; it omits the full JSON plans of s1–s3 and s6 and the unchanged `beta` line of s5:

```text
=== s1 initial load
acme: serving generation=1 checkpoint=acme-s1 chunks=12 docs=['holiday-2023', 'legal-hold', 'refund-policy', 'vpn-setup']
beta: serving generation=2 checkpoint=beta-s1 chunks=6 docs=['expenses', 'laptop-return', 'onboarding']

=== s2 edit, source delete, ACL withdrawal
acme: serving generation=3 checkpoint=acme-s2 chunks=7 docs=['legal-hold', 'refund-policy', 'vpn-setup']
beta: serving generation=2 checkpoint=beta-s2 chunks=6 docs=['expenses', 'laptop-return', 'onboarding']

=== s3 same input again
acme: serving generation=3 checkpoint=acme-s3 chunks=7 docs=['legal-hold', 'refund-policy', 'vpn-setup']
beta: serving generation=2 checkpoint=beta-s3 chunks=6 docs=['expenses', 'laptop-return', 'onboarding']

=== s4 truncated acme listing, beta timeout
{"desired_from": "acme-s3", "index": {"action": "already_current"}, "landing": "incomplete: 2/5 rows", "tenant": "acme"}
{"index": {"action": "not_attempted"}, "landing": "failed: PipelineStepFailed", "tenant": "beta"}
acme: serving generation=3 checkpoint=acme-s3 chunks=7 docs=['legal-hold', 'refund-policy', 'vpn-setup']
beta: serving generation=2 checkpoint=beta-s3 chunks=6 docs=['expenses', 'laptop-return', 'onboarding']

=== s5 complete acme snapshot, index build fails
{"index": "build failed: embedding service unavailable", "tenant": "acme"}
acme: serving generation=3 checkpoint=acme-s3 chunks=7 docs=['legal-hold', 'refund-policy', 'vpn-setup']

=== s6 retry
acme: serving generation=4 checkpoint=acme-s5 chunks=9 docs=['legal-hold', 'refund-policy', 'sso-login', 'vpn-setup']
beta: serving generation=5 checkpoint=beta-s6 chunks=6 docs=['expenses', 'laptop-return', 'onboarding']
```

In s2 the plan for `acme` was `content_changed: [refund-policy]`, `metadata_changed: [vpn-setup]`, `removed: [holiday-2023]`, with 2 chunks encoded and 5 reused. The truncated listing in s4 landed 2 of 5 declared rows; the latest *complete* snapshot was still `acme-s3`, so nothing was deleted. In s6 the retry built from the snapshot that had already landed in s5 (`acme-s5`), encoded only the new SSO page and reused 7 chunks.

After s6, keyword searches returned:

```text
search: {"acme 'refund days' as all": ["refund-policy"], "acme 'gateway' as support": [], "acme 'gateway' as eng": ["vpn-setup"], "acme 'December' as all": [], "acme 'price' as all": []}
antipattern: [["acme/holiday-2023"], ["acme/legal-hold"], ["acme/pricing-draft"], ["acme/refund-policy"], ["acme/sso-login"], ["acme/vpn-setup"]]
desired: [["acme/legal-hold"], ["acme/refund-policy"], ["acme/sso-login"], ["acme/vpn-setup"]]
landed_holiday_rows: [["acme-s1"]]
```

The naive incremental model still holds the deleted holiday notice (and the archived draft, which it never filtered); the full-table model does not. The `support` group no longer finds the VPN guide. The change-feed scenario ended with the tombstoned `c` stored as `('c', 2, None, True)`, `a` still at `alpha v1` because the late edit was below the cursor, and the dbt model returning `[('a', 1, 'alpha v1'), ('b', 2, 'bravo v2')]`.

The tests assert complete sets and chunk text rather than counts: for example, that the shortened refund policy is exactly two chunks with the expected text and no chunk ID shared with the old version, that the VPN guide kept its chunk IDs but changed its ACL, and that serving state after the failed steps is equal, chunk for chunk, to the state before them. Direct index tests cover re-chunking, stale snapshots, promotion conflicts, failed builds and retry, and an empty authoritative desired state deleting only its tenant. Regression tests first exposed acceptance of damaged chunk/FTS text, a missing tail, mixed metadata, an incorrect tenant or ID, ambiguous ID field boundaries and promotion of another tenant's candidate. Those tests failed before the fixes and passed afterwards. A later technical review found five more gaps, each first reproduced by a failing test and then fixed: a retry after a crashed landing (in normalisation, in the row load, or in the manifest load) loaded only the leftover packages and failed with an empty manifest; a reused attempt ID was appended again; a manifest could be appended although the extractor never ran or a load job failed; a title/body pair that differed only in where a line break sat got the same content hash, so retries failed validation over and over; and a cleanup failure after a committed promotion was raised as an error although the new generation was already serving. It is now reported as `cleanup: pending`. The demo output was byte-identical before and after these fixes. The interrupted earlier test run is not evidence of a pass.

## What this does not give you

**Not physical erasure.** Removing a document from the serving index makes it unretrievable through this index. The landed snapshot rows still contain it; so do old dbt targets, backups, logs and any cache downstream. In the demo the deleted holiday notice is still in `landing.snapshot_rows` for snapshot `acme-s1`, and a test asserts that. If you need erasure (a legal request, a leaked secret), it is a separate process with its own retention rules, across every copy.

**Not instant security.** An ACL withdrawal reaches the index on the next successful run. Between the change at the source and promotion, the old ACL is served. If a run fails, the window gets longer. If that window is not acceptable, check permissions at query time against the source of truth as well, and treat the index ACL as a pre-filter, not the only gate.

The synthetic search function accepts a tenant and one group from its caller; it performs no authentication or group-membership lookup. A real application must derive both from trusted identity and authorisation data, never from a user-supplied group claim. The comma-separated ACL convention assumes canonical group names without commas or surrounding whitespace. FTS relevance statistics are shared across tenants, so this demo is not a design for isolated per-tenant ranking or protection against statistical side channels.

**Not exactly-once.** Fresh landing attempts use distinct snapshot IDs; landing refuses an ID that already has rows or a manifest instead of appending it again, and a crashed attempt's pending packages are discarded, not completed. The dbt models are full rebuilds, and index promotion is conditional. Repeated source contents and retries from durable desired state converge to the same serving chunks. That is index idempotence, not a guarantee that nothing ever runs twice.

**Not a remote vector store.** All of the index lives in one SQLite file, so "promote and checkpoint in one transaction" is real here. A hosted vector database usually gives you per-request writes and maybe aliases or namespaces. The same idea carries over (build a new namespace or collection, validate it, switch an alias, store the checkpoint *after* the switch succeeds, delete the old one later), but the atomicity is only as good as what that store offers. Check it; do not assume it.

**Not a fit for every source.** Full snapshots cost a full listing per run. For large sources, use a change feed with tombstones for frequent runs and a full snapshot reconciliation now and then to catch whatever the feed missed.

## Checklist

- Know which contract your source gives you: complete snapshot per scope, or a feed with tombstones. Never infer deletion from an incremental page.
- Record completeness explicitly (manifest, declared total, consistent version) and reject listings that do not meet it.
- Keep the tenant in every key, diff and query.
- Build desired state as a full table; let deletion be absence from that table.
- Derive chunk IDs from tenant, document, content hash, chunker/embedding spec and position.
- Build a new generation, validate it against desired state by content, then promote and checkpoint together.
- Filter by ACL inside the query, before the limit.
- Run one writer per tenant, or lock.
- Treat deletion from the index, withdrawal of visibility and physical erasure as three different jobs.

## Sources

- dlt 0.4.2, incremental loading (tagged docs): <https://github.com/dlt-hub/dlt/blob/0.4.2/docs/website/docs/general-usage/incremental-loading.md>
- dlt 0.4.2 release: <https://github.com/dlt-hub/dlt/releases/tag/0.4.2>
- dbt incremental models: <https://docs.getdbt.com/docs/build/incremental-models>
- dbt `unique_key`: <https://docs.getdbt.com/reference/resource-configs/unique_key>
- dbt-core 1.7.4 release: <https://github.com/dbt-labs/dbt-core/releases/tag/v1.7.4>
- dbt-duckdb 1.7.0 release: <https://github.com/duckdb/dbt-duckdb/releases/tag/1.7.0>
- DuckDB 0.9.2 package release timestamps: <https://pypi.org/pypi/duckdb/0.9.2/json>
- SQLite FTS5: <https://www.sqlite.org/fts5.html>
- uv documentation (modern reproduction tooling): <https://docs.astral.sh/uv/>

The dbt, SQLite and uv pages are current documentation (read 12 October 2026); the dbt behaviour was cross-checked against the installed dbt-core 1.7.4 and dbt-duckdb 1.7.0 packages. Package release dates were checked on PyPI and GitHub on 12 October 2026. Current adapter feature tables and newer dlt refresh/CDC or dbt snapshot-YAML features are not historical evidence for these pins and are not used by this example.
