---
layout: post
lang: en
title: "Experiments, artifacts and data versions: what to record"
author: Rafał Prońko
date: 2024-07-25 09:00:00 +0200
permalink: /blog/experiments-artifacts-data-versions/
description: "What a training or evaluation run must record so you can later say which data, code, split and environment produced a result, and a small standard-library Python manifest that checks it."
categories: [mlops, data-engineering, python]
image: /assets/images/experiments-artifacts-data-versions.svg
---

Sooner or later someone asks a simple question about a model: *which data did it see?* Not "roughly the customer table from spring", but which rows, in which version, split how, by which code, in which environment. If the honest answer is "probably the file that is in the bucket now", the result cannot be checked, compared or safely rebuilt.

This article is about the record that answers that question. It is deliberately tool-agnostic and small: the vocabulary, the fields that matter, the traps (hashes that change when nothing changed, seeds that promise more than they deliver), and a little over 200 lines of standard-library Python that write and verify such a record. The next article in the series builds the same ideas into an MLflow pipeline; here the goal is to know *what* to record before choosing *where*.

![Run record and provenance chain: code, data version, contract, environment and seed feed a run, which produces artifacts; a verifier compares the record with what is on disk now]({{ "/assets/images/experiments-artifacts-data-versions.svg" | relative_url }})

## Four words that get mixed up

Tracking tools use slightly different terms, but most of them map onto the same four ideas. MLflow's tracking documentation is a useful reference for the first three: a **run** is an execution of some piece of data science code, "for example, a single `python train.py` execution", and each run records metadata (metrics, parameters, start and end times) and **artifacts**, the output files of the run such as model weights or images. An **experiment** "groups together runs for a specific task" ([MLflow Tracking concepts](https://mlflow.org/docs/2.14.3/tracking.html)).

- **Experiment**: a question you are trying to answer, for example "does adding support-ticket features improve churn ranking?". It is a folder for comparable runs, not a unit of reproducibility.
- **Run**: one execution with one set of inputs. This is the unit you want to be able to verify and, ideally, replay.
- **Artifact**: something a run produced: a model file, an evaluation report, a plot, a predictions table. Artifacts are outputs; they should be identified by content, not just by file name.
- **Data version**: a specific, retrievable state of an input dataset. It is an input to a run, and very often the output of an upstream run.

The fourth word is the one tools tend to treat loosely. A run that logs `data_path = "s3://bucket/customers.csv"` has recorded a location, not a version. If the object behind that path is overwritten next week, the run record still looks complete, and it now points at different data.

## What a run record has to answer

A useful test: could a colleague who was not involved answer these questions from the record alone?

| Question | What to record |
|---|---|
| Which code? | Commit ID, **and** whether the working tree was dirty, plus a digest of the uncommitted diff |
| Which data? | An immutable location for each input **and** a content digest of what was read |
| Which rows went where? | The key column, the split rule and its salt, and a digest of the IDs in each split |
| What was the model allowed to see? | Feature columns, target column, and how the evaluation metric is defined |
| In which environment? | Interpreter version, platform, installed packages (or a lock file digest) |
| With which randomness? | Seeds, and a note on known non-deterministic steps |
| What came out? | Artifacts with digests, and the metrics with the split they were computed on |

None of these is exotic. What usually goes wrong is that one of them is recorded in a form that *looks* precise but cannot be checked.

## Byte identity vs logical identity

The standard way to identify a file's contents is a cryptographic digest. Python's `hashlib` always provides `sha256()`, and since Python 3.11 `hashlib.file_digest()` is a helper for efficient hashing of a binary file object ([Python 3.12 hashlib docs](https://docs.python.org/3.12/library/hashlib.html)). The same page warns that some algorithms, including MD5 and SHA-1, have known collision weaknesses, which is a good enough reason not to use them for content identity.

A SHA-256 of the file is **byte identity**: change one byte and the digest changes. That is exactly what you want for "is this the file the run read?". It is too strict for "are these the same records?". Re-exporting a CSV with Windows line endings, writing rows in a different order, or reordering columns all change the bytes while the dataset means the same thing.

So the example records two digests:

- `sha256`: the bytes of the file.
- `logical_sha256`: a digest of a canonical form of the records: rows keyed by ID and sorted, columns sorted, values compared as text with surrounding whitespace stripped, line endings ignored.

When verification finds that `sha256` changed but `logical_sha256` did not, it reports `bytes_changed_logical_same`, a warning rather than an error. When both change, the records are different. Be explicit about what your canonical form ignores. This one deliberately does **not** normalise numbers, so `12.5` and `12.50` count as different values. That is a design choice to write down, not a detail to discover later.

Here is exactly what this canonical form does, because the details matter:

- Every value, including the key column's value, is stripped of leading and trailing whitespace before hashing, so `" 10 "` and `10` are the same value.
- Record *identity* uses the raw key, not the stripped one. `c1` and `c1 ` are two different records: they are not rejected as duplicates, and each raw key is hashed independently for the split, so they may or may not land in the same split. A re-export that pads a key can leave `logical_sha256` unchanged, but it is still caught as `split_changed`, because the split ID digests use the raw keys: the recorded membership digest changes even when the padded ID keeps its train or eval assignment.
- Only the header's columns count, and both of the following cases assume the row's key is still present. Extra trailing fields are ignored: the row hashes the same as without them, so only the byte digest changes and the result is the `bytes_changed_logical_same` warning. Missing trailing fields are read as empty strings: if those fields were empty in the recorded data, that is the same warning; if a non-empty value was lost, `logical_sha256` changes and `verify` reports a `content_changed` error.

## A digest is not a copy

A digest tells you whether something changed. It cannot give you the data back. A run record that contains only `sha256: a143e5…` lets you *reject* the wrong file; it does not tell you where to get the right one.

That is why each data entry in the manifest has both:

- a `uri` pointing to an **immutable** copy: an object path that is never overwritten, a versioned object, a snapshot table, or a content-addressed path, and
- the digest, so whoever retrieves that copy can confirm it is what the run read.

Neither half is enough alone. An immutable URI without a digest relies on everybody respecting "immutable". A digest without a retrievable copy turns replay into an archaeology project.

## Code: the commit is not enough

Recording `git rev-parse HEAD` is the obvious first step. The less obvious second step is recording whether the working tree was clean. A run started from a dirty checkout did not run the committed code. The example records the commit, a `dirty` flag, a SHA-256 of `git diff HEAD --binary`, and the list of untracked files. The diff digest covers changes to tracked files only. Untracked files are recorded by name, and their contents are neither hashed nor archived. The diff digest will not let you rebuild the code, but it tells you that the commit alone is not the whole story and lets you match a run to a saved patch. Many teams simply refuse to start a "real" run from a dirty tree; recording the flag makes that rule enforceable.

## The data contract: key, features, target, split, metric

Two runs trained on the same file can still be incomparable. One used `support_tickets` as a feature, the other did not; one put customer `c00042` in eval, the other in train. The record needs an explicit **contract**:

- the **key** column that identifies a record,
- the **feature** columns and the **target** column (and a check that the target is not a feature),
- the **split rule**: here a hash of `salt:record_id` mapped to a bucket, so the assignment does not depend on row order and does not change when new rows are appended,
- a digest of the IDs that ended up in each split,
- the definition of the **evaluation metric**, including which split it is computed on.

Recording split *membership* (via the ID digest) rather than just "80/20, seed 42" matters. A hash-based split with a 20% target does not produce exactly 20%: in the demo below, 117 of 500 rows land in eval. Recording the actual count and the ID digest removes the guesswork.

## Environment

Interpreter version, implementation, platform, and the installed packages. The example lists every installed distribution as `name==version` and stores a digest of that list, so verification can flag "the environment changed" in one comparison. In a real project, recording the digest of a lock file is usually more useful than the raw list. Either way, the point is the same: "Python 3, latest pandas" is not an environment.

## Randomness and non-determinism

A seed is necessary and nowhere near sufficient.

- Seeding a generator makes *that generator* repeatable on the same code path. It says nothing about other sources: library-internal RNGs, data loader worker order, multi-threaded reductions where floating-point addition order varies, or GPU kernels that are non-deterministic unless explicitly configured otherwise.
- String hashing in Python is randomised per process unless `PYTHONHASHSEED` is fixed, so iterating over a `set` of strings can produce a different order in a different process. Code that "sorts later" is fine; code that takes "the first 1000 items of a set" is not.
- I would not assume that a seeded generator produces identical output across different Python or library versions. That is one reason the example records the **digest of the generated data** and not only the seed that produced it.

Record seeds, record which steps are known to be non-deterministic, and judge replay by comparing artifacts and metrics within a tolerance you have decided in advance, rather than assuming bit-for-bit equality.

## The provenance chain

One run's artifact is the next run's data version. A raw extract becomes a cleaned snapshot, which becomes a feature table, which is split, which trains a model, which produces an evaluation report. If every step records the digests of its inputs and outputs, the steps link into a chain: given the report, you can walk back to the exact snapshot it came from, and given a bad snapshot, you can find every model that consumed it.

The chain breaks at the first step that records a path instead of a version. In practice that is usually the very first one: "we read from the production table". Snapshot it, or record the query plus a consistent read point plus the digest of what came back.

## LLM systems: recorded drift vs guaranteed replay

Retrieval-augmented and prompt-based systems have the same problem with more moving parts. A useful run record for an LLM pipeline includes:

- the **corpus** version (documents and their digests),
- the **chunking** configuration and the **embedding model** identifier,
- the **index** build: which corpus version and embedding model produced it,
- the **prompt template** (by content digest, not just by name),
- the **model identifier** and **provider**, plus the generation parameters.

The difference from classical ML is the last item. With a model you host, the weights file is an artifact like any other. With a hosted model behind an API, you usually cannot retrieve the weights, and unless the provider documents a pinned snapshot, it is safer to assume the same name may not always mean the same model. Sampling settings can also make outputs vary between calls.

So be honest about which kind of replay you have. **Guaranteed replay**, re-running and getting the same outputs, is only possible when every input, including the model, is retrievable and the generation is deterministic. Otherwise you have **recorded drift**: you store the inputs and the outputs you actually got, which lets you verify what happened and detect when a re-run differs, but not promise that it will not. Both are useful. Calling the second one "reproducible" is not.

## Retention and security

A manifest is metadata, and metadata leaks.

- Do not put secrets into parameters or environment snapshots. Record *which* credential or config profile was used, never its value. The example records package names and versions, not environment variables.
- Do not copy personal data into run records. Record digests of IDs and counts, not rows. A digest of IDs is still sensitive metadata, though: anyone who holds a candidate ID list can recompute the digest and confirm which population the run used, and customer IDs are rarely secret enough to prevent that. Protect manifests with the same access rules as the data they describe.
- Decide retention for the data versions explicitly. A digest can outlive the data it describes; when a snapshot is deleted (for example, to honour an erasure request), replay of runs that used it becomes impossible by design. Record that as a status rather than letting it surface as a broken link.
- The manifest's own digest (`manifest_sha256` in the example) detects accidental edits. It is **not** authenticity: anyone who can edit the manifest can recompute it. If you need to show that a record has not been tampered with, keep it in write-once storage or sign it with a key the editor does not have.

## The example: a run manifest in standard-library Python

The code below writes a synthetic churn table, records a manifest for a "run" that would consume it, and verifies the data against the manifest. There is no model training: the point is the record, not the model. It uses only the standard library and declares Python 3.12 or newer, which is the version it was written and tested against.

Project layout:

```text
runrecord/
├── pyproject.toml
├── runrecord.py
└── tests/
    └── test_runrecord.py
```

`pyproject.toml` declares the Python requirement and pytest as an optional development dependency. The pytest settings let the tests import `runrecord` without installing the project:

```toml
[project]
name = "runrecord"
version = "0.1.0"
description = "Standard-library run manifest: data digests, split contract, code and environment"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["pytest"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

`runrecord.py`:

```python
"""runrecord: a small, standard-library-only run manifest (Python 3.12+).

Records what a run consumed so it can be checked later: file bytes (sha256),
logical content identity, split membership, the data contract, code state,
environment and seed. Run `python runrecord.py` for a self-contained demo.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ERRORS = {"manifest_tampered", "missing", "content_changed", "split_changed"}


@dataclass(frozen=True)
class DataContract:
    dataset: str  # file name relative to the data root
    key: str
    features: tuple[str, ...]
    target: str
    split_salt: str
    eval_percent: int
    eval_metric: str

    def __post_init__(self) -> None:
        if self.target in self.features or self.key in self.features:
            raise ValueError("key and target must not be listed as features")
        if not 0 < self.eval_percent < 100:
            raise ValueError("eval_percent must be between 1 and 99")


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str


CONTRACT = DataContract(
    dataset="customers.csv",
    key="customer_id",
    features=("tenure_months", "monthly_spend", "support_tickets"),
    target="churned",
    split_salt="split-v1",
    eval_percent=20,
    eval_metric="ROC AUC on the eval split, positive class churned=1",
)


def generate_synthetic(path: Path, *, seed: int, rows: int) -> Path:
    """Deterministic synthetic churn table: same seed and code -> same bytes."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow([CONTRACT.key, *CONTRACT.features, CONTRACT.target])
        for i in range(rows):
            tenure, spend, tickets = rng.randint(1, 72), rng.uniform(10, 120), rng.randint(0, 6)
            churned = int(rng.random() < 0.1 + 0.05 * tickets - 0.001 * tenure)
            out.writerow([f"c{i:05d}", tenure, f"{spend:.2f}", tickets, churned])
    return path


def canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_table(path: Path, key: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = sorted(reader.fieldnames or [])
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            if row[key] in rows:
                raise ValueError(f"duplicate key {row[key]!r}")
            rows[row[key]] = {c: (row[c] or "").strip() for c in columns}
    return columns, rows


def logical_digest(path: Path, key: str) -> str:
    """Identity of the records, ignoring row order, column order and line endings."""
    columns, rows = read_table(path, key)
    body = {"columns": columns, "rows": [[rows[k][c] for c in columns] for k in sorted(rows)]}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def assign_split(record_id: str, *, salt: str, eval_percent: int) -> str:
    """Split from the record ID, never from its position in the file."""
    digest = hashlib.sha256(f"{salt}:{record_id}".encode("utf-8")).digest()
    return "eval" if int.from_bytes(digest[:8], "big") % 100 < eval_percent else "train"


def split_summary(path: Path, contract: DataContract) -> dict[str, dict[str, object]]:
    columns, rows = read_table(path, contract.key)
    missing = {contract.key, contract.target, *contract.features} - set(columns)
    if missing:
        raise ValueError(f"columns required by the contract are missing: {sorted(missing)}")
    ids: dict[str, list[str]] = {"train": [], "eval": []}
    for record_id in sorted(rows):
        ids[assign_split(record_id, salt=contract.split_salt, eval_percent=contract.eval_percent)].append(record_id)
    return {
        name: {"count": len(v), "ids_sha256": hashlib.sha256("\n".join(v).encode()).hexdigest()}
        for name, v in ids.items()
    }


def code_state(repo: Path) -> dict[str, object]:
    """Commit plus a digest of uncommitted changes: a dirty tree is not the commit."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain")
        diff = git("diff", "HEAD", "--binary")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"vcs": None, "note": "not a git checkout; code version unknown"}
    return {
        "vcs": "git",
        "commit": commit,
        "dirty": bool(status.strip()),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "untracked": sorted(line[3:] for line in status.splitlines() if line.startswith("??")),
    }


def environment() -> dict[str, object]:
    packages = sorted(
        {f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata["Name"]}
    )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
    }


def self_digest(manifest: dict) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})).hexdigest()


def build_manifest(
    data_root: Path, *, contract: DataContract, seed: int, params: dict, uri: str | None, code_dir: Path
) -> dict:
    path = data_root / contract.dataset
    manifest = {
        "manifest_version": 1,
        "run": {"seed": seed, "params": params},
        "data": {
            "path": contract.dataset,
            "uri": uri,  # where the immutable copy lives; the digest alone cannot give bytes back
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "logical_sha256": logical_digest(path, contract.key),
        },
        "contract": asdict(contract),
        "split": split_summary(path, contract),
        "code": code_state(code_dir),
        "environment": environment(),
    }
    manifest["manifest_sha256"] = self_digest(manifest)
    return manifest


def verify(manifest: dict, data_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    if manifest.get("manifest_sha256") != self_digest(manifest):
        findings.append(Finding("manifest_tampered", "manifest body does not match its digest"))
    raw = manifest["contract"]
    contract = DataContract(**{**raw, "features": tuple(raw["features"])})
    entry, path = manifest["data"], data_root / manifest["data"]["path"]
    if not path.is_file():
        return findings + [Finding("missing", f"{entry['path']} (expected sha256 {entry['sha256']})")]
    if sha256_file(path) != entry["sha256"]:
        if logical_digest(path, contract.key) == entry["logical_sha256"]:
            findings.append(Finding("bytes_changed_logical_same", "re-serialised file, same records"))
        else:
            findings.append(Finding("content_changed", "records differ from the recorded run"))
    if split_summary(path, contract) != manifest["split"]:
        findings.append(Finding("split_changed", "train/eval membership differs"))
    now, then = environment(), manifest["environment"]
    if (now["python"], now["packages_sha256"]) != (then["python"], then["packages_sha256"]):
        findings.append(Finding("environment_changed", f"python {then['python']} -> {now['python']}"))
    return findings


def report(label: str, findings: list[Finding]) -> None:
    status = "FAIL" if any(f.kind in ERRORS for f in findings) else "OK"
    kinds = ", ".join(sorted(f.kind for f in findings)) or "no findings"
    print(f"{label:<28} {status:<4} {kinds}")


def demo(workdir: Path) -> None:
    data = workdir / "data"
    path = generate_synthetic(data / CONTRACT.dataset, seed=7, rows=500)
    manifest = build_manifest(
        data, contract=CONTRACT, seed=7, params={"C": 1.0},
        uri="s3://example-bucket/churn/customers-v1.csv", code_dir=workdir,
    )
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"sha256         {manifest['data']['sha256']}")
    print(f"logical_sha256 {manifest['data']['logical_sha256']}")
    print(f"split          {manifest['split']['train']['count']} train / {manifest['split']['eval']['count']} eval")
    report("1. unchanged", verify(manifest, data))
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\r\n".join([lines[0], *reversed(lines[1:])]) + "\r\n", encoding="utf-8", newline="")
    report("2. reordered + CRLF", verify(manifest, data))
    path.write_text("\n".join(lines[:-5]) + "\n", encoding="utf-8")
    report("3. last 5 rows dropped", verify(manifest, data))
    path.unlink()
    report("4. file deleted", verify(manifest, data))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        demo(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tmp))
```

A few choices worth pointing out:

- `read_table` rejects duplicate keys. A logical identity keyed by ID would otherwise silently keep the last duplicate and hide a real data problem.
- `verify` separates errors (`missing`, `content_changed`, `split_changed`, `manifest_tampered`) from warnings (`bytes_changed_logical_same`, `environment_changed`). A changed environment does not invalidate data, but you want to see it next to the data verdict.
- The contract is reconstructed from the manifest during verification, so the verifier uses the rules that were recorded rather than whatever the current code defaults are.
- `verify` checks **data and environment only**. It compares the data digests, the split and, for the environment, the Python version and the package-set digest. It records but does not compare the platform, the interpreter implementation, the `code` section, the `uri`, the byte size or `manifest_version`. It does not check out code or replay the run. A changed `code` section with a recomputed manifest digest verifies clean.
- Malformed input raises an exception instead of returning a finding: a missing key column raises `KeyError`, a duplicate key or a missing feature or target column raises `ValueError`, and an invalid contract raises `ValueError` when it is constructed. That is deliberate in a teaching example, because a table that breaks the contract should stop the run loudly. A production verifier would turn these into error findings.
- `verify` expects a manifest written by `build_manifest` with `manifest_version` 1. Other shapes are not supported: a missing section raises `KeyError`, an unknown contract field raises `TypeError`, and `manifest_version` itself is not checked. The data path is joined to the data root as recorded, so only verify manifests from a source you trust.

## Running it

The commands below are the ones used to verify this example. They are a **modern reproduction**: they were run in 2026 with uv and a current Python 3.12 patch release, not in 2024. The date on this post is an editorial label that places it in the series; it does not mean the code was executed on that date. Save the three files in a `runrecord/` directory, then, from its parent directory:

```bash
uv run --directory runrecord --python 3.12 python runrecord.py
uv run --directory runrecord --python 3.12 --extra dev pytest -q
```

uv creates an isolated virtual environment for the project, installs pytest from the `dev` extra for the second command, and runs both commands inside that environment. Note that the example itself does not pin pytest: `pyproject.toml` asks for any version, so you get whatever current release uv resolves. The lock file used to verify this post is kept with the private evidence and is not one of the three files; for a real project, commit a lock file.

`python runrecord.py` runs the demo in a temporary directory: it records a manifest for 500 synthetic rows, then verifies four situations: unchanged data, the same records rewritten in reverse order with CRLF line endings, five rows dropped, and the file deleted. With Python 3.12 the output looks like this:

```text
sha256         a143e587cd511b15c626dce19b0832a802da44b562c3e1989bc4ce8acd155a97
logical_sha256 fadb0f6f7b6ec781563c1ea194e180820473e51eba22c5b519f911ed24c1e981
split          383 train / 117 eval
1. unchanged                 OK   no findings
2. reordered + CRLF          OK   bytes_changed_logical_same
3. last 5 rows dropped       FAIL content_changed, split_changed
4. file deleted              FAIL missing
```

The digests on your machine should match if your Python produces the same random sequence for seed 7; if they do not, the manifest-and-verify logic still behaves the same, and that difference is the reason to record data digests instead of trusting seeds.

The tests cover the cases that matter for a record like this: the same seed gives the same bytes and a different seed a different hash; reordering plus CRLF changes the byte hash but not the logical one; editing a single label changes the logical one; verifying unchanged data against a manifest that went through a JSON round trip produces no findings; missing, tampered and truncated files are reported with the right kinds; an edited manifest is caught; environment drift is reported; the split is stable under row reordering; and the contract rejects a target listed as a feature, missing columns and duplicate keys.

```python
import json
from dataclasses import replace
from pathlib import Path

import pytest

import runrecord as rr


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    rr.generate_synthetic(root / rr.CONTRACT.dataset, seed=7, rows=200)
    return root


@pytest.fixture
def manifest(data_root: Path, tmp_path: Path) -> dict:
    return rr.build_manifest(
        data_root, contract=rr.CONTRACT, seed=7, params={"C": 1.0},
        uri="s3://example-bucket/churn/customers-v1.csv", code_dir=tmp_path,
    )


def rewrite(path: Path, transform, sep: str = "\n") -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_bytes((sep.join(transform(lines)) + sep).encode("utf-8"))


def change_first_label(lines: list[str]) -> list[str]:
    first = lines[1].rsplit(",", 1)
    return [lines[0], f"{first[0]},{1 - int(first[1])}", *lines[2:]]


def kinds(findings: list[rr.Finding]) -> set[str]:
    return {f.kind for f in findings}


def test_same_seed_same_bytes_other_seed_other_hash(tmp_path: Path) -> None:
    a = rr.generate_synthetic(tmp_path / "a.csv", seed=1, rows=50)
    b = rr.generate_synthetic(tmp_path / "b.csv", seed=1, rows=50)
    c = rr.generate_synthetic(tmp_path / "c.csv", seed=2, rows=50)
    assert a.read_bytes() == b.read_bytes()
    assert rr.sha256_file(a) == rr.sha256_file(b) != rr.sha256_file(c)


def test_reordered_crlf_changes_bytes_not_logical_identity(data_root: Path) -> None:
    path = data_root / rr.CONTRACT.dataset
    before = (rr.sha256_file(path), rr.logical_digest(path, "customer_id"))
    rewrite(path, lambda ls: [ls[0], *reversed(ls[1:])], sep="\r\n")
    assert rr.sha256_file(path) != before[0]
    assert rr.logical_digest(path, "customer_id") == before[1]


def test_changed_value_changes_logical_identity(data_root: Path) -> None:
    path = data_root / rr.CONTRACT.dataset
    before = rr.logical_digest(path, "customer_id")
    rewrite(path, change_first_label)
    assert rr.logical_digest(path, "customer_id") != before


def test_replay_of_unchanged_data_is_clean(manifest: dict, data_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(manifest))
    loaded = json.loads(out.read_text())
    assert loaded["code"]["vcs"] is None  # tmp_path is not a git checkout
    assert rr.verify(loaded, data_root) == []


def test_missing_file(manifest: dict, data_root: Path) -> None:
    (data_root / rr.CONTRACT.dataset).unlink()
    findings = rr.verify(manifest, data_root)
    assert kinds(findings) == {"missing"}
    assert manifest["data"]["sha256"] in findings[0].detail


def test_tampered_value_is_a_content_change(manifest: dict, data_root: Path) -> None:
    rewrite(data_root / rr.CONTRACT.dataset, change_first_label)
    assert kinds(rr.verify(manifest, data_root)) == {"content_changed"}


def test_reserialised_file_is_a_warning(manifest: dict, data_root: Path) -> None:
    rewrite(data_root / rr.CONTRACT.dataset, lambda ls: ls, sep="\r\n")
    findings = rr.verify(manifest, data_root)
    assert kinds(findings) == {"bytes_changed_logical_same"}
    assert not kinds(findings) & rr.ERRORS


def test_dropped_rows_change_content_and_split(manifest: dict, data_root: Path) -> None:
    rewrite(data_root / rr.CONTRACT.dataset, lambda ls: ls[:-10])
    assert kinds(rr.verify(manifest, data_root)) == {"content_changed", "split_changed"}


def test_edited_manifest_is_detected(manifest: dict, data_root: Path) -> None:
    manifest["run"]["seed"] = 8
    assert kinds(rr.verify(manifest, data_root)) == {"manifest_tampered"}


def test_environment_drift_is_reported(manifest: dict, data_root: Path) -> None:
    manifest["environment"]["python"] = "3.11.0"
    manifest["manifest_sha256"] = rr.self_digest(manifest)
    assert kinds(rr.verify(manifest, data_root)) == {"environment_changed"}


def test_split_is_stable_and_order_independent(data_root: Path) -> None:
    path = data_root / rr.CONTRACT.dataset
    before = rr.split_summary(path, rr.CONTRACT)
    rewrite(path, lambda ls: [ls[0], *reversed(ls[1:])])
    assert rr.split_summary(path, rr.CONTRACT) == before
    assert before["train"]["count"] + before["eval"]["count"] == 200
    assert before["eval"]["count"] > 0


def test_contract_rejects_target_as_feature() -> None:
    with pytest.raises(ValueError):
        replace(rr.CONTRACT, features=("tenure_months", "churned"))


def test_contract_columns_must_exist(data_root: Path) -> None:
    contract = replace(rr.CONTRACT, features=("tenure_months", "region"))
    with pytest.raises(ValueError, match="region"):
        rr.split_summary(data_root / rr.CONTRACT.dataset, contract)


def test_duplicate_keys_are_rejected(data_root: Path) -> None:
    path = data_root / rr.CONTRACT.dataset
    rewrite(path, lambda ls: [*ls, ls[1]])
    with pytest.raises(ValueError, match="duplicate key"):
        rr.logical_digest(path, "customer_id")


def test_demo_runs_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rr.demo(tmp_path)
    rows = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    assert rows[3:] == [
        "1. unchanged OK no findings",
        "2. reordered + CRLF OK bytes_changed_logical_same",
        "3. last 5 rows dropped FAIL content_changed, split_changed",
        "4. file deleted FAIL missing",
    ]
```

## What this example does not do

It records one dataset per run, it does not upload anything, and it does not record output artifacts. Adding them is the same pattern (path, URI, digest) applied to what the run wrote. It does not sign manifests. `code_state` is exercised by the tests only outside a git checkout. It is a teaching-sized record, not a tracking system.

That is the job of the next article: the same fields, recorded through MLflow runs and artifacts, so that experiments are searchable and comparable without hand-written JSON.

## Checklist

Before calling a result reproducible, check that the run record contains:

1. Commit ID, dirty flag and a digest of uncommitted changes.
2. For every input: an immutable URI **and** a content digest, plus a logical digest where re-serialisation is expected.
3. Key, features, target, split rule with salt, split ID digests and counts, and the metric definition.
4. Interpreter, platform and package set (or lock file digest).
5. Seeds and a note of known non-deterministic steps.
6. For every output artifact: a URI and a digest.
7. For LLM pipelines: corpus, index, prompt, model and provider identifiers, and whether replay is guaranteed or only drift is recorded.
8. No secrets, no raw personal data, a retention decision, and tamper evidence that does not depend on the editor's goodwill.
