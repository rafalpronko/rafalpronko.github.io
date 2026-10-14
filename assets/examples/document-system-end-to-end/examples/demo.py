"""End-to-end demo on synthetic data: ingest, replay, update, delete, search,
evaluate, then one real HTTP roundtrip against a loopback server.

Run from the project root:  uv run python examples/demo.py
"""

import json
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from docsys.api import make_server  # noqa: E402
from docsys.evaluate import evaluate  # noqa: E402
from docsys.schema import connect  # noqa: E402
from docsys.search import search  # noqa: E402
from docsys.store import VersionConflict, delete_document, upsert_document  # noqa: E402


def load_json(name):
    return json.loads((HERE / "fixtures" / name).read_text())


def ingest(conn, corpus):
    for doc in corpus:
        status = upsert_document(conn, doc["doc_id"], doc["version"], doc["title"], doc["body"])
        print(f"ingest {doc['doc_id']} v{doc['version']}: {status}")


def lifecycle(conn, corpus):
    first = corpus[0]
    args = (first["doc_id"], first["version"], first["title"], first["body"])
    print("replay same version:", upsert_document(conn, *args))
    try:
        upsert_document(conn, first["doc_id"], 1, first["title"], "edited body")
    except VersionConflict as error:
        print("conflicting v1:", error)
    print("delete faq-oncall v2:", delete_document(conn, "faq-oncall", 2))
    print("stale replay faq-oncall v1:", upsert_document(conn, "faq-oncall", 1, "x", "on-call"))


def show_search(conn, query):
    result = search(conn, query, limit=3)
    print(f"\nsearch {query!r} -> match {result['query']} (corpus_version={result['corpus_version']})")
    for hit in result["hits"]:
        print(f"  {hit['score']:>9.6f}  {hit['doc_id']}#{hit['chunk_no']} v{hit['version']}  {hit['text'][:60]}")
    if not result["hits"]:
        print("  (no hits)")


def http_roundtrip(db_path):
    """One GET against a loopback server on an ephemeral port (port=0)."""
    server = make_server(db_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}/search?q=rolled+back&limit=1"
        with urllib.request.urlopen(url, timeout=5) as response:
            print(f"\nHTTP GET {url} -> {response.status}")
            print(json.dumps(json.loads(response.read()), indent=2))
    finally:
        server.shutdown()
        server.server_close()  # closes the socket and the SQLite connection
        thread.join(timeout=5)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "demo.db"
        conn = connect(db_path)
        try:
            run_offline(conn)
        finally:
            conn.close()
        http_roundtrip(db_path)


def run_offline(conn):
    corpus = load_json("corpus.json")
    ingest(conn, corpus)
    lifecycle(conn, corpus)
    for query in ("restore snapshot", "on-call", "oncall",
                  "kubernetes autoscaling"):
        try:
            show_search(conn, query)
        except ValueError as error:
            print(f"\nsearch {query!r} rejected: {error}")
    cases = [case for case in load_json("queries.json")
             if "escalate" not in case["query"]]
    print("\nevaluation (faq-oncall deleted, its case removed):")
    print(json.dumps(evaluate(conn, cases, k=3), indent=2))


if __name__ == "__main__":
    main()
