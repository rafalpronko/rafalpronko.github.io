import json
import threading
import urllib.error
import urllib.request

import sqlite3
from http.server import HTTPServer

import pytest

from docsys import api
from docsys.api import DocServer, make_server
from docsys.schema import connect


@pytest.fixture
def server(tmp_path):
    server = make_server(tmp_path / "api.db", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def base_url(server):
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def call(method, url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_server_refuses_non_loopback_binding(tmp_path):
    with pytest.raises(ValueError):
        make_server(tmp_path / "x.db", host="0.0.0.0", port=0)


def test_put_search_delete_roundtrip(base_url):
    doc = {"version": 1, "title": "Backup", "body": "Restore drills\n\nquarterly snapshot"}
    assert call("PUT", f"{base_url}/documents/runbook", doc) == (200, {"status": "inserted"})
    assert call("PUT", f"{base_url}/documents/runbook", doc) == (200, {"status": "unchanged"})

    status, body = call("GET", f"{base_url}/search?q=snapshot&limit=2")
    assert status == 200
    assert body["hits"][0]["doc_id"] == "runbook"
    assert body["corpus_version"] == 1

    assert call("DELETE", f"{base_url}/documents/runbook?version=2") == (200, {"status": "deleted"})
    assert call("GET", f"{base_url}/search?q=snapshot")[1]["hits"] == []


def test_api_errors_are_json_with_status_codes(base_url):
    doc = {"version": 1, "title": "T", "body": "alpha"}
    call("PUT", f"{base_url}/documents/d1", doc)
    conflict = call("PUT", f"{base_url}/documents/d1", {**doc, "body": "beta"})
    assert conflict[0] == 409 and "error" in conflict[1]
    assert call("GET", f"{base_url}/search?q=back*")[0] == 400
    assert call("GET", f"{base_url}/search")[0] == 400
    assert call("GET", f"{base_url}/search?q=alpha&limit=500")[0] == 400
    assert call("PUT", f"{base_url}/documents/d2", {"version": "1"})[0] == 400
    assert call("DELETE", f"{base_url}/documents/d1")[0] == 400
    assert call("GET", f"{base_url}/nope")[0] == 404


def test_malformed_json_is_rejected(base_url):
    request = urllib.request.Request(
        f"{base_url}/documents/d1", data=b"{not json", method="PUT"
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 400


def test_version_outside_sqlite_integer_range_is_400(base_url):
    url = f"{base_url}/documents/d1"
    too_big = {"version": 2**63, "title": "T", "body": "x"}
    assert call("PUT", url, too_big)[0] == 400
    largest = {**too_big, "version": 2**63 - 1}
    assert call("PUT", url, largest) == (200, {"status": "inserted"})
    assert call("DELETE", f"{url}?version={2**63}")[0] == 400
    assert call("GET", f"{base_url}/search?q=x")[1]["corpus_version"] == 1


def test_encoded_slash_in_doc_id_is_rejected(base_url):
    doc = {"version": 1, "title": "T", "body": "x"}
    assert call("PUT", f"{base_url}/documents/a%2Fb", doc)[0] == 404


def test_server_close_closes_sqlite_connection(tmp_path):
    server = make_server(tmp_path / "close.db", port=0)
    conn = server.conn
    server.server_close()
    server.server_close()  # idempotent
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_socket_is_closed_when_database_cannot_open(tmp_path, monkeypatch):
    closed = []
    original = HTTPServer.server_close

    def record_close(self):
        closed.append(self)
        original(self)

    def broken_connect(path):
        raise sqlite3.OperationalError("no FTS5")

    monkeypatch.setattr(HTTPServer, "server_close", record_close)
    monkeypatch.setattr(api, "connect", broken_connect)
    with pytest.raises(sqlite3.OperationalError):
        make_server(tmp_path / "broken.db", port=0)
    assert len(closed) == 1 and closed[0].socket.fileno() == -1


@pytest.mark.parametrize("step", ["server_bind", "server_activate"])
def test_bind_or_activate_error_is_not_masked(tmp_path, monkeypatch, step):
    """HTTPServer.__init__ calls server_close() itself when binding fails."""
    sockets = []
    original = getattr(DocServer, step)

    def fail(self):
        sockets.append(self.socket)
        original(self)  # server_bind binds an OS-chosen port (port=0)
        raise OSError("simulated " + step + " failure")

    monkeypatch.setattr(DocServer, step, fail)
    with pytest.raises(OSError, match="simulated " + step):
        make_server(tmp_path / "bind.db", port=0)
    assert sockets[0].fileno() == -1
    assert not (tmp_path / "bind.db").exists()


def test_failed_rollback_makes_api_fail_closed(server, base_url, tmp_path, failing_commit):
    url = f"{base_url}/documents/d1"
    doc = {"version": 1, "title": "T", "body": "alpha"}
    assert call("PUT", url, doc) == (200, {"status": "inserted"})
    real = server.conn
    server.conn = failing_commit(real, fail_rollback=True)

    assert call("PUT", url, {**doc, "version": 2, "body": "beta"}) == (
        500, {"error": "internal error"})
    with pytest.raises(sqlite3.ProgrammingError):
        real.execute("SELECT 1")
    unavailable = (503, {"error": "database unavailable"})
    assert call("GET", f"{base_url}/health") == unavailable
    assert call("GET", f"{base_url}/search?q=alpha") == unavailable
    assert call("PUT", f"{base_url}/documents/d2", doc) == unavailable
    assert call("DELETE", f"{url}?version=3") == unavailable

    check = connect(tmp_path / "api.db")
    try:
        row = check.execute("SELECT doc_id, version FROM documents").fetchall()
        assert [tuple(r) for r in row] == [("d1", 1)]
        corpus = check.execute("SELECT value FROM meta").fetchone()[0]
        assert corpus == 1
    finally:
        check.close()


def test_failed_commit_with_successful_rollback_keeps_api_usable(
        server, base_url, failing_commit):
    url = f"{base_url}/documents/d1"
    doc = {"version": 1, "title": "T", "body": "alpha"}
    server.conn = failing_commit(server.conn)
    assert call("PUT", url, doc) == (500, {"error": "internal error"})
    assert call("GET", f"{base_url}/health") == (200, {"status": "ok"})
    assert call("PUT", url, doc) == (200, {"status": "inserted"})
    assert call("GET", f"{base_url}/search?q=alpha")[1]["corpus_version"] == 1
