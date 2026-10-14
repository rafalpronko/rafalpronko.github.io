"""Minimal local JSON API on http.server.

http.server is documented as not recommended for production: there is no
authentication, TLS, request limits beyond what is coded here, or concurrency
tuning. The server refuses to bind anything but loopback.

  GET    /health
  GET    /search?q=terms&limit=5
  PUT    /documents/<doc_id>            {"version": 1, "title": "...", "body": "..."}
  DELETE /documents/<doc_id>?version=2

If a failed request leaves the SQLite connection unusable, the server closes
it and answers 503 to /health and every database request until restarted.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from .query import QueryError
from .schema import close_quietly, connect, is_usable
from .search import search
from .store import VersionConflict, delete_document, upsert_document

LOOPBACK = {"127.0.0.1", "localhost"}
MAX_BODY_BYTES = 1_000_000


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class DocServer(HTTPServer):
    """Single-threaded server that owns its socket and one SQLite connection.

    server_close() closes both and can be called more than once. If binding
    or opening the database fails, the socket is closed and the original
    error propagates. A connection left closed or inside a transaction after
    a failed request is discarded; database requests then get 503.
    """

    def __init__(self, address: tuple[str, int], db_path) -> None:
        # HTTPServer.__init__ calls server_close() itself if binding fails.
        self.conn = None
        super().__init__(address, Handler)
        try:
            self.conn = connect(db_path)
        except BaseException:
            try:
                self.server_close()
            except Exception:
                pass  # keep the original error
            raise

    def database(self):
        if self.conn is None:
            raise ApiError(503, "database unavailable")
        return self.conn

    def discard_unusable_connection(self) -> None:
        """Fail closed: never reuse a connection a failed request broke."""
        if self.conn is not None and not is_usable(self.conn):
            conn, self.conn = self.conn, None
            close_quietly(conn)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            conn, self.conn = self.conn, None
            if conn is not None:
                conn.close()


def make_server(db_path, host: str = "127.0.0.1",
                port: int = 8780) -> DocServer:
    """Create a loopback-only server; port=0 picks a free ephemeral port."""
    if host not in LOOPBACK:
        raise ValueError("this example only binds to loopback")
    return DocServer((host, port), db_path)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._dispatch(self._get)

    def do_PUT(self):
        self._dispatch(self._put)

    def do_DELETE(self):
        self._dispatch(self._delete)

    def log_message(self, format, *args):
        pass  # keep demo and test output readable

    def _dispatch(self, route):
        try:
            status, payload = route(urlsplit(self.path))
        except ApiError as error:
            status, payload = error.status, {"error": str(error)}
        except VersionConflict as error:
            status, payload = 409, {"error": str(error)}
        except (QueryError, ValueError) as error:
            status, payload = 400, {"error": str(error)}
        except Exception:
            status, payload = 500, {"error": "internal error"}
        self.server.discard_unusable_connection()
        self._send(status, payload)

    def _get(self, url):
        if url.path == "/health":
            self.server.database()  # 503 once the connection is discarded
            return 200, {"status": "ok"}
        if url.path != "/search":
            raise ApiError(404, "not found")
        conn = self.server.database()
        params = parse_qs(url.query)
        query = _single(params, "q")
        limit = _int_param(params.get("limit", ["5"])[0], "limit")
        return 200, search(conn, query, limit)

    def _put(self, url):
        doc_id = _doc_id(url.path)
        conn = self.server.database()
        body = self._read_json()
        args = (body.get("version"), body.get("title"), body.get("body"))
        return 200, {"status": upsert_document(conn, doc_id, *args)}

    def _delete(self, url):
        doc_id = _doc_id(url.path)
        conn = self.server.database()
        version = _int_param(_single(parse_qs(url.query), "version"), "version")
        return 200, {"status": delete_document(conn, doc_id, version)}

    def _read_json(self) -> dict:
        length = _int_param(self.headers.get("Content-Length", "0"), "Content-Length")
        if length > MAX_BODY_BYTES:
            raise ApiError(413, "request body too large")
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError(400, "body must be valid JSON") from None
        if not isinstance(body, dict):
            raise ApiError(400, "body must be a JSON object")
        return body

    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _doc_id(path: str) -> str:
    """Decode the id first, so an encoded slash (%2F) is rejected too."""
    prefix = "/documents/"
    doc_id = unquote(path[len(prefix):])
    if not path.startswith(prefix) or "/" in doc_id:
        raise ApiError(404, "not found")
    return doc_id


def _single(params, name) -> str:
    values = params.get(name)
    if not values or len(values) != 1:
        raise ApiError(400, f"exactly one '{name}' parameter is required")
    return values[0]


def _int_param(value, name) -> int:
    if not (value.isascii() and value.isdigit()):
        raise ApiError(400, f"'{name}' must be a non-negative integer")
    return int(value)
