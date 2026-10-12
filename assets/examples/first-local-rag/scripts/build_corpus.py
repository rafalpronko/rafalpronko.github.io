"""Download RFC 9110 and cut three sections into unmodified local extracts.

The extracts are byte-for-byte copies of whole lines from the plain-text RFC.
Run from the examples directory: python scripts/build_corpus.py
Only the standard library is used.
"""

import datetime as dt
import hashlib
import json
import pathlib
import re
import urllib.request

RFC_TXT_URL = "https://www.rfc-editor.org/rfc/rfc9110.txt"
RFC_HTML_URL = "https://www.rfc-editor.org/rfc/rfc9110.html"
SECTIONS = ["9.2.2", "9.3.1", "9.3.2"]
CORPUS_DIR = pathlib.Path(__file__).resolve().parent.parent / "corpus"
HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.  (\S.*)$")
ATTRIBUTION = (
    "Unmodified extract from RFC 9110 (HTTP Semantics), IETF, June 2022. "
    "Copyright (c) 2022 IETF Trust and the persons identified as the "
    "document authors. All rights reserved. Reproduced under Section "
    "3.c.iii of the IETF Trust Legal Provisions."
)


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "first-local-rag/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def find_section(lines: list[str], number: str) -> tuple[int, int, str]:
    """Return (first_line, last_line, title); 1-based, inclusive, no trailing blanks."""
    start = title = None
    for index, line in enumerate(lines):
        match = HEADING.match(line)
        if not match:
            continue
        if start is None and match.group(1) == number:
            start, title = index, match.group(2)
        elif start is not None:
            end = index
            while not lines[end - 1].strip():
                end -= 1
            return start + 1, end, title
    raise ValueError(f"section {number} not found")


def build(raw: bytes, retrieved_utc: str) -> list[dict]:
    lines = raw.decode("utf-8").splitlines(keepends=True)
    records = []
    for number in SECTIONS:
        first, last, title = find_section(lines, number)
        extract = "".join(lines[first - 1:last]).encode("utf-8")
        if b"\f" in extract:
            raise ValueError(f"page break inside section {number}")
        name = f"rfc9110-section-{number}.txt"
        (CORPUS_DIR / name).write_bytes(extract)
        records.append({
            "id": f"rfc9110-{number}",
            "file": name,
            "title": f"RFC 9110, Section {number}. {title}",
            "url": f"{RFC_HTML_URL}#section-{number}",
            "source_text_url": RFC_TXT_URL,
            "source_text_sha256": hashlib.sha256(raw).hexdigest(),
            "source_lines": [first, last],
            "sha256": hashlib.sha256(extract).hexdigest(),
            "retrieved_utc": retrieved_utc,
            "attribution": ATTRIBUTION,
        })
    return records


def main() -> None:
    CORPUS_DIR.mkdir(exist_ok=True)
    retrieved = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    records = build(download(RFC_TXT_URL), retrieved)
    manifest = {"documents": records}
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for record in records:
        print(record["id"], record["source_lines"], record["sha256"])


if __name__ == "__main__":
    main()
