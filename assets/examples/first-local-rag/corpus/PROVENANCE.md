# Corpus provenance

The corpus contains three **unmodified** extracts from RFC 9110, *HTTP Semantics*
(IETF, Internet Standard, June 2022):

| File | Section | Lines in `rfc9110.txt` | SHA-256 of the extract |
|---|---|---|---|
| `rfc9110-section-9.2.2.txt` | 9.2.2. Idempotent Methods | 3861–3904 | `e18335c986fb3acb13939b61e2c3a2d36efb0e6566b9bb4ce5e7db4a18f51075` |
| `rfc9110-section-9.3.1.txt` | 9.3.1. GET | 3920–3983 | `439844c082fe493817e9302837149c1d00925c544becc2dc871690668b1d5ab2` |
| `rfc9110-section-9.3.2.txt` | 9.3.2. HEAD | 3985–4023 | `353472567d7553be5d3cb1e4da80689caa948cd64ad7988906dfb10a74b51752` |

- Source text: <https://www.rfc-editor.org/rfc/rfc9110.txt>
- Section links: <https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2>,
  `#section-9.3.1`, `#section-9.3.2`
- `scripts/build_corpus.py` downloads the plain-text RFC and copies whole lines,
  byte for byte, from each section heading up to the line before the next
  heading (trailing blank lines removed). `manifest.json` records the line range,
  the SHA-256 of the full downloaded RFC text and of each extract, and the
  retrieval time.
- The program verifies every extract against `manifest.json` before use and
  refuses to start if a file was changed.
- For retrieval and prompting the code collapses line wrapping and indentation
  into single spaces **in memory**. The words are unchanged; the files on disk
  stay identical to the RFC text.

## Licence and attribution

Copyright (c) 2022 IETF Trust and the persons identified as the document
authors. All rights reserved.

These are unmodified portions of an IETF Document, attributed to the IETF and
to RFC 9110, reproduced under Section 3.c.iii of the IETF Trust Legal
Provisions (TLP 5.0): <https://trustee.ietf.org/assets/trust-legal-provisions/>.
The three sections are a small fraction of the RFC. They are not modified and
are not a substitute for the full specification:
<https://www.rfc-editor.org/rfc/rfc9110.html>.
