"""Walidacja strony Jekyll: statyczna oraz (opcjonalnie) na zbudowanym wyjściu.

Uruchomienie: uv run --with pyyaml python -m unittest discover -s tests -v
Testy wyrenderowanej strony: RENDERED_SITE=/ścieżka/do/_site (bez tej zmiennej są pomijane).
"""
import ast
import hashlib
import html
import os
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent
POST = "_posts/2023-10-12-pipeline-idempotencja.md"
# Oczekiwany inwentarz wpisów: plik -> (permalink, tytuł). Nowy wpis = świadoma zmiana tej listy.
POSTS = {
    "2023-10-12-pipeline-idempotencja.md": ("/blog/pipeline-idempotencja/", "A pipeline you can run twice. Idempotency in practice"),
    "2023-10-19-data-contracts-validation.md": ("/blog/data-contracts-validation/", "Why a model cannot fix bad data: contracts and validation"),
    "2023-10-26-dlt-api-duckdb-rerun.md": ("/blog/dlt-api-duckdb-rerun/", "A small API-to-database pipeline with dlt — and a rerun test"),
    "2023-11-02-incremental-loading-merge-schema-evolution.md": ("/blog/incremental-loading-merge-schema-evolution/", "Incremental loading, merge and schema evolution: where correctness breaks"),
    "2023-11-09-quality-gate-before-migration.md": ("/blog/quality-gate-before-migration/", "A quality gate before migration: checking the data, not the job status"),
    "2023-11-16-train-validation-test-group-leakage.md": ("/blog/train-validation-test-group-leakage/", "Train, validation and test: how group leakage inflates your score"),
    "2023-11-23-baseline-lightgbm.md": ("/blog/baseline-lightgbm/", "A fair baseline for LightGBM: freeze the split before comparing models"),
    "2023-11-30-precision-recall-ap-calibration.md": ("/blog/precision-recall-ap-calibration/", "Precision, recall, AP and calibration answer different questions"),
    "2023-12-07-optuna-without-test-leakage.md": ("/blog/optuna-without-test-leakage/", "Tuning with Optuna without tuning on the test set"),
    "2023-12-14-tokens-embeddings-cosine-similarity.md": ("/blog/tokens-embeddings-cosine-similarity/", "Tokens, embeddings, and cosine similarity—without the magic"),
    "2023-12-21-first-local-rag.md": ("/blog/first-local-rag/", "Your first local RAG: public documents and traceable citations"),
    "2023-12-28-chunking-hybrid-search-reranking.md": ("/blog/chunking-hybrid-search-reranking/", "Chunking, hybrid search and reranking: a controlled comparison"),
    "2024-01-04-rag-corpus-dlt-dbt-index-reconciliation.md": ("/blog/rag-corpus-dlt-dbt-index-reconciliation/", "Keeping a RAG corpus current: dlt, dbt and index reconciliation, deletions included"),
    "2024-01-11-workflow-vs-agent.md": ("/blog/workflow-vs-agent/", "Workflow vs. agent: when not to let the model decide"),
    "2024-01-18-tool-calling-validation-timeouts-retries.md": ("/blog/tool-calling-validation-timeouts-retries/", "Tool calling with validation, timeouts, and bounded retries"),
    "2024-01-25-state-checkpoints-recovery.md": ("/blog/state-checkpoints-recovery/", "State, checkpoints and recovery: what must survive a restart"),
    "2024-02-08-python-mutability-aliasing.md": ("/blog/python-mutability-aliasing/", "Mutability and aliasing: the data bug your types won't catch"),
    "2024-02-15-pydantic-input-contracts.md": ("/blog/pydantic-input-contracts/", "Pydantic input contracts: valid JSON is not a valid decision"),
    "2024-02-29-ranking-is-not-eligibility.md": ("/blog/ranking-is-not-eligibility/", "Ranking is not eligibility: put hard constraints before scores"),
    "2024-03-07-q-k-v-attention-small-matrix.md": ("/blog/q-k-v-attention-small-matrix/", "Queries, keys and values: attention on a small matrix"),
    "2024-03-14-causal-attention-pytorch.md": ("/blog/causal-attention-pytorch/", "Causal attention in PyTorch: a small implementation with shape and leakage tests"),
    "2024-03-21-prefill-decode-kv-cache.md": ("/blog/prefill-decode-kv-cache/", "Prefill, decode and the KV cache: where LLM latency comes from"),
    "2024-03-28-model-selection-quality-latency.md": ("/blog/model-selection-quality-latency/", "Choosing a model for the task: benchmark quality and latency, not marketing labels"),
    "2024-04-04-sql-pandas-join-grain.md": ("/blog/sql-pandas-join-grain/", "SQL and pandas: joins, grain, and unexpected row multiplication"),
    "2024-04-11-dbt-staging-intermediate-marts.md": ("/blog/dbt-staging-intermediate-marts/", "dbt staging, intermediate and marts: a layered project you can run and break"),
    "2024-04-18-backfills-late-data-schema-history.md": ("/blog/backfills-late-data-schema-history/", "Backfills, late data and schema changes without silently losing history"),
    "2024-04-25-source-target-reconciliation.md": ("/blog/source-target-reconciliation/", "Source-to-target reconciliation: row counts are not enough"),
    "2024-05-02-gradients-autograd.md": ("/blog/gradients-autograd/", "Gradients from scratch to autograd: what actually flows backward"),
    "2024-05-09-tfidf-vs-embedding-classifier.md": ("/blog/tfidf-vs-embedding-classifier/", "Text classification: TF-IDF versus a small embedding model"),
    "2024-05-16-train-eval-padding-checkpoints-seeds.md": ("/blog/train-eval-padding-checkpoints-seeds/", "Train/eval, padding, checkpoints and seeds: four experiment traps"),
    "2024-05-23-ner-token-vs-entity-evaluation.md": ("/blog/ner-token-vs-entity-evaluation/", "NER evaluation: token scores, whole entities and boundary errors"),
    "2024-05-30-rag-evaluation-three-layers.md": ("/blog/rag-evaluation-three-layers/", "Evaluate retrieval, answer quality, and evidence support separately"),
    "2024-06-06-rag-test-set-regression.md": ("/blog/rag-test-set-regression/", "Building a RAG Test Set: Questions, References, and Regression Reports"),
    "2024-06-13-llm-as-judge.md": ("/blog/llm-as-judge/", "LLM-as-Judge: Rubrics, Bias, Answer Order, and Human Oversight"),
    "2024-06-20-dspy-vs-manual-prompts.md": ("/blog/dspy-vs-manual-prompts/", "DSPy vs hand-written prompts: a fair comparison with a holdout and a fixed budget"),
    "2024-06-27-prompting-rag-fine-tuning.md": ("/blog/prompting-rag-fine-tuning/", "Prompting, RAG, or Fine-Tuning: Start with the Failure"),
    "2024-07-04-lora-baseline-separate-evaluation.md": ("/blog/lora-baseline-separate-evaluation/", "A small LoRA experiment with a baseline and a separate evaluation"),
    "2024-07-11-qlora-quantization-memory-quality.md": ("/blog/qlora-quantization-memory-quality/", "QLoRA and quantization: the memory–quality trade-off"),
    "2024-07-18-batching-serving-under-load.md": ("/blog/batching-serving-under-load/", "Batching and serving: measure throughput, latency, and failures under load"),
    "2024-07-25-experiments-artifacts-data-versions.md": ("/blog/experiments-artifacts-data-versions/", "Experiments, artifacts and data versions: what to record"),
    "2024-08-01-reproducible-mlflow-pipeline.md": ("/blog/reproducible-mlflow-pipeline/", "A reproducible ML pipeline with MLflow and a regression test"),
    "2024-08-08-data-drift-concept-drift-answer-degradation.md": ("/blog/data-drift-concept-drift-answer-degradation/", "Data Drift, Concept Drift, and Answer Degradation Are Different Problems"),
    "2024-08-15-ci-quality-gates.md": ("/blog/ci-quality-gates/", "A green build is not a release decision: quality gates for ML in CI"),
    "2024-08-22-freshness-and-approval.md": ("/blog/freshness-and-approval/", "Freshness and approval: why a valid command can become unsafe"),
    "2024-08-29-device-command-journal.md": ("/blog/device-command-journal/", "After the timeout: a command journal, correlation IDs and read-after-write on a simulated device"),
    "2024-09-05-prompt-injection-trust-boundaries.md": ("/blog/prompt-injection-trust-boundaries/", "Prompt injection, tool permissions and trust boundaries"),
    "2024-09-12-human-in-the-loop-recovery.md": ("/blog/human-in-the-loop-recovery/", "Human-in-the-loop state machines: rollback and reconciliation after a crash"),
    "2024-09-19-building-a-benchmark.md": ("/blog/building-a-benchmark/", "Building a benchmark: baselines, constraints, and fair comparisons"),
    "2024-09-26-greedy-packing-local-improvement.md": ("/blog/greedy-packing-local-improvement/", "Greedy packing versus local improvement: measure, don't declare victory"),
    "2024-10-03-graph-rag-multimodal-documents.md": ("/blog/graph-rag-multimodal-documents/", "Graph RAG and multimodal documents: when the extra complexity pays off"),
    "2024-10-10-document-system-end-to-end.md": ("/blog/document-system-end-to-end/", "A document system end to end: ingestion, search, API and evaluation"),
    # Data = faktyczny dzień przygotowania: Jev jest publicznie udokumentowany dopiero od 2026 r.
    "2026-10-12-model-routing-jev.md": ("/blog/model-routing-jev/", "Model routing with Jev: allowlists, capability guards, and rollback"),
}
# Wpisy bez nagłówka H1 w treści — tytuł renderuje layout post.html (zrecenzowanej treści nie zmieniamy).
TITLE_FROM_LAYOUT = {
    "2024-01-18-tool-calling-validation-timeouts-retries.md",
    "2024-05-09-tfidf-vs-embedding-classifier.md",
    "2024-07-18-batching-serving-under-load.md",
    "2024-07-25-experiments-artifacts-data-versions.md",
}
# Celowe polskie słowa w angielskich wpisach (waluta, nazwa miasta, przykładowe wejście odrzucane przez parser).
POLISH_WORDS_ALLOWED = ("złoty", "Gdańsk", "zażółć")
# Pliki z nazwą zaczynającą się od "_" lub "." publikowane w przykładach (Jekyll wymaga ich w include).
HIDDEN_INCLUDES = {"__init__.py", "__main__.py", ".python-version"}
EXAMPLE_DIRS = ("assets/examples", "assets/code")
# Dozwolony ładunek publiczny w assets/ (ścieżki względne, wyrażenia regularne).
ASSET_ALLOWLIST = [
    r"assets/css/hud\.css",
    r"assets/images/[a-z0-9-]+\.svg",
    r"assets/downloads/[a-z0-9-]+\.zip",
    r"assets/(examples|code)/[a-z0-9-]+\.zip",
    r"assets/(examples|code)/[a-z0-9-]+/([a-z0-9_]+/)*"
    r"([A-Za-z0-9_.-]+\.(py|toml|lock|md|txt|json|jsonl|csv|sh|npz|bin|zip)|\.python-version|SHA256SUMS)",
]
FORBIDDEN_PATH = re.compile(r"(__pycache__|\.pyc$|\.venv|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.DS_Store|"
                            r"\.ipynb_checkpoints|(^|/)\.env$|\.sandbox|node_modules|\.hermes|\.remember)")
SECRET = re.compile(r"(/Users/|\.hermes|BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|"
                    r"sk-ant-|sk-[A-Za-z0-9]{20,}|ANTHROPIC_API_KEY|OPENAI_API_KEY)")
RELATIVE_URL = re.compile(r"\{\{\s*'(/[^']*)'\s*\|\s*relative_url\s*\}\}")
# Ostatni commit z polską wersją artykułu — stały punkt odniesienia dla tłumaczenia.
BASELINE = "ff2e6c338f812937fefd293dc1bb066a5c439fc3"
POLISH = r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]"
TEMPLATES = ["_layouts/default.html", "_layouts/home.html", "_layouts/post.html", "_layouts/example-doc.html",
             "_includes/date-pl.html"]
EXAMPLE_LAYOUT = "_layouts/example-doc.html"
VOID = {"meta", "link", "br", "img", "input", "hr", "source"}


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


def strip_code(text: str) -> str:
    """Treść bez bloków ``` (komentarze `# ...` w kodzie to nie nagłówki)."""
    return re.sub(r"^```.*?^```$", "", text, flags=re.S | re.M)


def strip_liquid(text: str) -> str:
    text = re.sub(r"\{%-?.*?-?%\}", "", text, flags=re.S)
    return re.sub(r"\{\{-?.*?-?\}\}", "X", text, flags=re.S)


class TagBalance(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        ident = dict(attrs).get("id")
        if ident:
            self.ids.append(ident)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}>, open: {self.stack[-3:]}")
        else:
            self.stack.pop()


def parse(rel: str) -> TagBalance:
    parser = TagBalance()
    parser.feed(strip_liquid(front_matter(read(rel))[1]))
    parser.close()
    return parser


class ConfigTest(unittest.TestCase):
    def test_config(self):
        cfg = yaml.safe_load(read("_config.yml"))
        self.assertEqual(cfg["title"], "Rafał Prońko")
        self.assertEqual(cfg["url"], "https://rafalpronko.github.io")
        self.assertEqual(cfg["baseurl"], "")
        self.assertIn("jekyll-feed", cfg["plugins"])
        self.assertNotIn("theme", cfg)
        self.assertIn("tests", cfg["exclude"])

    def test_front_matter(self):
        self.assertEqual(front_matter(read("index.md"))[0]["layout"], "home")
        for rel in ("_layouts/home.html", "_layouts/post.html"):
            self.assertEqual(front_matter(read(rel))[0]["layout"], "default")


def original_post() -> str:
    """Polska wersja artykułu sprzed tłumaczenia (przypięty commit, nie ruchoma gałąź)."""
    return subprocess.run(
        ["git", "show", f"{BASELINE}:{POST}"], cwd=ROOT, capture_output=True, check=True, text=True
    ).stdout


def code_blocks(text: str, lang: str) -> list[str]:
    return re.findall(rf"^```{lang}\n(.*?)^```$", text, re.S | re.M)


class NormalizeMessages(ast.NodeTransformer):
    """Zastępuje tylko komunikaty dla człowieka: nazwy w check(...) i tekst AssertionError("...")."""

    def visit_Call(self, node):
        self.generic_visit(node)
        name = getattr(node.func, "id", None)
        msg_args = node.args[1:2] if name == "check" else node.args[:1] if name == "AssertionError" else []
        for arg in msg_args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                arg.value = "<msg>"
        return node


def normalized_ast(code: str) -> str:
    return ast.dump(NormalizeMessages().visit(ast.parse(code)))


class PostPreservedTest(unittest.TestCase):
    def test_python_example_unchanged_except_messages(self):
        [original] = code_blocks(original_post(), "python")
        [current] = code_blocks(read(POST), "python")
        self.assertEqual(normalized_ast(current), normalized_ast(original))
        self.assertNotEqual(current, original)  # komunikaty i komentarz faktycznie przetłumaczone

    def test_sql_and_structure_unchanged(self):
        original, current = original_post(), read(POST)
        self.assertEqual(code_blocks(current, "sql"), code_blocks(original, "sql"))
        headings = lambda t: [len(h) for h in re.findall(r"^(#+) ", front_matter(t)[1], re.M)]
        self.assertEqual(headings(current), headings(original))
        links = lambda t: re.findall(r"\]\((https?://[^)]+)\)", t)
        self.assertEqual(links(current), links(original))

    def test_metadata_preserved(self):
        original, _ = front_matter(original_post())
        meta, _ = front_matter(read(POST))
        for key in ("layout", "date", "permalink", "author", "categories", "image"):
            with self.subTest(key=key):
                self.assertEqual(meta[key], original[key])
        self.assertTrue(read(POST).startswith("---\nlayout: post\n"))
        self.assertIn("\ndate: 2023-10-12 09:00:00 +0200\n", read(POST))

    def test_post_metadata(self):
        meta, _ = front_matter(read(POST))
        self.assertEqual(meta["layout"], "post")
        self.assertEqual(meta["lang"], "en")
        self.assertEqual(meta["permalink"], "/blog/pipeline-idempotencja/")
        self.assertTrue((ROOT / meta["image"].lstrip("/")).is_file())

    def test_post_translated(self):
        meta, body = front_matter(read(POST))
        self.assertEqual(meta["title"], "A pipeline you can run twice. Idempotency in practice")
        self.assertIn(f"# {meta['title']}\n", body)
        self.assertIn('<time datetime="2023-10-12">12 October 2023</time>', body)
        for rel, text in ((POST, body + meta["title"] + meta["description"]),
                          ("assets/images/pipeline-idempotencja.svg", read("assets/images/pipeline-idempotencja.svg"))):
            with self.subTest(rel=rel):
                self.assertIsNone(re.search(POLISH, text), rel)

    def test_python_example_runs(self):
        [code] = code_blocks(read(POST), "python")
        result = subprocess.run([sys.executable, "-"], input=code, capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 4, result.stdout)
        self.assertTrue(all(line.startswith("PASS: ") for line in lines), result.stdout)

    def test_html_lang_follows_page(self):
        self.assertIn("<html lang=\"{{ page.lang | default: site.lang | default: 'pl' }}\">",
                      read("_layouts/default.html"))
        self.assertEqual(yaml.safe_load(read("_config.yml"))["lang"], "pl")

    def test_home_references_translated_title(self):
        title = front_matter(read(POST))[0]["title"]
        self.assertIn(f"„{title}”", read("_layouts/home.html"))


class TemplateTest(unittest.TestCase):
    def test_liquid_blocks_balanced(self):
        for rel in TEMPLATES:
            text = read(rel)
            tags = re.findall(r"\{%-?\s*(\w+)", text)
            for opener in ("if", "for", "unless", "case", "capture"):
                with self.subTest(rel=rel, tag=opener):
                    self.assertEqual(tags.count(opener), tags.count("end" + opener))
            self.assertEqual(text.count("{{"), text.count("}}"), rel)
            self.assertEqual(text.count("{%"), text.count("%}"), rel)

    def test_html_balanced(self):
        for rel in TEMPLATES:
            with self.subTest(rel=rel):
                parser = parse(rel)
                self.assertEqual(parser.errors, [])
                self.assertEqual(parser.stack, [])

    def test_anchors_resolve(self):
        ids = set(parse("_layouts/default.html").ids) | set(parse("_layouts/home.html").ids)
        self.assertTrue({"main", "kontakt", "top", "moduly", "dziennik", "arsenal"} <= ids)
        for rel in TEMPLATES:
            for target in re.findall(r'href="[^"]*#([\w-]+)"', read(rel)):
                if "github.com" in target:
                    continue
                with self.subTest(rel=rel, anchor=target):
                    self.assertTrue(target in ids or target.startswith("L"), target)

    def test_nav_works_from_article(self):
        layout = read("_layouts/default.html")
        for anchor in ("moduly", "dziennik", "arsenal"):
            self.assertIn(f"{{{{ '/' | relative_url }}}}#{anchor}", layout)

    def test_internal_links_exist(self):
        for rel in TEMPLATES:
            for path in re.findall(r"\{\{\s*'(/[^']*)'\s*\|\s*relative_url", read(rel)):
                if path in ("/", "/feed.xml") or path.startswith("/blog/"):
                    continue
                with self.subTest(path=path):
                    self.assertTrue((ROOT / path.lstrip("/")).is_file(), path)
        permalink = front_matter(read(POST))[0]["permalink"]
        self.assertIn(f"'{permalink}'", read("_layouts/home.html"))

    def test_diary_uses_real_posts(self):
        home = read("_layouts/home.html")
        self.assertIn("for post in site.posts", home)
        for fake in ("2026", "września", "Routing modeli", "dlt zamiast", "[DATA]", "Post.dc.html"):
            self.assertNotIn(fake, home)

    def test_no_fabricated_or_runtime_content(self):
        banned = ["AGiM", "OSTIN", "JARVIS", "Hermes", "halobotics", "LinkedIn", "linkedin",
                  "mailto:", "TWÓJ E-MAIL", "x-dc", "react.js", "react-dom", "React", "support.js", "Warszawa", "DBT", "DLT"]
        for rel in TEMPLATES + ["assets/css/hud.css", "index.md", "_config.yml"]:
            text = read(rel)
            for word in banned:
                with self.subTest(rel=rel, word=word):
                    self.assertFalse(word in text, f"{word!r} in {rel}")

    def test_arsenal_excerpt_is_verbatim(self):
        home = read("_layouts/home.html")
        block = re.search(r"<pre><code>(.*?)</code></pre>", home, re.S).group(1)
        code = html.unescape(re.sub(r"<[^>]+>", "", block)).replace("▍", "")
        post_lines = read(POST).splitlines()
        self.assertIn(code, read(POST))
        start, end = map(int, re.search(r"#L(\d+)-L(\d+)", home).groups())
        self.assertEqual("\n".join(post_lines[start - 1:end]), code)
        self.assertIn("fragment", home.lower())

    def test_contact_only_github(self):
        cfg = yaml.safe_load(read("_config.yml"))
        self.assertEqual(cfg["github_username"], "rafalpronko")
        external = set(re.findall(r'href="(https?://[^"]+)"', read("_layouts/default.html")))
        self.assertTrue(all("github.com" in u or "fonts.g" in u for u in external), external)


class CssTest(unittest.TestCase):
    def test_design_tokens_and_a11y(self):
        css = read("assets/css/hud.css")
        for needle in ("#020A12", "#38E1FF", "48px 48px", "Oxanium", "Chakra Petch", "JetBrains Mono",
                       "prefers-reduced-motion", ":focus-visible", ".skip-link", "overflow-x: auto",
                       "max-width: 780px", "font-size: 18px", ".highlight .k"):
            with self.subTest(needle=needle):
                self.assertIn(needle, css)
        self.assertEqual(css.count("{"), css.count("}"))


class XmlTest(unittest.TestCase):
    def test_svg_well_formed(self):
        for svg in ROOT.glob("assets/**/*.svg"):
            with self.subTest(svg=svg.name):
                ET.parse(svg)


def post_meta(name: str) -> tuple[dict, str]:
    return front_matter(read(f"_posts/{name}"))


def asset_files() -> list[Path]:
    return sorted(p for p in (ROOT / "assets").rglob("*") if p.is_file())


class InventoryTest(unittest.TestCase):
    def test_exact_post_inventory(self):
        self.assertEqual(sorted(p.name for p in (ROOT / "_posts").iterdir()), sorted(POSTS))

    def test_post_front_matter(self):
        for name, (permalink, title) in POSTS.items():
            meta, body = post_meta(name)
            with self.subTest(post=name):
                self.assertEqual(meta["layout"], "post")
                self.assertEqual(meta["lang"], "en")
                self.assertEqual(meta["author"], "Rafał Prońko")
                self.assertEqual(meta["title"], title)
                self.assertEqual(meta["permalink"], permalink)
                self.assertEqual(permalink, f"/blog/{name[11:-3]}/")
                self.assertTrue(str(meta["date"]).startswith(name[:10]), meta["date"])
                self.assertTrue(meta.get("description"))
                if name in TITLE_FROM_LAYOUT:
                    self.assertIsNone(re.search(r"^# ", strip_code(body), re.M))
                else:
                    self.assertIn(f"# {title}\n", body)
                self.assertTrue((ROOT / meta["image"].lstrip("/")).is_file(), meta["image"])

    def test_unique_titles_and_permalinks(self):
        titles = [t.casefold() for _, t in POSTS.values()]
        permalinks = [p for p, _ in POSTS.values()]
        self.assertEqual(len(set(titles)), len(titles))
        self.assertEqual(len(set(permalinks)), len(permalinks))

    def test_no_polish_or_translated_duplicates(self):
        # Wszystkie wpisy są angielskie; żaden nie powiela tytułu ani adresu innego wpisu (brak „tłumaczeń”).
        for name in POSTS:
            meta, body = post_meta(name)
            with self.subTest(post=name):
                text = meta["title"] + meta["description"] + body
                for word in POLISH_WORDS_ALLOWED:
                    text = text.replace(word, "")
                self.assertIsNone(re.search(POLISH, text))


class AssetReferenceTest(unittest.TestCase):
    def test_post_asset_references_resolve(self):
        cfg = yaml.safe_load(read("_config.yml"))
        for name in POSTS:
            meta, body = post_meta(name)
            for path in RELATIVE_URL.findall(body) + [meta["image"]]:
                if path.startswith("/blog/"):
                    with self.subTest(post=name, link=path):
                        self.assertIn(path, [p for p, _ in POSTS.values()])
                    continue
                with self.subTest(post=name, asset=path):
                    target = ROOT / unquote(path).lstrip("/")
                    if path.endswith("/"):
                        target = target / "README.md"
                    self.assertTrue(target.is_file(), path)
                    hidden = [c for c in target.relative_to(ROOT).parts if c[0] in "_."]
                    self.assertTrue(set(hidden) <= set(cfg.get("include", [])), hidden)

    def test_jekyll_includes_hidden_example_files(self):
        # Jekyll pomija nazwy zaczynające się od "_" lub "." — każdy taki plik w assets/ musi być w include.
        include = set(yaml.safe_load(read("_config.yml")).get("include", []))
        for path in asset_files():
            hidden = [c for c in path.relative_to(ROOT).parts if c[0] in "_."]
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertTrue(set(hidden) <= include, hidden)


class ExampleLayoutTest(unittest.TestCase):
    def test_example_docs_use_prose_layout(self):
        defaults = yaml.safe_load(read("_config.yml")).get("defaults", [])
        scoped = [d["values"] for d in defaults if d["scope"].get("path") == "assets/examples"]
        self.assertEqual(scoped, [{"lang": "en", "layout": "example-doc"}])
        meta, body = front_matter(read(EXAMPLE_LAYOUT))
        self.assertEqual(meta["layout"], "default")
        self.assertRegex(body, r'<div class="prose">\s*\{\{ content \}\}\s*</div>')


class PayloadTest(unittest.TestCase):
    def test_assets_allowlist(self):
        for path in asset_files():
            rel = str(path.relative_to(ROOT))
            with self.subTest(path=rel):
                self.assertTrue(any(re.fullmatch(p, rel) for p in ASSET_ALLOWLIST), rel)
                self.assertIsNone(FORBIDDEN_PATH.search(rel))

    def test_no_secrets_or_private_paths(self):
        files = asset_files() + sorted((ROOT / "_posts").iterdir())
        for path in files:
            members = [(path.name, path.read_bytes())]
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as z:
                    members += [(f"{path.name}!{n}", z.read(n)) for n in z.namelist() if not n.endswith("/")]
            for name, data in members:
                with self.subTest(file=name):
                    self.assertIsNone(FORBIDDEN_PATH.search(name))
                    self.assertIsNone(SECRET.search(data.decode("utf-8", "replace")))

    def test_zip_archives_intact(self):
        for path in (ROOT / "assets").rglob("*.zip"):
            with self.subTest(zip=path.name), zipfile.ZipFile(path) as z:
                self.assertIsNone(z.testzip())
                for n in z.namelist():
                    self.assertFalse(n.startswith("/") or ".." in n.split("/"), n)


class RenderedPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang, self.ids, self.links, self.h1, self.diary = None, set(), [], [], []
        self._h1: list[str] | None = None
        self._divs: list[bool] = []  # stos otwartych <div>: czy to .prose
        self.outside_prose: list[str] = []  # h1/pre/table poza .prose

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang")
        if a.get("id"):
            self.ids.add(a["id"])
        self.links += [a[k] for k in ("href", "src") if a.get(k)]
        if tag == "a" and "diary-item" in (a.get("class") or ""):
            self.diary.append(a["href"])
        if tag == "h1":
            self._h1 = []
        if tag == "div":
            self._divs.append("prose" in (a.get("class") or "").split())
        if tag in ("h1", "pre", "table") and not any(self._divs):
            self.outside_prose.append(tag)

    def handle_endtag(self, tag):
        if tag == "div" and self._divs:
            self._divs.pop()
        if tag == "h1" and self._h1 is not None:
            self.h1.append(" ".join("".join(self._h1).split()))
            self._h1 = None

    def handle_data(self, data):
        if self._h1 is not None:
            self._h1.append(data)


@unittest.skipUnless(os.environ.get("RENDERED_SITE"), "ustaw RENDERED_SITE na zbudowany katalog _site")
class RenderedSiteTest(unittest.TestCase):
    """Sprawdza wynik `jekyll build`, a nie źródła."""

    @classmethod
    def setUpClass(cls):
        cls.site = Path(os.environ["RENDERED_SITE"]).resolve()
        cls.pages = {}
        for path in cls.site.rglob("*.html"):
            page = RenderedPage()
            page.feed(path.read_text(encoding="utf-8"))
            cls.pages[path] = page

    def target(self, page_path: Path, href: str) -> tuple[Path, str]:
        path, frag = urldefrag(href)
        url = urlparse(path)
        if url.scheme or url.netloc:
            path = url.path if href.startswith(yaml.safe_load(read("_config.yml"))["url"]) else None
            if path is None:
                return None, None
        if not path:
            return page_path, frag
        base = self.site if path.startswith("/") else page_path.parent
        target = base / unquote(path).lstrip("/")
        return (target / "index.html" if target.is_dir() else target), frag

    def test_post_pages(self):
        for permalink, title in POSTS.values():
            page = self.pages.get(self.site / permalink.lstrip("/") / "index.html")
            with self.subTest(permalink=permalink):
                self.assertIsNotNone(page)
                self.assertEqual(page.lang, "en")
                # kramdown zamienia apostrofy z nagłówka Markdown na typograficzne (’).
                self.assertEqual([h.replace("’", "'") for h in page.h1], [title])
        built = {p.parent.name for p in self.site.glob("blog/*/index.html")}
        self.assertEqual(built, {p.split("/")[2] for p, _ in POSTS.values()})

    def test_example_docs_wrapped_in_prose(self):
        # README/PROVENANCE przykładów renderowane przez GitHub Pages: treść w .prose (przewijane pre/table na mobile).
        docs = [p for p in self.pages if p.is_relative_to(self.site / "assets/examples")]
        self.assertGreaterEqual(len(docs), 6)
        for path in docs:
            page = self.pages[path]
            with self.subTest(page=str(path.relative_to(self.site))):
                self.assertEqual(len(page.h1), 1)
                self.assertEqual(page.outside_prose, [])

    def test_only_home_is_polish(self):
        # Wpisy i strony z README przykładów są po angielsku; po polsku zostaje tylko strona główna (HUD).
        for path, page in self.pages.items():
            with self.subTest(page=str(path.relative_to(self.site))):
                self.assertEqual(page.lang, "pl" if path == self.site / "index.html" else "en")

    def test_local_links_and_fragments_resolve(self):
        for path, page in self.pages.items():
            for href in page.links:
                if href.startswith(("mailto:", "data:")):
                    continue
                target, frag = self.target(path, href)
                if target is None:
                    continue
                with self.subTest(page=str(path.relative_to(self.site)), href=href):
                    self.assertTrue(target.is_file(), href)
                    if frag and target.suffix == ".html":
                        self.assertIn(frag, self.pages[target].ids)

    def test_diary_lists_all_posts_newest_first(self):
        home = self.pages[self.site / "index.html"]
        expected = [POSTS[name][0] for name in sorted(POSTS, reverse=True)]
        self.assertEqual(home.diary, expected)
        self.assertEqual(home.lang, "pl")

    def test_feed_links_built(self):
        root = ET.parse(self.site / "feed.xml").getroot()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        links = [e.find("a:link", ns).get("href") for e in root.findall("a:entry", ns)]
        newest = [POSTS[name][0] for name in sorted(POSTS, reverse=True)]
        self.assertEqual(links, [yaml.safe_load(read("_config.yml"))["url"] + p for p in newest[:len(links)]])
        self.assertGreaterEqual(len(links), min(10, len(POSTS)))

    def test_assets_served_byte_for_byte(self):
        for path in asset_files():
            rel = path.relative_to(ROOT)
            with self.subTest(asset=str(rel)):
                built = self.site / rel
                self.assertTrue(built.is_file(), rel)
                self.assertEqual(hashlib.sha256(built.read_bytes()).hexdigest(),
                                 hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
