"""Statyczna walidacja strony Jekyll (bez budowania).

Uruchomienie: uv run --with pyyaml python -m unittest discover -s tests -v
"""
import ast
import html
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
POST = "_posts/2023-10-12-pipeline-idempotencja.md"
# Ostatni commit z polską wersją artykułu — stały punkt odniesienia dla tłumaczenia.
BASELINE = "ff2e6c338f812937fefd293dc1bb066a5c439fc3"
POLISH = r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]"
TEMPLATES = ["_layouts/default.html", "_layouts/home.html", "_layouts/post.html", "_includes/date-pl.html"]
VOID = {"meta", "link", "br", "img", "input", "hr", "source"}


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


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


if __name__ == "__main__":
    unittest.main()
