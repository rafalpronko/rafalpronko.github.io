"""Statyczna walidacja strony Jekyll (bez budowania).

Uruchomienie: uv run --with pyyaml python -m unittest discover -s tests -v
"""
import html
import re
import subprocess
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
POST = "_posts/2023-10-12-pipeline-idempotencja.md"
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


class PostPreservedTest(unittest.TestCase):
    def test_only_layout_line_changed(self):
        original = subprocess.run(
            ["git", "show", f"origin/master:{POST}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        current = (ROOT / POST).read_bytes()
        expected = original.replace(b"\nlayout: default\n", b"\nlayout: post\n", 1)
        self.assertEqual(current, expected)

    def test_post_metadata(self):
        meta, _ = front_matter(read(POST))
        self.assertEqual(meta["layout"], "post")
        self.assertEqual(meta["permalink"], "/blog/pipeline-idempotencja/")
        self.assertTrue((ROOT / meta["image"].lstrip("/")).is_file())


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
