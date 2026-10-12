import json
import pathlib
import shutil
import tempfile
import unittest

from first_local_rag.corpus import DEFAULT_CORPUS, IntegrityError, load_chunks
from first_local_rag.retriever import TfidfRetriever, tokenize


class CorpusTest(unittest.TestCase):
    def test_extracts_match_manifest_hashes(self):
        chunks = load_chunks()
        self.assertEqual({c.doc_id for c in chunks},
                         {"rfc9110-9.2.2", "rfc9110-9.3.1", "rfc9110-9.3.2"})

    def test_heading_is_not_a_chunk(self):
        first = [c for c in load_chunks() if c.chunk_id == "rfc9110-9.2.2#p1"][0]
        self.assertTrue(first.text.startswith('A request method is considered "idempotent"'))

    def test_modified_extract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = pathlib.Path(tmp)
            for path in DEFAULT_CORPUS.iterdir():
                shutil.copy(path, corpus / path.name)
            record = json.loads((corpus / "manifest.json").read_text())["documents"][0]
            target = corpus / record["file"]
            target.write_text(target.read_text().replace("SHOULD NOT", "MAY"))
            with self.assertRaises(IntegrityError):
                load_chunks(corpus)


class RetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = TfidfRetriever(load_chunks())

    def top_doc(self, question):
        return self.retriever.search(question)[0].chunk.doc_id

    def test_tokenize_drops_stopwords_and_keeps_hyphenated_terms(self):
        self.assertEqual(tokenize("Is a non-idempotent request safe?"),
                         ["non-idempotent", "request", "safe"])

    def test_idempotent_question_ranks_section_9_2_2_first(self):
        self.assertEqual(self.top_doc("Which request methods are idempotent?"), "rfc9110-9.2.2")

    def test_head_question_ranks_section_9_3_2_first(self):
        self.assertEqual(self.top_doc("How is the HEAD method different from GET?"),
                         "rfc9110-9.3.2")

    def test_unrelated_question_returns_no_hits(self):
        self.assertEqual(self.retriever.search("Who won the 2018 FIFA World Cup?"), [])

    def test_synonym_is_a_lexical_blind_spot(self):
        # "repeatable" never occurs in the extracts, so TF-IDF cannot find 9.2.2.
        hits = self.retriever.search("Which calls are safely repeatable?")
        self.assertNotIn("rfc9110-9.2.2", [h.chunk.doc_id for h in hits])


if __name__ == "__main__":
    unittest.main()
