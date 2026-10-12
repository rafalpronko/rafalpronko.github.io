"""End-to-end test with the real flan-t5-small model on CPU.

Skipped unless RUN_MODEL=1, because the first run downloads the model weights.
The assertions check behaviour we can state honestly: the pipeline generates
text, cites the right section and is deterministic with greedy decoding.
They do not claim that the generated wording is correct.
"""

import os
import unittest

from first_local_rag.corpus import load_chunks
from first_local_rag.pipeline import answer
from first_local_rag.retriever import TfidfRetriever


@unittest.skipUnless(os.environ.get("RUN_MODEL") == "1", "set RUN_MODEL=1 to run the model")
class EndToEndModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from first_local_rag.generator import FlanT5Generator
        cls.generator = FlanT5Generator()
        cls.retriever = TfidfRetriever(load_chunks())

    def test_generates_an_answer_with_retrieved_citations(self):
        result = answer("Which request methods are idempotent?", self.retriever, self.generator)
        self.assertEqual(result.status, "answered")
        self.assertTrue(result.text)
        self.assertEqual(result.citations[0].chunk_id, "rfc9110-9.2.2#p1")
        self.assertLessEqual(result.prompt_tokens, 512)

    def test_greedy_generation_is_repeatable(self):
        question = "How is the HEAD method different from GET?"
        first = answer(question, self.retriever, self.generator).text
        second = answer(question, self.retriever, self.generator).text
        self.assertEqual(first, second)

    def test_unknown_question_never_reaches_the_model(self):
        result = answer("Who won the 2018 FIFA World Cup?", self.retriever, self.generator)
        self.assertEqual(result.status, "no_evidence")


if __name__ == "__main__":
    unittest.main()
