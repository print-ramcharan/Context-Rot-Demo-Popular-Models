"""Tests for LongMemEval metrics and dataset structure."""

import unittest
from evaluation.runners.run_longmem_eval import compute_token_f1, compute_accuracy


class TestLongMemEval(unittest.TestCase):
    def test_token_f1_exact_match(self):
        f1 = compute_token_f1("Barnaby, a Golden Retriever", "Barnaby, a Golden Retriever")
        self.assertAlmostEqual(f1, 1.0)

    def test_token_f1_partial_overlap(self):
        # pred: "The dog's name is Barnaby" (overlap with "Barnaby")
        f1 = compute_token_f1("The dog is Barnaby", "Barnaby the dog")
        self.assertTrue(f1 > 0.6)

    def test_token_f1_zero_overlap(self):
        f1 = compute_token_f1("I do not know", "Barnaby, a Golden Retriever")
        self.assertAlmostEqual(f1, 0.0)

    def test_accuracy_keyword_present(self):
        acc = compute_accuracy("You adopted a dog named Barnaby recently.", ["barnaby", "golden retriever"])
        self.assertEqual(acc, 1.0)

    def test_accuracy_keyword_absent(self):
        acc = compute_accuracy("I am an AI and I do not have access to your history.", ["barnaby", "golden retriever"])
        self.assertEqual(acc, 0.0)


if __name__ == "__main__":
    unittest.main()
