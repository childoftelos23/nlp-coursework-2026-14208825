"""Tests for Part Two cleaning, splitting and custom tokenisation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from PartTwo import (
    clean_hansard,
    custom_tokenizer,
    make_custom_vectorizer,
    split_data,
)


class PartTwoTests(unittest.TestCase):
    def test_custom_tokenizer_removes_stopwords_and_stems(self):
        tokens = custom_tokenizer(
            "The members were arguing, argued and ARGUES."
        )
        self.assertNotIn("the", tokens)
        self.assertIn("member", tokens)
        self.assertEqual(tokens.count("argu"), 3)

    def test_cleaning_follows_the_required_order(self):
        long_speech = "word " * 250
        rows = []
        for party in [
            "Conservative",
            "Labour",
            "Labour (Co-op)",
            "Scottish National Party",
            "Liberal Democrat",
            "Speaker",
        ]:
            for number in range(4):
                rows.append(
                    {
                        "speech": long_speech,
                        "party": party,
                        "speech_class": "Speech",
                        "speakername": f"Person {number}",
                    }
                )
        rows.append(
            {
                "speech": "short",
                "party": "Conservative",
                "speech_class": "Speech",
                "speakername": "Short",
            }
        )
        rows.append(
            {
                "speech": long_speech,
                "party": "Conservative",
                "speech_class": "Procedural",
                "speakername": "Procedure",
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            cleaned, audit = clean_hansard(path)

        self.assertEqual(
            set(cleaned["party"]),
            {
                "Conservative",
                "Labour",
                "Scottish National Party",
                "Liberal Democrat",
            },
        )
        self.assertNotIn("Labour (Co-op)", set(cleaned["party"]))
        self.assertTrue(
            cleaned["speech"].str.len().ge(1_000).all()
        )
        self.assertEqual(
            audit["top_four_parties_after_labour_merge"][0],
            "Labour",
        )

    def test_stratified_split_and_feature_limit(self):
        rows = []
        for party in ["A", "B", "C", "D"]:
            for number in range(10):
                rows.append(
                    {
                        "speech": f"{party} speech number {number} economy",
                        "party": party,
                    }
                )
        dataframe = pd.DataFrame(rows)
        x_train, x_test, y_train, y_test = split_data(dataframe)
        self.assertEqual(
            y_test.value_counts().to_dict(),
            {"A": 2, "B": 2, "C": 2, "D": 2},
        )

        vectorizer = make_custom_vectorizer(5)
        matrix = vectorizer.fit_transform(x_train)
        self.assertLessEqual(matrix.shape[1], 5)


if __name__ == "__main__":
    unittest.main()

