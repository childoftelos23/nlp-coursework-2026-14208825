"""Small tests for the main Part One calculations."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import spacy

from PartOne import (
    count_syl,
    direct_object_arcs,
    fk_level,
    nltk_ttr,
    parse,
    pmi_verbs_for_object,
    read_novels,
    top_direct_objects,
)


class PartOneTests(unittest.TestCase):
    def test_count_syl_uses_cmudict_and_fallback(self):
        dictionary = {
            "example": [
                ["IH0", "G", "Z", "AE1", "M", "P", "AH0", "L"],
            ]
        }
        self.assertEqual(count_syl("example", dictionary), 3)
        self.assertEqual(count_syl("beaulieu", {}), 2)

    def test_ttr_ignores_case_and_punctuation(self):
        self.assertAlmostEqual(
            nltk_ttr("Cat, cat! Dog."),
            2 / 3,
        )

    def test_reading_ease_uses_the_right_formula(self):
        dictionary = {
            "cat": [["K", "AE1", "T"]],
            "sits": [["S", "IH1", "T", "S"]],
        }
        expected = 206.835 - (1.015 * 2) - (84.6 * 1)
        self.assertAlmostEqual(
            fk_level("Cat sits.", dictionary),
            expected,
        )

    def test_read_novels_sorts_and_resets_index(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "Later-Writer-1900.txt").write_text(
                "Later.",
                encoding="utf-8",
            )
            (folder / "Earlier_Title-Author-1800.txt").write_text(
                "Earlier.",
                encoding="utf-8",
            )
            dataframe = read_novels(folder)

        self.assertEqual(
            dataframe["year"].tolist(),
            [1800, 1900],
        )
        self.assertEqual(
            dataframe.index.tolist(),
            [0, 1],
        )
        self.assertEqual(
            dataframe.iloc[0]["title"],
            "Earlier Title",
        )

    def test_parse_and_dependency_answers(self):
        nlp = spacy.load("en_core_web_sm")
        dataframe = pd.DataFrame(
            [
                {
                    "text": "Alice saw him. Bob helped her. Alice saw him.",
                    "title": "Test",
                    "author": "A",
                    "year": 1900,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            parsed = parse(
                dataframe,
                directory,
                "test.pickle",
                nlp=nlp,
            )
            reloaded = pd.read_pickle(
                Path(directory) / "test.pickle",
            )

        arcs = direct_object_arcs(parsed.iloc[0]["doc"])
        self.assertIn(("see", "him"), arcs)
        self.assertIn(("help", "her"), arcs)
        self.assertIn(
            ("him", 2),
            top_direct_objects(reloaded.iloc[0]["doc"]),
        )
        self.assertEqual(
            pmi_verbs_for_object(
                reloaded.iloc[0]["doc"],
                "him",
            )[0][0],
            "see",
        )


if __name__ == "__main__":
    unittest.main()
