"""Tests for Part Three data, prompts and label parsing."""

import unittest
from pathlib import Path

from PartThree import (
    PARTY_LABELS,
    choose_few_shot_examples,
    make_few_shot_prompt,
    make_zero_shot_prompt,
    normalise_model_label,
    prepare_llm_data,
    split_llm_data,
)


class PartThreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataframe, cls.audit = prepare_llm_data(
            Path("texts/hansard500.csv"),
            Path("texts/hansard10000.csv"),
        )
        cls.training_data, cls.test_data = split_llm_data(
            cls.dataframe
        )

    def test_same_label_set_and_stratified_split(self):
        self.assertEqual(
            self.audit["label_set_from_part_two"],
            PARTY_LABELS,
        )
        self.assertEqual(
            set(self.training_data["party"]),
            set(PARTY_LABELS),
        )
        self.assertEqual(
            set(self.test_data["party"]),
            set(PARTY_LABELS),
        )

    def test_few_shot_examples_use_training_data_only(self):
        examples = choose_few_shot_examples(self.training_data)
        self.assertEqual(
            [example["label"] for example in examples],
            PARTY_LABELS,
        )
        training_indices = set(
            self.training_data["source_index"].astype(str)
        )
        test_indices = set(
            self.test_data["source_index"].astype(str)
        )
        for example in examples:
            self.assertIn(
                example["source_index"],
                training_indices,
            )
            self.assertNotIn(
                example["source_index"],
                test_indices,
            )

    def test_prompts_demand_one_allowed_label(self):
        examples = choose_few_shot_examples(self.training_data)
        zero_prompt = make_zero_shot_prompt("A sample speech.")
        few_prompt = make_few_shot_prompt(
            "A sample speech.",
            examples,
        )
        for label in PARTY_LABELS:
            self.assertIn(label, zero_prompt)
            self.assertIn(label, few_prompt)
        self.assertIn(
            "Return exactly one allowed label",
            zero_prompt,
        )
        self.assertIn(
            "Return exactly one allowed label",
            few_prompt,
        )

    def test_label_parser_does_not_silently_guess(self):
        self.assertEqual(
            normalise_model_label("Labour"),
            "Labour",
        )
        self.assertEqual(
            normalise_model_label(
                "The answer is Scottish National Party."
            ),
            "Scottish National Party",
        )
        self.assertEqual(
            normalise_model_label("I cannot decide."),
            "__INVALID__",
        )


if __name__ == "__main__":
    unittest.main()

