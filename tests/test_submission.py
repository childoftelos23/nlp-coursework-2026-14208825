"""Checks for the easy-to-miss submission requirements."""

import re
import unittest
from pathlib import Path

from PartThree import MODEL_REVISION


ACADEMIC_DECLARATION = (
    "I have read and understood the sections of plagiarism in the College "
    "Policy on assessment offences and confirm that the work is my own, "
    "with the work of others clearly acknowledged. I give my permission to "
    "submit my report to the plagiarism testing database that the College is "
    "using and test it using plagiarism detection software, search engines "
    "or meta-searching software."
)


class SubmissionTests(unittest.TestCase):
    def test_readme_contains_exact_academic_declaration(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn(ACADEMIC_DECLARATION, readme)

    def test_both_limited_discussions_are_below_500_words(self):
        report = Path("REPORT.md").read_text(encoding="utf-8")
        for part in ("PART2", "PART3"):
            discussion = report.split(
                f"<!-- {part}_DISCUSSION_START -->"
            )[1].split(
                f"<!-- {part}_DISCUSSION_END -->"
            )[0]
            words = re.findall(
                r"\b[\w'-]+\b",
                discussion,
            )
            self.assertLessEqual(
                len(words),
                500,
                f"{part} discussion has {len(words)} words",
            )

    def test_model_revision_is_pinned(self):
        self.assertNotEqual(MODEL_REVISION, "main")
        self.assertEqual(len(MODEL_REVISION), 40)

    def test_required_submission_files_exist(self):
        for name in (
            "PartOne.py",
            "PartTwo.py",
            "PartThree.py",
            "README.md",
            "REPORT.md",
            "AI_USE_DECLARATION.md",
            "requirements.txt",
        ):
            self.assertTrue(
                Path(name).exists(),
                name,
            )


if __name__ == "__main__":
    unittest.main()

