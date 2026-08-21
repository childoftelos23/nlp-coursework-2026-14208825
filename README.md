# COIY064H7 Natural Language Processing

## Programming Portfolio Re-assessment

Student: Luke Joseph

This repository contains the three Python scripts, supplied datasets, tests and written answers for the reassessment.

## Academic Declaration

> “I have read and understood the sections of plagiarism in the College Policy on assessment offences and confirm that the work is my own, with the work of others clearly acknowledged. I give my permission to submit my report to the plagiarism testing database that the College is using and test it using plagiarism detection software, search engines or meta-searching software.”

## Files

- PartOne.py answers Part One, including reading and sorting the novels, NLTK TTR, Flesch Reading Ease, spaCy parsing and pickling, direct-object frequencies, and PMI rankings.
- PartTwo.py answers Part Two, including the exact cleaning order, TF-IDF baselines, n-grams, the custom tokenizer, cross-validation and classification reports.
- PartThree.py implements the zero-shot and few-shot Hugging Face evaluation and saves row-level predictions.
- REPORT.md contains the two short discussions and methodological choices.
- RESULTS.md records the outputs that were successfully executed.
- ASSESSMENT_MAPPING.md maps every mark-bearing instruction before the final work.
- RESEARCH_EVIDENCE.md records and verifies the supporting sources.
- EVIDENCE_AUDIT.md provides the requested final checks.
- VALIDATION.md records what was tested and what remains to run.

## Environment setup

Create and activate a Python virtual environment, then run:

    python -m pip install -r requirements.txt
    python -m spacy download en_core_web_sm

Part Three downloads the pinned google/flan-t5-small model files from Hugging Face the first time it is run.

## Run the work

From this folder:

    python PartOne.py
    python PartTwo.py
    python PartThree.py

Part One may take several minutes and creates pickles/parsed.pickle. On later runs:

    python PartOne.py --reuse-pickle

Run every automated test with:

    python -m unittest discover -s tests -v

## Part Three execution status

`PartThree.py` has been executed completely end-to-end using local inference with `google/flan-t5-small`. Zero-shot and few-shot classification reports and macro-F1 values (0.0000) have been verified, documented in `REPORT.md` and `RESULTS.md`, and saved to `results/part_three_predictions.csv`.

## Reproducibility controls

- Random seed: 26.
- Part Two: stratified 80/20 split; every TF-IDF vectorizer is fitted to training text only.
- Custom Part Two selection: five-fold stratified training-only cross-validation.
- Part Three: stratified 80/20 split; examples are selected from training data only.
- LLM decoding: deterministic greedy generation with sampling disabled.
- Model revision: 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.



