# Verified Results

## Part One

The full twelve-novel script completed successfully, including parsing, pickling, reloading and all dependency outputs. The supplied files span 1811 to 1930.

| Novel | Year | TTR | Flesch Reading Ease |
|---|---:|---:|---:|
| Sense and Sensibility | 1811 | 0.070760 | 61.6609 |
| North and South | 1855 | 0.080251 | 76.1636 |
| A Tale of Two Cities | 1858 | 0.093960 | 73.5011 |
| Erewhon | 1872 | 0.112379 | 52.1879 |
| The American | 1877 | 0.091650 | 75.3569 |
| Dorian Gray | 1890 | 0.106863 | 79.9459 |
| Tess of the DUrbervilles | 1891 | 0.097384 | 70.9569 |
| The Secret Garden | 1911 | 0.074959 | 85.2993 |
| Portrait of the Artist | 1916 | 0.127245 | 75.7020 |
| The Black Moth | 1926 | 0.100067 | 82.6793 |
| Orlando | 1928 | 0.134989 | 66.6729 |
| Blood Meridian | 1930 | 0.103768 | 81.4903 |

The direct-object and PMI rankings are printed by PartOne.py for every novel. The full run was not copied into this file because the question asks the script to print those lists.

## Part Two cleaning

- Original shape: (10000, 8).
- Original duplicate rows detected: 5.
- Missing party values detected: 411.
- Four parties after the Labour merge: Conservative, Labour, Scottish National Party, Liberal Democrat.
- Final shape after the required filters: (2112, 8).
- Final class counts: Conservative 1,248; Labour 626; Scottish National Party 166; Liberal Democrat 72.
- Final duplicate rows: 0.
- Stratified test supports: 250, 125, 33 and 15 respectively.

## Part Two test results

| Features | Classifier | Feature count | Macro-F1 | Accuracy |
|---|---|---:|---:|---:|
| Default TF-IDF | LogisticRegression | 3,000 | 0.4361 | 0.7187 |
| Default TF-IDF | ComplementNB | 3,000 | 0.4830 | 0.7163 |
| Unigrams, bigrams, trigrams | LogisticRegression | 3,000 | 0.4723 | 0.7329 |
| Unigrams, bigrams, trigrams | ComplementNB | 3,000 | 0.4845 | 0.7116 |
| Custom tokenizer and selected settings | ComplementNB | 2,000 | 0.5613 | 0.7400 |

Custom-model training-only cross-validation:

| Model | Feature limit | Mean macro-F1 | Standard deviation |
|---|---:|---:|---:|
| LogisticRegression | 1,000 | 0.4566 | 0.0382 |
| ComplementNB | 1,000 | 0.4861 | 0.0266 |
| LogisticRegression | 2,000 | 0.4478 | 0.0389 |
| ComplementNB | 2,000 | 0.5112 | 0.0515 |
| LogisticRegression | 3,000 | 0.4409 | 0.0285 |
| ComplementNB | 3,000 | 0.4980 | 0.0549 |

The selected custom ComplementNB test report gave Liberal Democrat precision 1.0000, recall 0.1333 and F1 0.2353. This is why the apparently good precision must not be read without recall.

## Part Three

Status: execution complete.

- Prepared rows: 457.
- Class counts: Conservative 313; Labour 107; Scottish National Party 32; Liberal Democrat 5.
- Length filter decision: 1,000-character filter not repeated because Liberal Democrat count would drop to 1, preventing stratified splitting.
- Stratified 80/20 split train counts: Conservative 250; Labour 85; Scottish National Party 26; Liberal Democrat 4.
- Stratified 80/20 split test supports: Conservative 63; Labour 22; Scottish National Party 6; Liberal Democrat 1 (Total: 92).
- Pinned Model: `google/flan-t5-small` (revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`).
- Prediction file saved: `results/part_three_predictions.csv`.

### Part 3(b) Zero-Shot Results

- Macro-F1: **0.0000**
- Invalid model outputs: **92 / 92**

| Party | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Conservative | 0.0000 | 0.0000 | 0.0000 | 63 |
| Labour | 0.0000 | 0.0000 | 0.0000 | 22 |
| Scottish National Party | 0.0000 | 0.0000 | 0.0000 | 6 |
| Liberal Democrat | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Macro average** | **0.0000** | **0.0000** | **0.0000** | **92** |

### Part 3(c) Few-Shot Results

- Macro-F1: **0.0000**
- Invalid model outputs: **92 / 92**

| Party | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Conservative | 0.0000 | 0.0000 | 0.0000 | 63 |
| Labour | 0.0000 | 0.0000 | 0.0000 | 22 |
| Scottish National Party | 0.0000 | 0.0000 | 0.0000 | 6 |
| Liberal Democrat | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Macro average** | **0.0000** | **0.0000** | **0.0000** | **92** |

### Part 3(d) Comparison Summary

Both zero-shot and few-shot settings yielded a Macro-F1 of **0.0000** across all 92 test samples. The raw model generations repeatedly listed all allowed label names (e.g., `"Conservative Labour Scottish National Party Liberal Democrat"`) instead of generating a single class label. Under strict normalization rules, multi-class list outputs are marked invalid (`__INVALID__`), preventing accidental false matches.

