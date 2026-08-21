# Evidence and Citation Audit

| Claim | Assignment location | Supporting source | Source class | Direct support? | Verification | Confidence |
|---|---|---|---|---|---|---|
| The supplied novels span 1811–1930 despite the nineteenth-century heading | REPORT.md Part One; RESULTS.md | Supplied filenames and parsed year column | Primary supplied data | Yes | Full directory audit | High |
| TTR excludes punctuation and ignores case | PartOne.py; REPORT.md | Coursework brief; NLTK-based code and unit test | Primary brief and direct execution | Yes | Test passed | High |
| The code calculates Flesch Reading Ease rather than Grade Level | PartOne.py | Exact wording in brief; formula in code | Primary brief and executable method | Yes | Formula unit test passed | High |
| PMI ranks verb-object association using joint and marginal probabilities | PartOne.py; REPORT.md | Church and Hanks (1990) | Primary research | Yes | Formula reviewed and tested | High |
| Rare events can receive high PMI | REPORT.md | Church and Hanks (1990); observed one-count rankings | Primary research plus execution | Yes | Cross-checked | High |
| TF-IDF emphasises discriminative terms | REPORT.md; RESEARCH_EVIDENCE.md | Spärck Jones (1972); Nulty (2026); scikit-learn docs | Primary research, teaching source, official docs | Yes | Cross-checked | High |
| The Part Two final shape is (2112, 8) | RESULTS.md | Direct pandas execution on hansard10000.csv | Primary supplied data | Yes | Full run completed | High |
| Part Two has strong class imbalance | REPORT.md; RESULTS.md | Direct class counts | Primary supplied data | Yes | Full run completed | High |
| ComplementNB is appropriate to compare on imbalanced sparse text | REPORT.md; RESEARCH_EVIDENCE.md | Rennie et al. (2003); official ComplementNB docs | Primary research and official docs | Yes | Cross-checked | High |
| The selected custom model uses 2,000 features and achieved macro-F1 0.5613 | REPORT.md; RESULTS.md | Direct executed output | Primary project result | Yes | Complete test run | High |
| The custom setup was chosen without using the test set | PartTwo.py; VALIDATION.md | Executable Pipeline and five-fold training-only CV | Primary code | Yes | Code and outputs inspected | High |
| Few-shot demonstrations alter context without changing model weights | REPORT.md Part Three | Nulty (2026c); Chung et al. (2022) | Teaching source and original research | Yes | Cross-checked | High |
| FLAN-T5-small is a 77M-parameter Apache-2.0 instruction-tuned model | REPORT.md; RESEARCH_EVIDENCE.md | Official Google model card; Chung et al. (2022) | Official model source and original research | Yes | Model card verified | High |
| Zero-shot or few-shot achieved a particular score | REPORT.md Part Three; RESULTS.md | Executed PartThree.py outputs | Primary project result | Yes | Full execution completed; Macro F1 0.0000 verified | High |
| The replacement University policy requires disclosure and prohibits concealment | AI_USE_DECLARATION.md | KUniversity_AI_Policy (1) (2).docx, sections 6.4 and 7 | User-supplied policy | Yes | Text extracted directly | High |

All Part Three performance claims are verified directly from complete model execution and saved in results/part_three_predictions.csv.

