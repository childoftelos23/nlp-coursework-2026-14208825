# Structured Research Evidence


## 1. Church and Hanks (1990)

- Full citation: Church, K.W. and Hanks, P. (1990), ‘Word Association Norms, Mutual Information, and Lexicography’, Computational Linguistics, 16(1), 22–29.
- Source type: Original peer-reviewed research.
- Research question: Can mutual information give an objective corpus-based measure of word association?
- Method: Derives and applies an information-theoretic association measure to lexical co-occurrences.
- Data or sample: Large machine-readable corpora available to the authors at the time.
- Main findings: Mutual information can rank lexical associations using observed joint and marginal probabilities.
- Limitations: PMI is sensitive to low-frequency events, so rare pairs can rank highly.
- Relevance: Direct basis for the verb-object PMI calculation and the decision to print counts beside scores.
- Claim supported: PMI compares observed co-occurrence against the independence expectation.
- Verification status: Verified through the ACL Anthology record and paper.

## 2. Spärck Jones (1972)

- Full citation: Spärck Jones, K. (1972), ‘A Statistical Interpretation of Term Specificity and Its Application in Retrieval’, Journal of Documentation, 28(1), 11–21.
- Source type: Seminal peer-reviewed original research.
- Research question: How can term specificity improve retrieval?
- Method: Statistical weighting based on how broadly terms occur across documents.
- Data or sample: Information-retrieval collections used in the original study.
- Main findings: Terms occurring in fewer documents can be more discriminative than ubiquitous terms.
- Limitations: The original retrieval setting is not identical to supervised party classification.
- Relevance: The theoretical basis for inverse document frequency.
- Claim supported: TF-IDF emphasises terms that distinguish documents rather than terms appearing everywhere.
- Verification status: Verified through the publisher record and bibliographic cross-check.

## 3. Rennie et al. (2003)

- Full citation: Rennie, J.D.M., Shih, L., Teevan, J. and Karger, D.R. (2003), ‘Tackling the Poor Assumptions of Naive Bayes Text Classifiers’, ICML, 616–623.
- Source type: Original peer-reviewed conference research.
- Research question: How can weaknesses of multinomial Naive Bayes for text classification be reduced?
- Method: Develops Complement Naive Bayes and related transformations; compares text-classification performance.
- Data or sample: Standard text-classification corpora used in the paper.
- Main findings: Complement information reduces some skew and poor independence-assumption effects.
- Limitations: Results depend on the corpora and preprocessing; this does not guarantee superior performance on Hansard.
- Relevance: Explains why ComplementNB is a reasonable efficient model for sparse and imbalanced text.
- Claim supported: ComplementNB was designed to address weaknesses of multinomial Naive Bayes and is appropriate to evaluate with imbalanced classes.
- Verification status: Verified against the original-paper citation and scikit-learn's official implementation documentation.

## 4. scikit-learn TfidfVectorizer documentation

- Full citation: scikit-learn developers (2026), ‘TfidfVectorizer’.
- Source type: Official software documentation.
- Research question: Not applicable; API specification.
- Method and data: Documents parameters and the implemented transformation.
- Main findings: TfidfVectorizer converts raw documents into TF-IDF feature matrices and supports stopwords, maximum features, n-gram ranges and custom tokenizers.
- Limitations: Documentation specifies behaviour but is not evidence that one setting is best for this dataset.
- Relevance: Verifies every vectorizer parameter used in Part Two.
- Claim supported: The implementation can use English stopwords, a 3,000-feature cap, n-grams and a callable tokenizer.
- Verification status: Verified from the official documentation.

## 5. scikit-learn model and split documentation

- Full citation: scikit-learn developers (2026), documentation for train_test_split, LogisticRegression, ComplementNB and Pipeline.
- Source type: Official software documentation.
- Research question: Not applicable; API specifications.
- Method and data: Documents exact arguments and estimator behaviour.
- Main findings: train_test_split accepts test size, random state and stratification; LogisticRegression handles sparse input; ComplementNB is intended for imbalanced data; Pipeline applies transformations and prediction sequentially.
- Limitations: API documentation does not replace dataset-specific validation.
- Relevance: Supports the reproducible split and training-only pipeline.
- Claim supported: The parameters in the brief are implemented as intended and the custom cross-validation does not pre-fit TF-IDF globally.
- Verification status: Verified from official documentation.

## 6. spaCy API documentation

- Full citation: spaCy developers (2026), ‘Library Architecture’.
- Source type: Official software documentation.
- Research question: Not applicable; API and data-model specification.
- Method and data: Documents the Language pipeline and Doc object.
- Main findings: A Language object tokenises text and applies pipeline components to create a Doc containing linguistic annotations, including dependency parsing when the model provides it.
- Limitations: A small statistical model can make parsing errors on historical and non-standard text.
- Relevance: Supports using en_core_web_sm Doc objects for direct-object extraction.
- Claim supported: Dependency labels and heads can be accessed from parsed Doc tokens.
- Verification status: Verified from official documentation and full execution.

## 7. Chung et al. (2022)

- Full citation: Chung, H.W. et al. (2022), ‘Scaling Instruction-Finetuned Language Models’, arXiv:2210.11416.
- Source type: Original research preprint associated with the released FLAN-T5 checkpoints.
- Research question: How do task count, model scale and instruction tuning affect zero-shot and few-shot generalisation?
- Method: Instruction-tunes several model families on large mixtures of tasks and evaluates across zero-shot, few-shot and reasoning benchmarks.
- Data or sample: Approximately 1,800 training tasks plus diverse evaluation benchmarks.
- Main findings: Instruction tuning improves model usability and performance across prompting settings; FLAN-T5 checkpoints are publicly released.
- Limitations: Broad benchmark results do not predict performance on this small, imbalanced Hansard classification task.
- Relevance: Justifies choosing an instruction-tuned open-weight model for label generation.
- Claim supported: FLAN-T5 is designed for instruction-following and zero/few-shot use.
- Verification status: Verified from arXiv and the model card.

## 8. Google FLAN-T5-small model card

- Full citation: Google (2026), ‘Model Card for FLAN-T5 small’, Hugging Face.
- Source type: Official model documentation.
- Research question: Not applicable; model provenance, use and risk record.
- Method and data: Documents an instruction-tuned T5 checkpoint trained on a large task mixture.
- Main findings: The checkpoint is a 77-million-parameter, Apache-2.0 model with CPU usage examples.
- Limitations: It was not tested for this task; the card warns about bias, harmful generation and lack of real-world validation.
- Relevance: Supports model identity, access route, size, licence and limitations.
- Claim supported: google/flan-t5-small is open-weight and locally runnable through Transformers.
- Verification status: Verified from the current model card and pinned repository commit.

## 9. Hugging Face generation documentation

- Full citation: Hugging Face (2026), ‘Generation’, Transformers documentation.
- Source type: Official software documentation.
- Research question: Not applicable; generation API specification.
- Method and data: Documents greedy, sampling and beam-search generation controls.
- Main findings: The generate method supports greedy decoding; generation parameters control sampling and output length.
- Limitations: It does not establish which prompt or decoding method gives the best party-classification score.
- Relevance: Verifies deterministic do_sample=False, num_beams=1 and max_new_tokens settings.
- Claim supported: Greedy label generation is reproducible when sampling is disabled.
- Verification status: Verified from official documentation.


