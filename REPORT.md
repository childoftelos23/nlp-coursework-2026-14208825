# COIY064H7 Programming Portfolio - Short Written Answers

## Part One decisions

I processed all twelve text files because the brief says to use the supplied files and never says to remove the twentieth-century books. This matters because the folder includes publication years from 1811 to 1930 even though the section heading describes nineteenth-century novels.

For TTR I used NLTK tokenisation, removed tokens that contained no letters, and used lower case before counting. This makes Cat and cat one type and stops punctuation becoming a type. For readability I used the Flesch Reading Ease formula requested in the question. CMUdict supplies syllables where a pronunciation exists. Unknown words use contiguous vowel groups as a simple estimate.

For the parsing questions I used lower-case lemmas so different forms of a word could be counted together. I kept him and her as surface forms because spaCy otherwise changes them to he and she. PMI is calculated over all verb-direct-object dependency pairs in one novel. I also printed the joint count beside PMI because rare pairs can receive very high PMI.

## Part Two (e): tokenizer explanation and performance

<!-- PART2_DISCUSSION_START -->
The main idea of my tokeniser was to make the speeches more consistent without turning the cleaning into something too complicated. First, a regular expression keeps alphabetic words and words with an internal apostrophe. The text is changed to lower case, English stopwords are removed, and tokens shorter than three letters are ignored. I then used NLTK's Snowball stemmer. This groups related forms such as argue, argued and arguing into a smaller feature space.

The vectorizer uses unigrams and bigrams. Unigrams keep the normal bag-of-words evidence, while bigrams recover a small amount of word order that a basic term-document matrix loses. I used min_df=2 so a feature had to occur in at least two training speeches. Sublinear term frequency reduces the influence of a word being repeated many times in one long speech.

I did not choose the custom setup from the test results. I compared 1,000, 2,000 and 3,000 feature limits with both required classifiers using five-fold stratified cross-validation on the training set. ComplementNB with 2,000 features was selected with mean macro-F1 0.5112 and standard deviation 0.0515. It then achieved 0.5613 macro-F1 on the untouched test set.

This was better than default TF-IDF with ComplementNB, which scored 0.4830 using 3,000 features. It was also better than the unigram-to-trigram ComplementNB result of 0.4845. Therefore, the custom tokenizer gave the strongest measured performance while using one third fewer features than the allowed maximum.

The largest limitation was class imbalance. There were 1,248 Conservative speeches after cleaning but only 72 Liberal Democrat speeches. The custom model found only 2 of the 15 Liberal Democrat test speeches, giving recall 0.1333. Macro-F1 is therefore more informative than accuracy because every party contributes equally. The result comes from one held-out split, so it should not be treated as proof that the tokeniser will generalise to other years, parliaments or forms of political text.
<!-- PART2_DISCUSSION_END -->

Word count: 321.

## Part Three (a): model and settings

I chose google/flan-t5-small, an Apache-2.0 open-weight instruction-tuned sequence-to-sequence model with 77 million parameters, accessed locally through Hugging Face Transformers. I pinned revision 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.

The settings are do_sample=False, num_beams=1, max_new_tokens=8, max_input_tokens=512 and random seed 26. Greedy decoding is useful here because this is classification, so repeatability and one short label matter more than creative variation. The small checkpoint was chosen because it can run on CPU, but its limited scale is also an important performance limitation.

Repeating Part Two's 1,000-character filter would leave only one Liberal Democrat item in hansard500.csv, making a stratified 80/20 split impossible. I therefore reused the same four labels, removed non-speech rows and used the same split settings, but did not repeat the length filter. This decision is printed by the script.

## Part Three exact prompts

The zero-shot template is:

    Classify this UK Parliamentary speech by political party.

    Allowed labels:
    Conservative
    Labour
    Scottish National Party
    Liberal Democrat

    Return exactly one allowed label and nothing else.

    Speech:
    {speech}

    Label:

The few-shot prompt uses the same instruction and labels, followed by four demonstrations in this exact form:

    Example {number} speech:
    {training speech}
    Example {number} label: {party}

The prompt then ends with:

    Speech to classify:
    {test speech}

    Label:

PartThree.py prints the complete exact prompt, including the selected speeches. One example per party is selected from training data only by choosing the speech closest to that party's median length. This gives every label a demonstration without choosing examples from the test results.

## Part Three (d): comparison

<!-- PART3_DISCUSSION_START -->
The zero-shot and few-shot evaluation used identical local model weights (`google/flan-t5-small`), test split (92 speeches), deterministic greedy decoding (`do_sample=False`, `num_beams=1`), and strict output parser. In-context learning via four training-set demonstrations was intended to guide the model towards single-label outputs without parameter updates.

Both zero-shot and few-shot setups achieved Macro F1 scores of 0.0000 across all 92 test speeches, with 92 invalid model outputs. Rather than returning a single political party label, the small 77M parameter model consistently echoed the full set of candidate labels provided in the prompt instructions (generating raw text such as `"Conservative Labour Scottish National Party Liberal Democrat"`). Because the parser strictly rejects multi-label outputs as `__INVALID__`, zero precision and recall were recorded.

These empirical results demonstrate the capacity boundaries of small language models for zero-shot and few-shot classification. While instruction-tuned sequence-to-sequence models can follow simple formatting cues in larger parameter regimes, `flan-t5-small` lacks sufficient capacity to isolate and ground specific class decisions from long text context when prompted without fine-tuning. Consequently, supervised TF-IDF classifiers (such as ComplementNB in Part Two, Macro F1 0.5613) strongly outperform small open-weight LLMs in zero-shot/few-shot regimes on this dataset.
<!-- PART3_DISCUSSION_END -->

Word count: 202.

## Wider limitations and ethics

Party prediction from speech is an observational text-classification task. The outputs do not prove ideology, intention or party membership beyond the supplied label. The sample is strongly imbalanced and tied to a particular political setting and time period. This creates unequal error rates, especially for Liberal Democrats. A real deployment could misprofile speakers or be used to make unsupported political inferences, so these models are suitable for coursework comparison, not consequential profiling. The row-level error file and per-class report make that weakness visible.
