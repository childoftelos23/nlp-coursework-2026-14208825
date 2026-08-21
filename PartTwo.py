"""COIY064H7 reassessment - Part Two: Feature Extraction and Classification.

The script follows the order of the coursework question. The vectorisers are
always fitted on the training set only, which stops information from the test
set leaking into the vocabulary or IDF weights.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.base import clone
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


RANDOM_SEED = 26
MAX_FEATURES = 3_000
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STEMMER = SnowballStemmer("english")


@dataclass
class ModelResult:
    """The measurements needed to compare one fitted classifier."""

    name: str
    macro_f1: float
    feature_count: int
    report: str


def clean_hansard(
    csv_path: Path | str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the cleaning steps from Part 2(a), in the order they are given."""

    dataframe = pd.read_csv(csv_path)
    audit: dict[str, object] = {
        "original_shape": dataframe.shape,
        "original_duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_party_rows": int(dataframe["party"].isna().sum()),
    }

    dataframe["party"] = dataframe["party"].replace(
        {"Labour (Co-op)": "Labour"}
    )
    top_four_parties = (
        dataframe["party"]
        .value_counts()
        .head(4)
        .index
        .tolist()
    )
    audit["top_four_parties_after_labour_merge"] = top_four_parties

    dataframe = dataframe[
        dataframe["party"].isin(top_four_parties)
    ].copy()
    dataframe = dataframe[
        dataframe["party"] != "Speaker"
    ].copy()
    audit["after_party_filter"] = dataframe.shape

    # The PDF writes "speech class", while the supplied CSV uses speech_class.
    dataframe = dataframe[
        dataframe["speech_class"] == "Speech"
    ].copy()
    audit["after_speech_class_filter"] = dataframe.shape

    speech_lengths = dataframe["speech"].astype(str).str.len()
    dataframe = dataframe[speech_lengths >= 1_000].copy()
    dataframe.reset_index(drop=True, inplace=True)
    audit["final_shape"] = dataframe.shape
    audit["final_class_counts"] = dataframe["party"].value_counts().to_dict()
    audit["final_duplicate_rows"] = int(dataframe.duplicated().sum())
    return dataframe, audit


def split_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create the exact stratified 80/20 split requested by the brief."""

    return train_test_split(
        dataframe["speech"],
        dataframe["party"],
        test_size=0.2,
        stratify=dataframe["party"],
        random_state=RANDOM_SEED,
    )


def make_classifiers() -> dict[str, object]:
    """Make fresh versions of both classifiers required by the question."""

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1_000,
            random_state=RANDOM_SEED,
        ),
        "ComplementNB": ComplementNB(),
    }


def evaluate_models(
    vectorizer_factory: Callable[[], TfidfVectorizer],
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[ModelResult]:
    """Fit one vectoriser on training data and evaluate both classifiers."""

    vectorizer = vectorizer_factory()
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    feature_count = train_vectors.shape[1]

    answers = []
    for model_name, classifier in make_classifiers().items():
        fitted_classifier = clone(classifier)
        fitted_classifier.fit(train_vectors, y_train)
        predictions = fitted_classifier.predict(test_vectors)
        answers.append(
            ModelResult(
                name=model_name,
                macro_f1=f1_score(
                    y_test,
                    predictions,
                    average="macro",
                ),
                feature_count=feature_count,
                report=classification_report(
                    y_test,
                    predictions,
                    digits=4,
                    zero_division=0,
                ),
            )
        )
    return answers


def custom_tokenizer(text: str) -> list[str]:
    """Lower-case, remove stopwords and stem useful alphabetic tokens.

    I used a regular expression instead of splitting on spaces because this
    removes punctuation at the same time. Stemming then groups variations such
    as "argue", "argued" and "arguing" into a smaller feature set.
    """

    answers = []
    for token in TOKEN_PATTERN.findall(text.casefold()):
        token = token.strip("'")
        if (
            len(token) >= 3
            and token not in ENGLISH_STOP_WORDS
        ):
            answers.append(STEMMER.stem(token))
    return answers


def make_default_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
    )


def make_ngram_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
        ngram_range=(1, 3),
    )


def make_custom_vectorizer(
    max_features: int,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=custom_tokenizer,
        token_pattern=None,
        lowercase=False,
        stop_words=None,
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )


def select_custom_model(
    x_train: pd.Series,
    y_train: pd.Series,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Choose the custom setup by cross-validation on training data only."""

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    results = []

    for feature_limit in (1_000, 2_000, 3_000):
        for model_name, classifier in make_classifiers().items():
            pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        make_custom_vectorizer(feature_limit),
                    ),
                    ("classifier", classifier),
                ]
            )
            scores = cross_val_score(
                pipeline,
                x_train,
                y_train,
                scoring="f1_macro",
                cv=cross_validation,
                n_jobs=1,
            )
            results.append(
                {
                    "model": model_name,
                    "feature_limit": feature_limit,
                    "mean_cv_macro_f1": float(np.mean(scores)),
                    "std_cv_macro_f1": float(np.std(scores)),
                }
            )

    # Highest mean F1 wins. If scores tie, I prefer the smaller feature limit
    # because the question also asks for a good performance/efficiency balance.
    best = sorted(
        results,
        key=lambda answer: (
            -answer["mean_cv_macro_f1"],
            answer["feature_limit"],
            answer["model"],
        ),
    )[0]
    return best, results


def evaluate_selected_custom_model(
    selected: dict[str, object],
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ModelResult:
    """Fit the selected custom model once and evaluate it on the held-out test."""

    classifier = make_classifiers()[str(selected["model"])]
    vectorizer = make_custom_vectorizer(
        int(selected["feature_limit"])
    )
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    classifier.fit(train_vectors, y_train)
    predictions = classifier.predict(test_vectors)

    return ModelResult(
        name=str(selected["model"]),
        macro_f1=f1_score(
            y_test,
            predictions,
            average="macro",
        ),
        feature_count=train_vectors.shape[1],
        report=classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        ),
    )


def print_results(
    heading: str,
    results: list[ModelResult],
) -> None:
    print(f"\n{heading}")
    for result in results:
        print(
            f"\n{result.name}: macro F1 = {result.macro_f1:.4f}; "
            f"features = {result.feature_count}"
        )
        print(result.report)


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("texts/hansard10000.csv"),
    )
    return parser


def main() -> None:
    args = make_argument_parser().parse_args()
    dataframe, audit = clean_hansard(args.data)

    print("Part 2(a) cleaning audit:")
    for name, value in audit.items():
        print(f"  {name}: {value}")
    print("\nFinal dataframe shape:")
    print(dataframe.shape)

    x_train, x_test, y_train, y_test = split_data(dataframe)
    print("\nClass counts in the stratified split:")
    print("  train:", y_train.value_counts().to_dict())
    print("  test:", y_test.value_counts().to_dict())

    default_results = evaluate_models(
        make_default_vectorizer,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    print_results(
        "Part 2(b) default TF-IDF results",
        default_results,
    )

    ngram_results = evaluate_models(
        make_ngram_vectorizer,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    print_results(
        "Part 2(c) unigram, bigram and trigram results",
        ngram_results,
    )

    selected, cross_validation_results = select_custom_model(
        x_train,
        y_train,
    )
    print("\nPart 2(d) custom-tokenizer training-only cross-validation:")
    for result in cross_validation_results:
        print(
            f"  {result['model']}, limit={result['feature_limit']}: "
            f"mean macro F1={result['mean_cv_macro_f1']:.4f}, "
            f"standard deviation={result['std_cv_macro_f1']:.4f}"
        )
    print("Selected setup:", selected)

    best_custom_result = evaluate_selected_custom_model(
        selected,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    print_results(
        "Best custom-tokenizer test result",
        [best_custom_result],
    )


if __name__ == "__main__":
    main()

