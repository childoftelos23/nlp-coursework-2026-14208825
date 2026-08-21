"""COIY064H7 reassessment - Part One: Syntax and Style.

Run this from the coursework folder with:
    python PartOne.py

I have kept each task in its own function so the order follows the brief and
each answer can be checked on its own.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
import spacy
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from spacy.language import Language
from spacy.tokens import Doc


WORD_TOKENIZER = TreebankWordTokenizer()
SENTENCE_TOKENIZER = PunktSentenceTokenizer()
DIRECT_OBJECT_LABELS = {"dobj", "obj"}


def word_tokens(text: str) -> list[str]:
    """Use NLTK to tokenise and then take out punctuation-only tokens."""

    return [
        token
        for token in WORD_TOKENIZER.tokenize(text)
        if any(character.isalpha() for character in token)
    ]


def count_syl(word: str, dictionary: dict[str, Any]) -> int:
    """Count syllables from CMUdict and estimate unknown words by vowel groups."""

    clean_word = re.sub(r"[^a-z']", "", word.casefold()).strip("'")
    if not clean_word:
        return 0

    pronunciations = dictionary.get(clean_word)
    if pronunciations:
        # In CMUdict each vowel ends in a stress number, such as AH0 or AE1.
        return max(
            1,
            sum(sound[-1].isdigit() for sound in pronunciations[0]),
        )

    # The brief says unknown words can be estimated. The simple idea here is
    # that a new group of vowels usually represents a new syllable.
    return max(1, len(re.findall(r"[aeiouy]+", clean_word)))


def fk_level(text: str, dictionary: dict[str, Any]) -> float:
    """Calculate the Flesch Reading Ease score asked for in the brief."""

    words = word_tokens(text)
    sentences = [
        sentence
        for sentence in SENTENCE_TOKENIZER.tokenize(text)
        if word_tokens(sentence)
    ]
    if not words or not sentences:
        raise ValueError("A Reading Ease score needs words and sentences")

    syllables = sum(count_syl(word, dictionary) for word in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)
    return (
        206.835
        - (1.015 * words_per_sentence)
        - (84.6 * syllables_per_word)
    )


def load_cmudict() -> dict[str, Any]:
    """Load CMUdict, including a fallback that works without an NLTK download."""

    try:
        return nltk.corpus.cmudict.dict()
    except LookupError:
        try:
            import cmudict
        except ImportError as error:
            raise RuntimeError(
                "CMUdict is missing. Install cmudict or download the NLTK corpus."
            ) from error
        return cmudict.dict()


def read_novels(
    path: Path | str = Path.cwd() / "texts" / "novels",
) -> pd.DataFrame:
    """Read all supplied novels and return them in publication-year order."""

    rows: list[dict[str, Any]] = []
    novels_path = Path(path)

    for text_file in sorted(novels_path.glob("*.txt")):
        try:
            title, author, year_text = text_file.stem.rsplit("-", 2)
            year = int(year_text)
        except ValueError as error:
            raise ValueError(
                f"The filename does not match Title-Author-Year: {text_file.name}"
            ) from error

        rows.append(
            {
                "text": text_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                ),
                "title": title.replace("_", " "),
                "author": author.replace("_", " "),
                "year": year,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No novel text files were found in {novels_path}")

    dataframe = pd.DataFrame(
        rows,
        columns=["text", "title", "author", "year"],
    )
    return dataframe.sort_values("year", ignore_index=True)


def load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """Load the English parser and increase its maximum document length."""

    try:
        nlp = spacy.load(model_name)
    except OSError as error:
        raise RuntimeError(
            f"spaCy model {model_name} is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from error

    nlp.max_length = max(nlp.max_length, 2_000_000)
    return nlp


def parse(
    dataframe: pd.DataFrame,
    store_path: Path | str = Path.cwd() / "pickles",
    out_name: str = "parsed.pickle",
    nlp: Language | None = None,
) -> pd.DataFrame:
    """Parse each text, add a Doc column and save the dataframe as a pickle."""

    parsed_dataframe = dataframe.copy()
    pipeline = nlp if nlp is not None else load_spacy_model()
    longest_text = int(parsed_dataframe["text"].str.len().max())
    pipeline.max_length = max(pipeline.max_length, longest_text + 1)

    parsed_dataframe["doc"] = list(
        pipeline.pipe(
            parsed_dataframe["text"],
            batch_size=1,
        )
    )

    output_folder = Path(store_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    parsed_dataframe.to_pickle(output_folder / out_name)
    return parsed_dataframe


def nltk_ttr(text: str) -> float:
    """Calculate TTR while ignoring case and punctuation."""

    tokens = [token.casefold() for token in word_tokens(text)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def get_ttrs(dataframe: pd.DataFrame) -> dict[str, float]:
    """Map every title to its type-token ratio."""

    answers = {}
    for _, row in dataframe.iterrows():
        answers[row["title"]] = round(nltk_ttr(row["text"]), 6)
    return answers


def get_fks(dataframe: pd.DataFrame) -> dict[str, float]:
    """Map every title to its Flesch Reading Ease score."""

    dictionary = load_cmudict()
    answers = {}
    for _, row in dataframe.iterrows():
        answers[row["title"]] = round(
            fk_level(row["text"], dictionary),
            4,
        )
    return answers


def normalise_parse_token(token: Any) -> str:
    """Use a lower-case lemma and fall back to the original token if needed."""

    # I kept these two pronouns as surface forms because the question asks for
    # him and her specifically. spaCy otherwise changes them to he and she.
    if token.lower_ in {"him", "her"}:
        return token.lower_

    lemma = token.lemma_.casefold().strip()
    if lemma and lemma != "-pron-":
        return lemma
    return token.lower_.strip()


def direct_object_arcs(doc: Doc) -> list[tuple[str, str]]:
    """Return pairs of governing verb and direct-object lemmas."""

    answers = []
    for token in doc:
        if (
            token.dep_ in DIRECT_OBJECT_LABELS
            and token.head.pos_ in {"VERB", "AUX"}
        ):
            verb = normalise_parse_token(token.head)
            direct_object = normalise_parse_token(token)
            if direct_object and any(
                character.isalpha() for character in direct_object
            ):
                answers.append((verb, direct_object))
    return answers


def top_direct_objects(doc: Doc, top_n: int = 10) -> list[tuple[str, int]]:
    """Return the ten most common direct objects and their counts."""

    object_counts = Counter(
        direct_object
        for _, direct_object in direct_object_arcs(doc)
    )
    return object_counts.most_common(top_n)


def pmi_verbs_for_object(
    doc: Doc,
    target_object: str,
    top_n: int = 10,
) -> list[tuple[str, float, int]]:
    """Rank verbs connected to one object using pointwise mutual information."""

    arcs = direct_object_arcs(doc)
    if not arcs:
        return []

    target_object = target_object.casefold()
    joint_counts = Counter(arcs)
    verb_counts = Counter(verb for verb, _ in arcs)
    object_counts = Counter(direct_object for _, direct_object in arcs)
    target_count = object_counts[target_object]

    if target_count == 0:
        return []

    total_arcs = len(arcs)
    scores = []
    for (verb, direct_object), joint_count in joint_counts.items():
        if direct_object == target_object:
            score = math.log2(
                (joint_count * total_arcs)
                / (verb_counts[verb] * target_count)
            )
            # included the raw count because PMI can make rare pairs look
            # stronger than they really are.
            scores.append((verb, round(score, 4), joint_count))

    scores.sort(
        key=lambda result: (
            -result[1],
            -result[2],
            result[0],
        )
    )
    return scores[:top_n]


def print_dependency_results(dataframe: pd.DataFrame) -> None:
    """Print all the answers requested in Part 1(e)."""

    for _, row in dataframe.iterrows():
        print(f"\n{row['title']}")
        print(
            "  Ten most common direct objects:",
            top_direct_objects(row["doc"]),
        )
        print(
            "  Ten PMI verbs with him:",
            pmi_verbs_for_object(row["doc"], "him"),
        )
        print(
            "  Ten PMI verbs with her:",
            pmi_verbs_for_object(row["doc"], "her"),
        )


def make_argument_parser() -> argparse.ArgumentParser:
    """Keep file paths adjustable without making the main code confusing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--novels",
        type=Path,
        default=Path("texts/novels"),
    )
    parser.add_argument(
        "--pickle",
        type=Path,
        default=Path("pickles/parsed.pickle"),
    )
    parser.add_argument(
        "--reuse-pickle",
        action="store_true",
        help="Use a pickle already created instead of parsing every novel again.",
    )
    return parser


def main() -> None:
    args = make_argument_parser().parse_args()
    dataframe = read_novels(args.novels)

    print("Novels sorted by year:")
    print(
        dataframe[["title", "author", "year"]].to_string(
            index=False,
        )
    )

    print("\nType-token ratios:")
    print(get_ttrs(dataframe))

    print("\nFlesch Reading Ease scores:")
    print(get_fks(dataframe))

    if not args.reuse_pickle:
        parse(
            dataframe,
            args.pickle.parent,
            args.pickle.name,
        )

    if not args.pickle.exists():
        raise FileNotFoundError(
            f"No parsed dataframe exists at {args.pickle}. "
            "Run once without --reuse-pickle."
        )

    reloaded_dataframe = pd.read_pickle(args.pickle)
    print_dependency_results(reloaded_dataframe)


if __name__ == "__main__":
    main()
