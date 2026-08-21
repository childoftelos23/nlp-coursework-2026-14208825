"""COIY064H7 reassessment - Part Three: Zero-shot and Few-shot LLM Classification.

The chosen model is google/flan-t5-small through Hugging Face Transformers.
It is open-weight and small enough to run locally on an ordinary CPU. This is
more reproducible than relying on a paid API, although the small size also
limits the classification performance.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 26
MODEL_NAME = "google/flan-t5-small"
MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
PARTY_LABELS = [
    "Conservative",
    "Labour",
    "Scottish National Party",
    "Liberal Democrat",
]
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 8
INPUT_SPEECH_CHARACTERS = 650
EXAMPLE_SPEECH_CHARACTERS = 180


ZERO_SHOT_TEMPLATE = """Classify this UK Parliamentary speech by political party.

Allowed labels:
Conservative
Labour
Scottish National Party
Liberal Democrat

Return exactly one allowed label and nothing else.

Speech:
{speech}

Label:"""


def find_part_two_labels(
    csv_path: Path | str,
) -> list[str]:
    """Get the four labels from Part Two after merging Labour (Co-op)."""

    dataframe = pd.read_csv(csv_path)
    dataframe["party"] = dataframe["party"].replace(
        {"Labour (Co-op)": "Labour"}
    )
    return (
        dataframe["party"]
        .value_counts()
        .head(4)
        .index
        .tolist()
    )


def prepare_llm_data(
    sample_path: Path | str,
    label_source_path: Path | str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare hansard500 with the Part Two label set.

    The 1,000-character filter is not repeated here. If it was applied to the
    sample it would leave only one Liberal Democrat speech, which makes the
    required stratified train/test split impossible. Part Three asks for the
    same labels and split, but does not repeat that length filter.
    """

    labels = find_part_two_labels(label_source_path)
    if labels != PARTY_LABELS:
        raise ValueError(
            f"Unexpected Part Two label set: {labels}"
        )

    dataframe = pd.read_csv(sample_path)
    original_shape = dataframe.shape
    dataframe["party"] = dataframe["party"].replace(
        {"Labour (Co-op)": "Labour"}
    )
    dataframe = dataframe[
        dataframe["party"].isin(labels)
    ].copy()
    dataframe = dataframe[
        dataframe["speech_class"] == "Speech"
    ].copy()
    dataframe.reset_index(drop=True, inplace=True)

    long_only_counts = (
        dataframe[
            dataframe["speech"].astype(str).str.len() >= 1_000
        ]["party"]
        .value_counts()
        .reindex(labels, fill_value=0)
        .to_dict()
    )
    audit = {
        "original_shape": original_shape,
        "label_set_from_part_two": labels,
        "final_shape": dataframe.shape,
        "final_class_counts": (
            dataframe["party"]
            .value_counts()
            .reindex(labels, fill_value=0)
            .to_dict()
        ),
        "counts_if_1000_character_filter_was_repeated": long_only_counts,
        "length_filter_decision": (
            "Not repeated because one class would have only one item, "
            "making the required stratified split impossible."
        ),
    }
    return dataframe, audit


def split_llm_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make the same 80/20 stratified split and seed used in Part Two."""

    train, test = train_test_split(
        dataframe,
        test_size=0.2,
        stratify=dataframe["party"],
        random_state=RANDOM_SEED,
    )
    return (
        train.reset_index(drop=False).rename(
            columns={"index": "source_index"}
        ),
        test.reset_index(drop=False).rename(
            columns={"index": "source_index"}
        ),
    )


def shorten(text: str, maximum_characters: int) -> str:
    """Keep prompts inside the small model's context window."""

    clean_text = " ".join(str(text).split())
    if len(clean_text) <= maximum_characters:
        return clean_text
    return clean_text[:maximum_characters].rsplit(" ", 1)[0] + "..."


def choose_few_shot_examples(
    training_data: pd.DataFrame,
) -> list[dict[str, str]]:
    """Choose one representative training speech from each party.

    For each class I choose the speech closest to that class's median length.
    This is deterministic, uses training data only, and avoids picking examples
    because they happen to give a better test score.
    """

    examples = []
    for label in PARTY_LABELS:
        candidates = training_data[
            training_data["party"] == label
        ].copy()
        if candidates.empty:
            raise ValueError(
                f"No training example is available for {label}"
            )

        candidates["speech_length"] = (
            candidates["speech"].astype(str).str.len()
        )
        median_length = candidates["speech_length"].median()
        candidates["distance_from_median"] = (
            candidates["speech_length"] - median_length
        ).abs()
        selected = candidates.sort_values(
            ["distance_from_median", "source_index"],
        ).iloc[0]
        examples.append(
            {
                "label": label,
                "speech": shorten(
                    selected["speech"],
                    EXAMPLE_SPEECH_CHARACTERS,
                ),
                "source_index": str(selected["source_index"]),
            }
        )
    return examples


def make_zero_shot_prompt(speech: str) -> str:
    return ZERO_SHOT_TEMPLATE.format(
        speech=shorten(
            speech,
            INPUT_SPEECH_CHARACTERS,
        )
    )


def make_few_shot_prompt(
    speech: str,
    examples: list[dict[str, str]],
) -> str:
    """Build one prompt containing four labelled demonstrations."""

    demonstration_text = []
    for number, example in enumerate(examples, start=1):
        demonstration_text.append(
            f"Example {number} speech:\n"
            f"{example['speech']}\n"
            f"Example {number} label: {example['label']}"
        )

    demonstrations = "\n\n".join(demonstration_text)
    return (
        "Classify a UK Parliamentary speech by political party.\n\n"
        "Allowed labels:\n"
        "Conservative\n"
        "Labour\n"
        "Scottish National Party\n"
        "Liberal Democrat\n\n"
        "The labelled examples below are demonstrations. "
        "Return exactly one allowed label and nothing else.\n\n"
        f"{demonstrations}\n\n"
        "Speech to classify:\n"
        f"{shorten(speech, INPUT_SPEECH_CHARACTERS)}\n\n"
        "Label:"
    )


def normalise_model_label(raw_answer: str) -> str:
    """Map a label-only answer to the exact class name or mark it invalid."""

    cleaned = re.sub(
        r"\s+",
        " ",
        raw_answer.strip().strip(".,:;!?\"'"),
    )
    for label in PARTY_LABELS:
        if cleaned.casefold() == label.casefold():
            return label

    matches = [
        label
        for label in PARTY_LABELS
        if label.casefold() in cleaned.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    return "__INVALID__"


def load_model(
    model_name: str,
    revision: str,
) -> tuple[Any, Any, Any, str]:
    """Load the open-weight sequence-to-sequence model locally."""

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Part Three needs torch, transformers and sentencepiece. "
            "Install the packages listed in requirements.txt."
        ) from error

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        revision=revision,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, torch, device


def generate_answers(
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    device: str,
    batch_size: int,
) -> list[str]:
    """Generate deterministic label answers in small batches."""

    answers = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        encoded = {
            name: values.to(device)
            for name, values in encoded.items()
        }
        with torch_module.inference_mode():
            output_tokens = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        answers.extend(
            tokenizer.batch_decode(
                output_tokens,
                skip_special_tokens=True,
            )
        )
    return answers


def report_results(
    heading: str,
    true_labels: pd.Series,
    raw_answers: list[str],
) -> tuple[list[str], float]:
    """Print macro F1, invalid-output count and the classification report."""

    predictions = [
        normalise_model_label(answer)
        for answer in raw_answers
    ]
    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=PARTY_LABELS,
        average="macro",
        zero_division=0,
    )
    invalid_count = predictions.count("__INVALID__")

    print(f"\n{heading}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Invalid model outputs: {invalid_count}")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=PARTY_LABELS,
            digits=4,
            zero_division=0,
        )
    )
    return predictions, macro_f1


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("texts/hansard500.csv"),
    )
    parser.add_argument(
        "--label-source",
        type=Path,
        default=Path("texts/hansard10000.csv"),
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional smoke-test limit. Do not use this for final results.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/part_three_predictions.csv"),
    )
    return parser


def main() -> None:
    args = make_argument_parser().parse_args()
    dataframe, audit = prepare_llm_data(
        args.sample,
        args.label_source,
    )
    training_data, test_data = split_llm_data(dataframe)
    examples = choose_few_shot_examples(training_data)

    if args.limit is not None:
        print(
            "WARNING: --limit is active. These are smoke-test results, "
            "not the complete coursework evaluation."
        )
        test_data = test_data.head(args.limit).copy()

    print("Part Three data audit:")
    for name, value in audit.items():
        print(f"  {name}: {value}")
    print(
        "  training_class_counts:",
        training_data["party"].value_counts().to_dict(),
    )
    print(
        "  test_class_counts:",
        test_data["party"].value_counts().to_dict(),
    )

    print("\nModel and access route:")
    print(f"  model: {args.model}")
    print("  access: Hugging Face Transformers, local inference")
    print(f"  revision: {args.revision}")
    print(
        "  generation parameters:",
        {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "random_seed": RANDOM_SEED,
            "max_input_tokens": MAX_INPUT_TOKENS,
        },
    )

    zero_shot_prompts = [
        make_zero_shot_prompt(speech)
        for speech in test_data["speech"]
    ]
    few_shot_prompts = [
        make_few_shot_prompt(speech, examples)
        for speech in test_data["speech"]
    ]

    print("\nExact zero-shot prompt template:")
    print(ZERO_SHOT_TEMPLATE)
    print("\nFew-shot examples selected from training data:")
    for example in examples:
        print(
            f"  source_index={example['source_index']}, "
            f"label={example['label']}, speech={example['speech']}"
        )
    print("\nExact few-shot prompt used for the first test speech:")
    print(few_shot_prompts[0])

    model, tokenizer, torch_module, device = load_model(
        args.model,
        args.revision,
    )
    print(f"\nInference device: {device}")

    zero_raw_answers = generate_answers(
        zero_shot_prompts,
        model,
        tokenizer,
        torch_module,
        device,
        args.batch_size,
    )
    zero_predictions, zero_macro_f1 = report_results(
        "Part 3(b) zero-shot results",
        test_data["party"],
        zero_raw_answers,
    )

    few_raw_answers = generate_answers(
        few_shot_prompts,
        model,
        tokenizer,
        torch_module,
        device,
        args.batch_size,
    )
    few_predictions, few_macro_f1 = report_results(
        "Part 3(c) few-shot results",
        test_data["party"],
        few_raw_answers,
    )

    output = test_data[
        ["source_index", "party", "speech"]
    ].copy()
    output["zero_shot_raw"] = zero_raw_answers
    output["zero_shot_prediction"] = zero_predictions
    output["few_shot_raw"] = few_raw_answers
    output["few_shot_prediction"] = few_predictions
    args.predictions.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        args.predictions,
        index=False,
    )

    print("\nPart 3(d) direct comparison:")
    print(f"  zero-shot macro F1: {zero_macro_f1:.4f}")
    print(f"  few-shot macro F1: {few_macro_f1:.4f}")
    print(f"  prediction file: {args.predictions}")


if __name__ == "__main__":
    main()
