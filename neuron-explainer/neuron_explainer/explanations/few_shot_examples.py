# Few-shot examples for generating and simulating neuron explanations.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.fast_dataclasses import FastDataclass


@dataclass
class Example(FastDataclass):
    activation_records: List[ActivationRecord]
    explanation: str
    first_revealed_activation_indices: List[int]
    """
    For each activation record, the index of the first token for which the activation value in the
    prompt should be an actual number rather than "unknown".

    Examples all start with the activations rendered as "unknown", then transition to revealing
    specific normalized activation values. The goal is to lead the model to predict that activation
    sequences will eventually transition to predicting specific activation values instead of just
    "unknown". This lets us cheat and get predictions of activation values for every token in a
    single round of inference by having the activations in the sequence we're predicting always be
    "unknown" in the prompt: the model will always think that maybe the next token will be a real
    activation.
    """
    token_index_to_score: Optional[int] = None
    """
    If the prompt is used as an example for one-token-at-a-time scoring, this is the index of the
    token to score.
    """


class FewShotExampleSet(Enum):
    """Determines which few-shot examples to use when sampling explanations."""

    ORIGINAL = "original"
    NEWER = "newer"
    TEST = "test"

    @classmethod
    def from_string(cls, string: str) -> FewShotExampleSet:
        for example_set in FewShotExampleSet:
            if example_set.value == string:
                return example_set
        raise ValueError(f"Unrecognized example set: {string}")

    def get_examples(self) -> list[Example]:
        """Returns regular examples for use in a few-shot prompt."""
        if self is FewShotExampleSet.ORIGINAL:
            return ORIGINAL_EXAMPLES
        elif self is FewShotExampleSet.NEWER:
            return NEWER_EXAMPLES
        elif self is FewShotExampleSet.TEST:
            return TEST_EXAMPLES
        else:
            raise ValueError(f"Unhandled example set: {self}")

    def get_single_token_prediction_example(self) -> Example:
        """
        Returns an example suitable for use in a subprompt for predicting a single token's
        normalized activation, for use with the "one token at a time" scoring approach.
        """
        if self is FewShotExampleSet.NEWER:
            return NEWER_SINGLE_TOKEN_EXAMPLE
        elif self is FewShotExampleSet.TEST:
            return TEST_SINGLE_TOKEN_EXAMPLE
        else:
            raise ValueError(f"Unhandled example set: {self}")


TEST_EXAMPLES = [
    Example(
        activation_records=[
            ActivationRecord(
                tokens=["a", "b", "c"],
                activations=[1.0, 0.0, 0.0],
            ),
            ActivationRecord(
                tokens=["d", "e", "f"],
                activations=[0.0, 1.0, 0.0],
            ),
        ],
        explanation="vowels",
        first_revealed_activation_indices=[0, 1],
    ),
]

TEST_SINGLE_TOKEN_EXAMPLE = Example(
    activation_records=[
        ActivationRecord(
            activations=[0.0, 0.0, 1.0],
            tokens=["g", "h", "i"],
        ),
    ],
    first_revealed_activation_indices=[],
    token_index_to_score=2,
    explanation="test explanation",
)


ORIGINAL_EXAMPLES = [
    Example(
        activation_records=[
            ActivationRecord(
                tokens=[
                    "t",
                    "urt",
                    "ur",
                    "ro",
                    " is",
                    " fab",
                    "ulously",
                    " funny",
                    " and",
                    " over",
                    " the",
                    " top",
                    " as",
                    " a",
                    " '",
                    "very",
                    " sneaky",
                    "'",
                    " but",
                    "ler",
                    " who",
                    " excel",
                    "s",
                    " in",
                    " the",
                    " art",
                    " of",
                    " impossible",
                    " disappearing",
                    "/",
                    "re",
                    "app",
                    "earing",
                    " acts",
                ],
                activations=[
                    -0.71,
                    -1.85,
                    -2.39,
                    -2.58,
                    -1.34,
                    -1.92,
                    -1.69,
                    -0.84,
                    -1.25,
                    -1.75,
                    -1.42,
                    -1.47,
                    -1.51,
                    -0.8,
                    -1.89,
                    -1.56,
                    -1.63,
                    0.44,
                    -1.87,
                    -2.55,
                    -2.09,
                    -1.76,
                    -1.33,
                    -0.88,
                    -1.63,
                    -2.39,
                    -2.63,
                    -0.99,
                    2.83,
                    -1.11,
                    -1.19,
                    -1.33,
                    4.24,
                    -1.51,
                ],
            ),
            ActivationRecord(
                tokens=[
                    "esc",
                    "aping",
                    " the",
                    " studio",
                    " ,",
                    " pic",
                    "col",
                    "i",
                    " is",
                    " warm",
                    "ly",
                    " affecting",
                    " and",
                    " so",
                    " is",
                    " this",
                    " ad",
                    "roit",
                    "ly",
                    " minimalist",
                    " movie",
                    " .",
                ],
                activations=[
                    -0.69,
                    4.12,
                    1.83,
                    -2.28,
                    -0.28,
                    -0.79,
                    -2.2,
                    -2.03,
                    -1.77,
                    -1.71,
                    -2.44,
                    1.6,
                    -1,
                    -0.38,
                    -1.93,
                    -2.09,
                    -1.63,
                    -1.94,
                    -1.82,
                    -1.64,
                    -1.32,
                    -1.92,
                ],
            ),
        ],
        first_revealed_activation_indices=[10, 3],
        explanation="present tense verbs ending in 'ing'",
    ),
    Example(
        activation_records=[
            ActivationRecord(
                tokens=[
                    "as",
                    " sac",
                    "char",
                    "ine",
                    " movies",
                    " go",
                    " ,",
                    " this",
                    " is",
                    " likely",
                    " to",
                    " cause",
                    " massive",
                    " cardiac",
                    " arrest",
                    " if",
                    " taken",
                    " in",
                    " large",
                    " doses",
                    " .",
                ],
                activations=[
                    -0.14,
                    -1.37,
                    -0.68,
                    -2.27,
                    -1.46,
                    -1.11,
                    -0.9,
                    -2.48,
                    -2.07,
                    -3.49,
                    -2.16,
                    -1.79,
                    -0.23,
                    -0.04,
                    4.46,
                    -1.02,
                    -2.26,
                    -2.95,
                    -1.49,
                    -1.46,
                    -0.6,
                ],
            ),
            ActivationRecord(
                tokens=[
                    "shot",
                    " perhaps",
                    " '",
                    "art",
                    "istically",
                    "'",
                    " with",
                    " handheld",
                    " cameras",
                    " and",
                    " apparently",
                    " no",
                    " movie",
                    " lights",
                    " by",
                    " jo",
                    "aquin",
                    " b",
                    "aca",
                    "-",
                    "as",
                    "ay",
                    " ,",
                    " the",
                    " low",
                    "-",
                    "budget",
                    " production",
                    " swings",
                    " annoy",
                    "ingly",
                    " between",
                    " vert",
                    "igo",
                    " and",
                    " opacity",
                    " .",
                ],
                activations=[
                    -0.09,
                    -3.53,
                    -0.72,
                    -2.36,
                    -1.05,
                    -1.12,
                    -2.49,
                    -2.14,
                    -1.98,
                    -1.59,
                    -2.62,
                    -2,
                    -2.73,
                    -2.87,
                    -3.23,
                    -1.11,
                    -2.23,
                    -0.97,
                    -2.28,
                    -2.37,
                    -1.5,
                    -2.81,
                    -1.73,
                    -3.14,
                    -2.61,
                    -1.7,
                    -3.08,
                    -4,
                    -0.71,
                    -2.48,
                    -1.39,
                    -1.96,
                    -1.09,
                    4.37,
                    -0.74,
                    -0.5,
                    -0.62,
                ],
            ),
        ],
        first_revealed_activation_indices=[5, 20],
        explanation="words related to physical medical conditions",
    ),
    Example(
        activation_records=[
            ActivationRecord(
                tokens=[
                    "the",
                    " sense",
                    " of",
                    " together",
                    "ness",
                    " in",
                    " our",
                    " town",
                    " is",
                    " strong",
                    " .",
                ],
                activations=[
                    0,
                    0,
                    0,
                    1,
                    2,
                    0,
                    0.23,
                    0.5,
                    0,
                    0,
                    0,
                ],
            ),
            ActivationRecord(
                tokens=[
                    "a",
                    " buoy",
                    "ant",
                    " romantic",
                    " comedy",
                    " about",
                    " friendship",
                    " ,",
                    " love",
                    " ,",
                    " and",
                    " the",
                    " truth",
                    " that",
                    " we",
                    "'re",
                    " all",
                    " in",
                    " this",
                    " together",
                    " .",
                ],
                activations=[
                    -0.15,
                    -2.33,
                    -1.4,
                    -2.17,
                    -2.53,
                    -0.85,
                    0.23,
                    -1.89,
                    0.09,
                    -0.47,
                    -0.5,
                    -0.58,
                    -0.87,
                    0.22,
                    0.58,
                    1.34,
                    0.98,
                    2.21,
                    2.84,
                    1.7,
                    -0.89,
                ],
            ),
        ],
        first_revealed_activation_indices=[0, 10],
        explanation="phrases related to community",
    ),
]


NEWER_EXAMPLES = [
    Example(
        activation_records=[
            ActivationRecord(
                tokens=[
                    "The",
                    " editors",
                    " of",
                    " Bi",
                    "opol",
                    "ym",
                    "ers",
                    " are",
                    " delighted",
                    " to",
                    " present",
                    " the",
                    " ",
                    "201",
                    "8",
                    " Murray",
                    " Goodman",
                    " Memorial",
                    " Prize",
                    " to",
                    " Professor",
                    " David",
                    " N",
                    ".",
                    " Ber",
                    "atan",
                    " in",
                    " recognition",
                    " of",
                    " his",
                    " seminal",
                    " contributions",
                    " to",
                    " bi",
                    "oph",
                    "ysics",
                    " and",
                    " their",
                    " impact",
                    " on",
                    " our",
                    " understanding",
                    " of",
                    " charge",
                    " transport",
                    " in",
                    " biom",
                    "olecules",
                    ".\n\n",
                    "In",
                    "aug",
                    "ur",
                    "ated",
                    " in",
                    " ",
                    "200",
                    "7",
                    " in",
                    " honor",
                    " of",
                    " the",
                    " Bi",
                    "opol",
                    "ym",
                    "ers",
                    " Found",
                    "ing",
                    " Editor",
                    ",",
                    " the",
                    " prize",
                    " is",
                    " awarded",
                    " for",
                    " outstanding",
                    " accomplishments",
                ],
                activations=[
                    0,
                    0.01,
                    0.01,
                    0,
                    0,
                    0,
                    -0.01,
                    0,
                    -0.01,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.04,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.39,
                    0.12,
                    0,
                    -0.01,
                    0,
                    0,
                    0,
                    0,
                    -0,
                    0,
                    -0,
                    0,
                    0,
                    -0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -0,
                    0,
                    0,
                    -0.01,
                    0,
                    0.41,
                    0,
                    0,
                    0,
                    -0.01,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
            # We sometimes exceed the max context size when this is included :(
            # ActivationRecord(
            #     tokens=[
            #         " We",
            #         " are",
            #         " proud",
            #         " of",
            #         " our",
            #         " national",
            #         " achievements",
            #         " in",
            #         " mastering",
            #         " all",
            #         " aspects",
            #         " of",
            #         " the",
            #         " fuel",
            #         " cycle",
            #         ".",
            #         " The",
            #         " current",
            #         " international",
            #         " interest",
            #         " in",
            #         " closing",
            #         " the",
            #         " fuel",
            #         " cycle",
            #         " is",
            #         " a",
            #         " vind",
            #         "ication",
            #         " of",
            #         " Dr",
            #         ".",
            #         " B",
            #         "hab",
            #         "ha",
            #         "’s",
            #         " pioneering",
            #         " vision",
            #         " and",
            #         " genius",
            #     ],
            #     activations=[
            #         -0,
            #         -0,
            #         0,
            #         -0,
            #         -0,
            #         0,
            #         0,
            #         0,
            #         -0,
            #         0,
            #         0,
            #         -0,
            #         0,
            #         -0.01,
            #         0,
            #         0,
            #         -0,
            #         -0,
            #         0,
            #         0,
            #         0,
            #         -0,
            #         -0,
            #         -0.01,
            #         0,
            #         0,
            #         -0,
            #         0,
            #         0,
            #         0,
            #         0,
            #         0,
            #         -0,
            #         0,
            #         0,
            #         0,
            #         2.15,
            #         0,
            #         0,
            #         0.03,
            #     ],
            # ),
        ],
        first_revealed_activation_indices=[7],  # , 19],
        explanation="language related to something being groundbreaking",
    ),
    Example(
        activation_records=[
            ActivationRecord(
                tokens=[
                    '{"',
                    "widget",
                    "Class",
                    '":"',
                    "Variant",
                    "Matrix",
                    "Widget",
                    '","',
                    "back",
                    "order",
                    "Message",
                    '":"',
                    "Back",
                    "ordered",
                    '","',
                    "back",
                    "order",
                    "Message",
                    "Single",
                    "Variant",
                    '":"',
                    "This",
                    " item",
                    " is",
                    " back",
                    "ordered",
                    '.","',
                    "ordered",
                    "Selection",
                    '":',
                    "true",
                    ',"',
                    "product",
                    "Variant",
                    "Id",
                    '":',
                    "0",
                    ',"',
                    "variant",
                    "Id",
                    "Field",
                    '":"',
                    "product",
                    "196",
                    "39",
                    "_V",
                    "ariant",
                    "Id",
                    '","',
                    "back",
                    "order",
                    "To",
                    "Message",
                    "Single",
                    "Variant",
                    '":"',
                    "This",
                    " item",
                    " is",
                    " back",
                    "ordered",
                    " and",
                    " is",
                    " expected",
                    " by",
                    " {",
                    "0",
                    "}.",
                    '","',
                    "low",
                    "Price",
                    '":',
                    "999",
                    "9",
                    ".",
                    "0",
                    ',"',
                    "attribute",
                    "Indexes",
                    '":[',
                    '],"',
                    "productId",
                    '":',
                    "196",
                    "39",
                    ',"',
                    "price",
                    "V",
                    "ariance",
                    '":',
                    "true",
                    ',"',
                ],
                activations=[
                    0,
                    0,
                    0,
                    0,
                    4.2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.7,
                    0,
                    0,
                    0,
                    0,
                    4.02,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.5,
                    3.7,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2.9,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2.3,
                    2.24,
                    0,
                    0,
                    0,
                ],
            ),
            ActivationRecord(
                tokens=[
                    "A",
                    " regular",
                    " look",
                    " at",
                    " the",
                    " ups",
                    " and",
                    " downs",
                    " of",
                    " variant",
                    " covers",
                    " in",
                    " the",
                    " comics",
                    " industry",
                    "…\n\n",
                    "Here",
                    " are",
                    " the",
                    " Lego",
                    " variant",
                    " sketch",
                    " covers",
                    " by",
                    " Leon",
                    "el",
                    " Cast",
                    "ell",
                    "ani",
                    " for",
                    " a",
                    " variety",
                    " of",
                    " Marvel",
                    " titles",
                    ",",
                ],
                activations=[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    6.52,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1.62,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.23,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
        ],
        first_revealed_activation_indices=[2, 8],
        explanation="the word “variant” and other words with the same ”vari” root",
    ),
]


NEWER_SINGLE_TOKEN_EXAMPLE = Example(
    activation_records=[
        ActivationRecord(
            tokens=[
                "B",
                "10",
                " ",
                "111",
                " MON",
                "DAY",
                ",",
                " F",
                "EB",
                "RU",
                "ARY",
                " ",
                "11",
                ",",
                " ",
                "201",
                "9",
                " DON",
                "ATE",
                "fake higher scoring token",  # See below.
            ],
            activations=[
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.37,
                # This fake activation makes the previous token's activation normalize to 8, which
                # might help address overconfidence in "10" activations for the one-token-at-a-time
                # scoring prompt. This value and the associated token don't actually appear anywhere
                # in the prompt.
                0.45,
            ],
        ),
    ],
    first_revealed_activation_indices=[],
    token_index_to_score=18,
    explanation="instances of the token 'ate' as part of another word",
)
