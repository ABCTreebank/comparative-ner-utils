import enum
from collections import Counter
from functools import cache
import re
from typing import TypeVar, Any, Iterable, Literal, overload, TypedDict, Sequence, Type, Union, Counter

import numpy as np
from numpy.typing import NDArray

from scipy.optimize import linear_sum_assignment

import torch
from torch import Tensor
from transformers import BertJapaneseTokenizer, BertForTokenClassification
import datasets
import evaluate

import abctk.obj.comparative as aoc

BertForNERWithRoot = BertForTokenClassification

@enum.unique
class NERWithRootLabel(enum.IntEnum):
    IGNORE = -100
    O = 0
    deg = 1
    prej = 2
    cont = 3
    diff = 4
    root = 5

LABEL2ID = {
    label.name : label.value
    for label in NERWithRootLabel
}

LABEL2ID_WO_IGNORE = {
    k:v
    for k, v in LABEL2ID.items()
    if k != "IGNORE"
}

ID2LABEL = {
    i:label
    for label, i in LABEL2ID.items()
    if label != "IGNORE"
}

MAX_INPUT_LENGTH = 256
"""
The maximum length of an input sentence.
"""

BERT_MODEL = "cl-tohoku/bert-base-japanese-whole-word-masking"


@cache
def get_tokenizer() -> BertJapaneseTokenizer:
    return BertJapaneseTokenizer.from_pretrained(BERT_MODEL)

ID_CLS: int = get_tokenizer().cls_token_id or 2
ID_SEP: int = get_tokenizer().sep_token_id or 3

# ====================
# I/O functions
# ====================

_RE_FEAT_ARTIFACTS = re.compile(r"^(?P<name>[a-zA-Z]+)[0-9]?")
"""
A regex that detects extra letters in feature names.
"""
@overload
def _convert_record_to_vector_internal(
    comps: list[dict[str, Any]] | dict[str, list[Any]] | None,
    input_ids: Iterable[int],
    tokens_subword: Iterable[str],
    return_type: Literal["np"],
) -> NDArray:
    ...

@overload
def _convert_record_to_vector_internal(
    comps: list[dict[str, Any]] | dict[str, list[Any]] | None,
    input_ids: Iterable[int],
    tokens_subword: Iterable[str],
    return_type: Literal["pt"],
) -> Tensor:
    ...

@overload
def _convert_record_to_vector_internal(
    comps: list[dict[str, Any]] | dict[str, list[Any]] | None,
    input_ids: Iterable[int],
    tokens_subword: Iterable[str],
    return_type: Literal["py"],
) -> list[int]:
    ...

def _convert_record_to_vector_internal(
    comps: list[dict[str, Any]] | dict[str, list[Any]] | None,
    input_ids: Iterable[int],
    tokens_subword: Iterable[str],
    return_type: Literal["np", "pt", "py"] = "py",
) -> list[int] | Tensor | NDArray:
    # ------
    # Normalize the list of comparative features
    # ------
    feat_list: list[tuple[int, int, str]]

    match comps:
        case dict():
            # entry["comp"] looks like:
            # {
            #   begin: [1, 6, 7, 8],
            #   end: [2, 9, 11, 13],
            #   label:["root", "comp", "deg", "diff"]
            # }
            # which is to be converted to
            # [(1, 2, "root"), (6, 9, "comp"), ...]
            feat_list: list[tuple[int, int, str]] = list(
                zip(
                    comps["start"],
                    comps["end"],
                    comps["label"],
                )
            )
        case list():
            # Convert into tuples
            feat_list = [
                (e["start"], e["end"], e["label"])
                for e in comps
            ]
        case None:
            # No annotation
            feat_list = []
        case _:
            raise TypeError(
                f"Expected a list or a dict, but got a {type(comps)}"
            )

    # ------
    # Make the label ID tensor
    # No alignment on the first run
    # ------
    label_ids_no_alignment: list[int] = [0] * MAX_INPUT_LENGTH
    feat_root_candidate = tuple(
        (b, e, label)
        for b, e, label in feat_list
        if label == "root"
    )

    if feat_root_candidate:
        b, e, _ = feat_root_candidate[0]
        label_ids_no_alignment[b:e] = (LABEL2ID["root"], ) * (e - b)
    
    for b, e, label in feat_list:
        if label != "root":
            if (match := _RE_FEAT_ARTIFACTS.match(label)):
                label = match.group("name") or label
            
            label_ids_no_alignment[b:e] = (LABEL2ID[label], ) * (e - b)

    # ------
    # align to subwords
    # ------
    # Define states
    label_ids = [0] * MAX_INPUT_LENGTH
    pos_word: int = -1  # the pointer to original words

    # Enumerate words and subwords from the retokenization in a parallel way
    #       pos_subword: position
    #       input_id: subword ID
    #       input_token: subword translated back
    for pos_subword, (input_id, input_token) in enumerate(
        zip(input_ids, tokens_subword)
    ):
        # ------
        # Increment the word pointer
        # ------
        if input_id == ID_SEP:
            # if it reaches the end of the sentence [SEP]
            # end the procedure
            break
        elif input_id == ID_CLS:
            # if it hits on the beginning of the sentence [CLS]
            pass
        elif input_token.startswith("##"):
            # if it's a subword
            
            # don't move the pos_word pointer 
            # keep the current one

            label_ids[pos_subword] = label_ids_no_alignment[pos_word]
        else:
            # if it is not a subword
            # increment the pos_word pointer
            pos_word += 1
            
            label_ids[pos_subword] = label_ids_no_alignment[pos_word]

    match return_type:
        case "pt":
            res = Tensor(label_ids)
            return res
        case "np":
            res = np.array(label_ids, dtype = "int")
            return res
        case "py":
            return label_ids
        case _:
            raise ValueError

E = TypeVar("E", datasets.arrow_dataset.Example, datasets.arrow_dataset.Batch)
def convert_annotation_entries_to_matrices(
    entry: E, 
    return_type: Literal["np", "pt", "py"] = "py",
) -> E:
    """
    Convert a readable data record to a computable set of indices.

    A data record looks like:
        {
            "ID": "5_BCCWJ-ABC-aa-simple", 
            "tokens": ["妻", "が", "仕事", "に", ...], 
            "comp": [
                {"start": 7, "end": 9, "label": "cont"},
                {"start": 9, "end": 11, "label": "prej"},
                ...
            ]
        }
    which is converted into:
        {
            "ID": "5_BCCWJ-ABC-aa-simple",
            "input_ids": [0, 1333, 24, 245, ...],
            "token_subwords", ["[CLS]", "妻", "が", "仕", "##事", "に", ...],
            "token_type_ids": [0, 0, 0, 0, 0, ...],
            "attention_mask": [0, 0, 0, 0, 0, ...],
            "label_ids": [0, 0, 3, 4, 0, ...],
        }

    A data batch looks like:
        {
            "ID": ["5_BCCWJ-ABC-aa-simple", "6_BCCWJ-ABC-bb-simple", ...],
            "tokens": [["妻", "が", "仕事", "に", ...], ["夫", "が", "ごろ寝", ...]
            "comp": [
                [
                    {"start": 7, "end": 9, "label": "cont"},
                    {"start": 9, "end": 11, "label": "prej"},
                    ...
                ],
                [
                    {"start": 7, "end": 9, "label": "cont"},
                    {"start": 9, "end": 11, "label": "prej"},
                    ...
                ],
                ...
            ]
        }
    which is converted into:
        {
            "ID": ["5_BCCWJ-ABC-aa-simple", "6_BCCWJ-ABC-bb-simple", ...],
            "input_ids": [
                [0, 1333, 24, 245, ...],
                [0, 123, 24, 21354, 245, ...],
                ...
            ],
            "token_subwords": [
                ["[CLS]", "妻", "が", "仕", "##事", "に", ...],
                ["[CLS", "夫", "が", "ごろ", "##寝", ...],
            ]
            "token_type_ids": [
                [0, 1, 1, 1, 1, ...],
                [0, 1, 1, 1, 1, ...],
            ],
            "attention_mask": [
                [0, 0, 0, 0, 0, ...],
                [0, 0, 0, 0, 0, ...],
            ],
            "label_ids": [
                [0, 0, 0, 2, 6, ...],
                [0, 0, 0, 2, 6, ...],
            ],
        }
    """

    match entry:
        case datasets.arrow_dataset.Example():
            # ------
            # REtokenize the given sentence
            # ------
            # entry["tokens"]: record
            #   ↓
            # k ∈ {"input_ids", "", "attention_mask"}
            # v: vector
            entry.update(
                (
                    k,
                    v.reshape( (-1,) ),
                    # Since the example contains only one record, vectors can be flattened.
                ) for k, v in get_tokenizer()(
                    entry["tokens"],
                    is_split_into_words = True,
                    return_tensors = "np",
                    max_length = MAX_INPUT_LENGTH,
                    padding = "max_length",
                ).items()
            )

            input_ids = entry["input_ids"]

            # translate input_ids (subword IDs) back to Japanese
            tokens_subword = (
                get_tokenizer()
                .convert_ids_to_tokens(input_ids)
            )
            entry["token_subwords"] = tokens_subword

            match return_type:
                case "pt":
                    entry["label_ids"] = _convert_record_to_vector_internal(
                        comps = entry["comp"],
                        input_ids = input_ids,
                        tokens_subword = tokens_subword,
                        return_type = "pt",
                    )
                case "np":
                   entry["label_ids"] = _convert_record_to_vector_internal(
                        comps = entry["comp"],
                        input_ids = input_ids,
                        tokens_subword = tokens_subword,
                        return_type = "np",
                    )
                case "py":
                    entry["label_ids"] = _convert_record_to_vector_internal(
                        comps = entry["comp"],
                        input_ids = input_ids,
                        tokens_subword = tokens_subword,
                        return_type = "py",
                    )
                case _:
                    raise ValueError()
        case datasets.arrow_dataset.Batch():
            batch_size = len(entry["ID"])
            match return_type:
                case "np":
                    label_ids = np.full(
                        (batch_size, MAX_INPUT_LENGTH),
                        LABEL2ID["O"]
                    )
                    rt = None
                case "pt":
                    label_ids = torch.full(
                        (batch_size, MAX_INPUT_LENGTH),
                        LABEL2ID["O"]
                    )
                    rt = return_type
                case "py":
                    label_ids = [ 
                        [LABEL2ID["O"]] * MAX_INPUT_LENGTH 
                        for _ in range(batch_size)
                    ]
                    rt = return_type
                case _:
                    raise ValueError(f"Illegal return type: {return_type}")

            entry.update(
                get_tokenizer()(
                    entry["tokens"],
                    is_split_into_words = True,
                    return_tensors = rt,
                    max_length = MAX_INPUT_LENGTH,
                    padding = "max_length",
                )
            )
            entry["token_subwords"] = []

            for i in range(batch_size):
                # translate input_ids (subword IDs) back to Japanese
                input_ids_i = entry["input_ids"][i]
                tokens_subword_i = (
                    get_tokenizer()
                    .convert_ids_to_tokens(input_ids_i)
                )
                entry["token_subwords"].append(tokens_subword_i)

                match return_type:
                    case "pt":
                        array = _convert_record_to_vector_internal(
                            comps = entry["comp"][i],
                            input_ids = input_ids_i,
                            tokens_subword = tokens_subword_i,
                            return_type = return_type,
                        )
                        label_ids[i] = array
                    case "np":
                        array = _convert_record_to_vector_internal(
                            comps = entry["comp"][i],
                            input_ids = input_ids_i,
                            tokens_subword = tokens_subword_i,
                            return_type = return_type,
                        )
                        label_ids[i] = array
                    case "py":
                        array = _convert_record_to_vector_internal(
                            comps = entry["comp"][i],
                            input_ids = input_ids_i,
                            tokens_subword = tokens_subword_i,
                            return_type = return_type,
                        )
                        label_ids.append(array)
                    case _:
                        raise ValueError(f"Illegal return type: {return_type}")
            
            entry["label_ids"] = label_ids
        case _:
            raise TypeError(f"Illegal input type: {type(entry)}")

    return entry


def _convert_vector_to_record_internal(
    label_ids: Sequence[int],
) -> list[aoc.CompSpan]:
    res: list[aoc.CompSpan] = []

    label_stack: list[tuple[int, str]] = []
    # (start_pos, label)

    # First iteration
    # to find root spans
    for pos_subword, l_id in enumerate(label_ids):
        label = ID2LABEL[l_id]

        if label not in ("O", "IGNORE"):
            if not label_stack:
                # stack is empty
                # a new root span
                label_stack.append(
                    (pos_subword, "root")
                )
            # else:
                # already in a root span
                # do nothing
        else:
            if label_stack:
                # end of all spans
                b, _ = label_stack.pop()
                res.append(
                    {
                        "start": b,
                        "end": pos_subword,
                        "label": "root",
                    }
                )
                label_stack.clear()
            # else:
                  # do nothing

    # postprocessing
    if label_stack:
        b, _ = label_stack.pop()
        res.append(
            {
                "start": b,
                "end": len(label_ids),
                "label": "root",
            }
        )

    # clear the stack
    label_stack.clear()

    # Second iteration
    # to find specific spans
    for pos_subword, l_id in enumerate(label_ids):
        label = ID2LABEL[l_id]
        
        if label in ("O", "IGNORE", "root"):
            if label_stack:
                # end of all spans
                for b, label in label_stack:
                    res.append(
                        {
                            "start": b,
                            "end": pos_subword,
                            "label": label,
                        }
                    )

                # clear the stack
                label_stack.clear()
            # else:
                # do nothing
        else:
            # for non-root labels
            if not label_stack:
                # register the beginning of the span
                label_stack.append(
                    (pos_subword, label)
                )
            else:
                b, prev_label = label_stack[-1]
                if prev_label != label:
                    # if the span changes
                    # pop out the previous span
                    res.append(
                        {
                            "start": b,
                            "end": pos_subword,
                            "label": prev_label,
                        }
                    )
                    label_stack.pop()

                    # register the new span
                    label_stack.append(
                        (pos_subword, label)
                    )
                # else:
                #     # if the span not changes
                #     # leave the things as they are
                #     pass

    # determine other spans
    return res

def convert_predictions_to_annotations(
    entry: E,
    label_ids_key: str = "label_ids",
    comp_key: str = "comp",
) -> E:
    match entry:
        case datasets.arrow_dataset.Example():
            pass
        case datasets.arrow_dataset.Batch():
            result: list[list[aoc.CompSpan]] = []

            batch_size = len(entry[label_ids_key])

            for i in range(batch_size):
                comp = _convert_vector_to_record_internal(
                    entry[label_ids_key][i]
                )
                result.append(comp)

            entry[comp_key] = result
        case _:
            raise TypeError(f"Illegal input type: {type(entry)}")

    return entry

class NERWithRootMetrics(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description = "TBW",
            citation = "",
            inputs_description = "",
            features = datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                },
            ),
            reference_urls = [],
        )

    def _compute(
        self, 
        predictions: Iterable[Sequence[int]], 
        references: Iterable[Sequence[int]]
    ) -> aoc.Metrics:
        predictions_span = tuple(
            _convert_vector_to_record_internal(pred)
            for pred in predictions
        )
        references_span = tuple(
            _convert_vector_to_record_internal(ref)
            for ref in references
        )
        
        return aoc.calc_prediction_metrics(
            predictions_span,
            references_span,
        )