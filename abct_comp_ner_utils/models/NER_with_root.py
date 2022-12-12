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

            # increment the pointer to words
            pos_word += 1
        elif not input_token.startswith("##"):
            # if it is not a subword
            # increment the word pointer
            label_ids[pos_subword] = label_ids_no_alignment[pos_word]
            pos_word += 1
        else:
            # if its a subword
            
            label_ids[pos_subword] = label_ids_no_alignment[pos_word]
            # no incr

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


class CompDict(TypedDict):
    start: int
    end: int
    label: str

def _convert_vector_to_record_internal(
    label_ids: Iterable[int],
) -> list[CompDict]:
    res: list[CompDict] = []

    label_stack: list[tuple[int, str]] = []
    # (start_pos, label)

    for pos_subword, l_id in enumerate(label_ids):
        label = ID2LABEL[l_id]
        
        match label:
            case "root":
                if not label_stack:
                    # stack is empty 
                    # a new root span
                    label_stack.append(
                        (pos_subword, label)
                    )
                # else:
                    # already in a root span
                    # do nothing
            case "O" | "IGNORE":
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
            case _:
                # for non-root labels
                if not label_stack:
                    # a new root span
                    label_stack.append(
                        (pos_subword, "root")
                    )

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
            result: list[list[CompDict]] = []

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

PENALTY_NONE = 0
PENALTY_HALF = 1
PENALTY_FULL = 2
J = TypeVar("J", bound = "NERWithRootJudgment")
@enum.unique
class NERWithRootJudgment(enum.IntEnum):
    CORRECT = enum.auto()

    WRONG_SPAN = enum.auto()
    WRONG_LABEL = enum.auto()
    WRONG_LABEL_SPAN = enum.auto()

    MISSING = enum.auto()
    SPURIOUS = enum.auto()

    # https://stackoverflow.com/a/44644576/15279488
    @classmethod
    @cache
    def penalty(cls: Type[J], val: Union[J, int]):
        if isinstance(val, cls):
            match val:
                case val.CORRECT:
                    return PENALTY_NONE
                case val.WRONG_SPAN:
                    return PENALTY_HALF
                case _:
                    return PENALTY_FULL
        else:
            if val == cls.CORRECT.value:
                return PENALTY_NONE
            elif val == cls.WRONG_SPAN.value:
                return PENALTY_HALF
            else:
                return PENALTY_FULL
            
    @overload
    @classmethod
    def judge(cls, pred: CompDict, ref: CompDict) -> "NERWithRootJudgment":
        ...

    @overload
    @classmethod
    def judge(cls, pred: None, ref: CompDict) -> Literal["NERWithRootJudgment.MISSING"]:
        ...

    @overload
    @classmethod
    def judge(cls, pred: CompDict, ref: None) -> Literal["NERWithRootJudgment.SPURIOUS"]:
        ...

    @overload
    @classmethod
    def judge(cls, pred: None, ref: None) -> None:
        ...

    @classmethod
    def judge(
        cls,
        pred: CompDict | None,
        ref: CompDict | None,
    ) -> Literal[
        "NERWithRootJudgment.SPURIOUS",
        "NERWithRootJudgment.MISSING",
    ] | "NERWithRootJudgment" | None:
        if ref is None:
            if pred is None:
                return None
            else:
                return cls.SPURIOUS
        else:
            if pred is None:
                return cls.MISSING
            else:
                eq_start = ref["start"] == pred["start"]
                eq_end = ref["end"] == pred["end"]
                eq_label = ref["label"] == pred["label"]
                
                match eq_start, eq_end, eq_label:
                    case True, True, True:
                        return cls.CORRECT
                    case True, True, False:
                        return cls.WRONG_LABEL
                    case _, _, True:
                        return cls.WRONG_SPAN
                    case _, _, _:
                        return cls.WRONG_LABEL_SPAN

    @classmethod
    def align(
        cls,
        predictions: Sequence[CompDict],
        references: Sequence[CompDict],
    ) -> tuple[
        tuple[tuple[int, "NERWithRootJudgment"], ...],
        tuple[tuple[int, "NERWithRootJudgment"], ...],
    ]:
        size_pred = len(predictions)
        size_ref = len(references)

        if size_pred == 0 and size_ref == 0:
            return tuple(), tuple()
        # ------------
        # Make judgment tables
        # ------------
        judgments = np.array(
            [
                [
                    NERWithRootJudgment
                    .judge(
                        pred = predictions[p],
                        ref = references[r],
                    )
                    for r in range(size_ref)
                ]
                for p in range(size_pred)
            ]
        ).reshape(
            (size_pred, size_ref)
        )

        costs = np.vectorize(
            NERWithRootJudgment.penalty, 
            otypes = [np.int_],
        )(judgments)

        padding_pred_idle = np.full(
            (size_pred, size_pred),
            PENALTY_FULL,
            dtype = np.int_
        )
        padding_idle_ref = np.full(
            (size_ref, size_ref),
            PENALTY_FULL,
            dtype = np.int_
        )
        padding_idle_idle = np.full(
            (size_ref, size_pred),
            PENALTY_NONE,
            dtype = np.int_
        )
        costs_ext = np.block(
            [
                [costs, padding_pred_idle],
                [padding_idle_ref, padding_idle_idle],
            ]
        )

        # -------------
        # Compute strict scores
        # -------------
        opt_pred: NDArray[np.int_]
        opt_ref: NDArray[np.int_]
        opt_pred, opt_ref = linear_sum_assignment(costs_ext)

        map_pred_ref = tuple(
            (
                (r, NERWithRootJudgment(judgments[p][r]) )
                if r < size_ref
                else (-1, NERWithRootJudgment.SPURIOUS)
            )
            for p, r in zip(opt_pred, opt_ref)
            if p < size_pred
        )

        map_ref_pred = tuple(
            ( 
                (p, NERWithRootJudgment(judgments[p][r]))
                if p < size_pred
                else (-1, NERWithRootJudgment.MISSING)
            )
            for p, r in zip(opt_pred, opt_ref)
            if r < size_ref
        )

        return map_pred_ref, map_ref_pred

class NERWithRootMetricsResult(TypedDict):
    scores_spanwise: dict[NERWithRootLabel, dict[str, float]]
    F1_strict_average: float
    F1_partial_average: float

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
    ) -> NERWithRootMetricsResult:
        result_bin: dict[
            NERWithRootLabel,
            Counter[NERWithRootJudgment]
        ] = {
                label: Counter()
                for label in NERWithRootLabel
            }

        for pred, ref in zip(predictions, references):
            pred_complist = _convert_vector_to_record_internal(pred)
            ref_complist = _convert_vector_to_record_internal(ref)

            map_p2r, map_r2p = NERWithRootJudgment.align(
                pred_complist,
                ref_complist,
            )

            for p, pred_comp in enumerate(pred_complist):
                _, pred_ref_jud = map_p2r[p]

                label_enum = NERWithRootLabel[
                    pred_comp["label"]
                ]
                result_bin[label_enum][pred_ref_jud] += 1
            
            for r, ref_comp in enumerate(ref_complist):
                pred_index, ref_pred_jud = map_r2p[r]
                if pred_index < 0:
                    label_enum = NERWithRootLabel[
                        ref_comp["label"]
                    ]
                    result_bin[label_enum][ref_pred_jud] += 1

        # ------
        # Calc spanwise scores
        # ------
        res_per_label: dict[
            NERWithRootLabel,
            dict[str, float],
        ] = {}
        for label in (
            l for l in NERWithRootLabel
            if l not in (
                NERWithRootLabel.O,
                NERWithRootLabel.IGNORE,
            )
        ):
            ct = result_bin[label]
            possible_entries = (
                ct[NERWithRootJudgment.CORRECT] 
                + ct[NERWithRootJudgment.WRONG_SPAN]
                + ct[NERWithRootJudgment.WRONG_LABEL]
                + ct[NERWithRootJudgment.WRONG_LABEL_SPAN]
                + ct[NERWithRootJudgment.MISSING]
            )

            actual_entries = (
                possible_entries
                - ct[NERWithRootJudgment.MISSING]
                + ct[NERWithRootJudgment.SPURIOUS]
            )

            precision_strict = ct[NERWithRootJudgment.CORRECT] / actual_entries
            recall_strict = ct[NERWithRootJudgment.CORRECT] / possible_entries
            F1_strict = (
                2 * precision_strict * recall_strict
                / (precision_strict + recall_strict)
            )

            correct_with_partial = (
                ct[NERWithRootJudgment.CORRECT]
                + 0.5 * ct[NERWithRootJudgment.WRONG_SPAN]
            )
            precision_partial = correct_with_partial / actual_entries
            recall_partial = correct_with_partial / possible_entries
            F1_partial = (
                2 * precision_partial * recall_partial
                / (precision_partial + recall_partial)
            )

            res_per_label[label] = {
                key.name: value
                for key, value in ct.items()
            }
            res_per_label[label]["possible_entries"] = possible_entries
            res_per_label[label]["actual_entries"] = actual_entries
            
            res_per_label[label]["precision_strict"] = precision_strict
            res_per_label[label]["recall_strict"] = recall_strict
            res_per_label[label]["F1_strict"] = F1_strict

            res_per_label[label]["precision_partial"] = precision_partial
            res_per_label[label]["recall_partial"] = recall_partial
            res_per_label[label]["F1_partial"] = F1_partial
        
        F1_strict_list = tuple(
            res["F1_strict"]
            for res in res_per_label.values()
        )
        F1_partial_list = tuple(
            res["F1_partial"]
            for res in res_per_label.values()
        )
        
        return {
            "scores_spanwise": res_per_label,
            "F1_strict_average": sum(F1_strict_list) / len(F1_strict_list),
            "F1_partial_average": sum(F1_partial_list) / len(F1_partial_list),
        }