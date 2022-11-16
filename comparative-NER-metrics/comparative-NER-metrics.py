from collections import Counter, defaultdict
import dataclasses
from dataclasses import dataclass
from typing import Sequence

from sklearn.metrics import classification_report

import datasets
import datasets.features as df

import evaluate
import numpy as np
import torch

@dataclass(slots = True)
class MetricState:
    """
    A container class representing internal states of metric calculation.
    """

    pos_begin: int = 0
    """

    """

    label: str = ""
    """

    """

    consumed: bool = False

    id2label_detailed: dict[int, tuple[str, str]] = dataclasses.field(default_factory = dict)
    """

    """

    def __repr__(self):
        return (
            f"<MetricState pos_begin: {self.pos_begin}, "
            f"label: {self.label}, "
            f"consumed: {self.consumed}, "
            f"id2label_detailed: {id(self.id2label_detailed)}>"
        )

    def __str__(self):
        return (
            f"<MetricState pos_begin: {self.pos_begin}, "
            f"label: {self.label}, "
            f"consumed: {self.consumed}>"
        )

    def __bool__(self):
        return bool(self.label)

    def has_same_span(self, other: "MetricState") -> bool:
        return self.pos_begin == other.pos_begin

    def has_same_label(self, other: "MetricState") -> bool:
        return self.label == other.label
    
    def get_label_unconsumed(self):
        if self.consumed:
            return "O"
        else:
            return self.label

    def tell_change(self, next: "MetricState"):
        label = self.get_label_unconsumed()
        next_label = next.get_label_unconsumed()

        match label, next_label:
            case "" | "O" | "IGNORE", "" | "O" | "IGNORE":
                return "KEEP_EMPTY"
            case "" | "O" | "IGNORE", _:
                return "ENTER_STATE"
            case _, "" | "O" | "IGNORE":
                return "END_STATE"
            case p, n if p == n:
                return "KEEP_STATE"
            case _:
                return "CHANGE_STATE"

    def update(self, pos: int, label_id: int) -> "MetricState":
        """ 
        Update the state.

        Parameters
        ----------

        """

        label, *_ = self.id2label_detailed[label_id]
        if label == self.label:
            # continue the state
            return self
        else:
            # make a new state
            return self.__class__(
                pos_begin = pos,
                label = label,
                consumed = False,
                id2label_detailed = self.id2label_detailed
            )

def _safe_divide(devidend: int | float, divisor: int | float) -> int | float:
    if divisor == 0:
        return 0
    else:
        return devidend / divisor


# https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
def _calc_spanwise_per_feat_result(ct: Counter[str]) -> dict[str, int | float]:
    res: dict[str, int | float] = {
        "possible_entries": ct["CORRECT"] + ct["WRONG_SPAN"] + ct["WRONG_LABEL"] + ct["WRONG_LABEL_SPAN"] + ct["MISSING"]
    }
    res["actual_entries"] = res["possible_entries"] - ct["MISSING"] + ct["SPURIOUS"]

    res["precision_strict"] = _safe_divide(ct["CORRECT"], res["actual_entries"])
    res["recall_strict"] = _safe_divide(ct["CORRECT"], res["possible_entries"])
    res["F1_strict"] = 2 * _safe_divide(
        res["precision_strict"] * res["recall_strict"],
        (res["precision_strict"] + res["recall_strict"]),
    )

    correct_with_partial = ct["CORRECT"] + 0.5 * ct["WRONG_SPAN"]
    res["precision_partial"] = _safe_divide(correct_with_partial, res["actual_entries"])
    res["recall_partial"] = _safe_divide(correct_with_partial, res["possible_entries"])
    res["F1_partial"] = 2 * _safe_divide(
        res["precision_partial"] * res["recall_partial"],
        (res["precision_partial"] + res["recall_partial"]),
    )

    return res

_comp_feat_labels = ["deg", "diff", "cont", "prej", "root"]
_record_structure = {
    "id": datasets.Value("string"),
    "tokens": datasets.features.Sequence(
        datasets.Value("string")
    ),
    "comp": datasets.features.Sequence(
        {
            "start": datasets.Value("int64"),
            "end": datasets.Value("int64"),
            "label": datasets.ClassLabel(
                len(_comp_feat_labels),
                _comp_feat_labels
            ),
        }
    ),
}
class ComparativeNERAccuracy(evaluate.module.Metric):
    """
    The evaluator of comparative-NER accuracy.
    """

    def _info(self):
        return evaluate.MetricInfo(
            description = "TBW",
            citation = "",
            features = datasets.Features(
                {
                    "ID": df.Value("string"),
                    "input_ids": df.Sequence(feature = df.Value("int32")),
                    "references": df.Sequence(feature = df.Value("int64")),
                    "predictions": df.Sequence(feature = df.Value("int64")),
                }
            ),
            reference_urls = [],
        )

    def _compute_raw(
        self,
        predictions: Sequence | np.ndarray |  torch.Tensor,
        references: Sequence | np.ndarray |  torch.Tensor,
        *,
        input_ids,
        label2id: dict[str, int],
        id2label_detailed: dict[int, tuple[str, str]],
        special_ids: Sequence[int] = [],
        normalize = True,
        sample_weight = True,
        **kwargs,
    ):
        match predictions:
            case torch.Tensor():
                predictions = predictions.argmax(dim = 2)
            case np.ndarray():
                predictions = predictions.argmax(axis = 2)
            case _:
                raise TypeError
        
        return self._compute(
            predictions,
            references, 
            input_ids = input_ids,
            label2id = label2id,
            id2label_detailed = id2label_detailed,
            special_ids = special_ids,
            normalize = normalize,
            sample_weight = sample_weight,
            **kwargs
        )

    def _compute(
        self,
        predictions: torch.Tensor | np.ndarray | Sequence[Sequence[int]],
        references: torch.Tensor | np.ndarray | Sequence[Sequence[int]],
        ID: torch.Tensor | np.ndarray | Sequence[str],
        input_ids,
        label2id: dict[str, int],
        id2label_detailed: dict[int, tuple[str, str]],
        special_ids: Sequence[int] = [],
    ):
        """
        Compute the accuracy for a sentence.

        Arguments
        ---------
        predictions
            Predicted NER labels.
        references
            Gold labels.
        ID
            Data IDs.
        input_ids
            Sentence words.
        label2id
            A dictionary from NER labels to IDs.
        id2label_detailed
            A dictionary from IDs to labels.
        """
        
        result_bin: defaultdict[str, Counter[str]] = defaultdict()
        result_bin.default_factory = Counter
        error_list: list[list[tuple[str, str]]] = [ [] for _ in range(len(ID)) ]
        result_tokenwise_given = []
        result_tokenwise_pred = []

        for sent_ID, sent, label_ids, pred_ids, errors in zip(
            ID,
            input_ids,
            references,
            predictions,
            error_list
        ):
            prev_given_state = MetricState(id2label_detailed = id2label_detailed)
            current_given_state = MetricState(id2label_detailed = id2label_detailed)
            prev_pred_state = MetricState(id2label_detailed = id2label_detailed)
            current_pred_state = MetricState(id2label_detailed = id2label_detailed)

            
            for token, label_id, pred_id in zip(sent, label_ids, pred_ids):
                if token in special_ids:
                    continue
                result_tokenwise_given.append(label_id)
                result_tokenwise_pred.append(pred_id)

            for i, (label_id, pred_id) in enumerate(
                zip(
                    label_ids,
                    pred_ids.tolist() 
                    if isinstance(pred_ids, torch.Tensor)
                    else pred_ids
                )
            ):
                # update states
                prev_given_state, current_given_state = (
                    current_given_state,
                    current_given_state.update(i, label_id)
                )

                prev_pred_state, current_pred_state = (
                    current_pred_state,
                    current_pred_state.update(i, pred_id)
                )

                # For debugging
                # if sent_ID == "53_BCCWJ-ABC-ak-simple":
#                     print(
#                         f"""
# cursor: {i}
# given: {label_id}
# given_state: {prev_given_state}
# → {current_given_state}
# transition: {prev_given_state.tell_change(current_given_state)}

# predicted: {pred_id}
# predicted_state: {prev_pred_state}
# → {current_pred_state}
# transition: {prev_pred_state.tell_change(current_pred_state)}
#                         """
#                     )
                
                match (
                    prev_given_state.tell_change(current_given_state),
                    prev_pred_state.tell_change(current_pred_state),
                ):
                    case "KEEP_EMPTY" | "ENTER_STATE", "CHANGE_STATE" | "END_STATE":
                        #                   |------------------------
                        # given O           | label or O
                        #                   |------------------------
                        #                     @
                        # ==================|------------------------
                        # predicted label   | label' or O
                        # (if not consumed) |
                        # ==================|------------------------
                        _mark_error(
                            prev_pred_state.label,
                            "SPURIOUS",
                            result_bin,
                            errors,
                            ID = sent_ID,
                        )
                    case "KEEP_STATE", "CHANGE_STATE" | "END_STATE":
                        # =========================================
                        # given label [TO CONSUME!]
                        # =========================================
                        #                   @
                        # ================|----------------
                        # predicted label | label' or O
                        # ================|----------------
                        if prev_given_state.label == prev_pred_state.label:
                            _mark_error(
                                prev_pred_state.label,
                                "WRONG_SPAN",
                                result_bin,
                                errors,
                                ID = sent_ID,
                            )
                        else:
                            _mark_error(
                                prev_pred_state.label,
                                "WRONG_LABEL_SPAN",
                                result_bin,
                                errors,
                                ID = sent_ID,
                            )
                        current_given_state.consumed = True
                    case "CHANGE_STATE" | "END_STATE", "KEEP_EMPTY" | "ENTER_STATE":
                        # ================|-------------
                        # given label     | label' or O
                        # ================|-------------
                        #                   @
                        #                 |-------------
                        # predicted O     | label' or O
                        #                 |-------------
                        _mark_error(
                            prev_given_state.label,
                            "MISSING",
                            result_bin,
                            errors,
                            ID = sent_ID,
                        )
                    case "CHANGE_STATE" | "END_STATE", "KEEP_STATE":
                        # ================|------------------------
                        # given label     | label' or O
                        # ================|------------------------
                        #                   @
                        # =========================================
                        # predicted label [TO CONSUME!]
                        # =========================================
                        if prev_given_state.label == prev_pred_state.label:
                            _mark_error(
                                prev_pred_state.label,
                                "WRONG_SPAN",
                                result_bin,
                                errors,
                                ID = sent_ID,
                            )
                        else:
                            _mark_error(
                                prev_pred_state.label,
                                "WRONG_LABEL_SPAN",
                                result_bin,
                                errors,
                                ID = sent_ID,
                            )
                        current_pred_state.consumed = True
                    case "CHANGE_STATE" | "END_STATE", "CHANGE_STATE" | "END_STATE":
                        # ================|------------------------
                        # given label     | label' or O
                        # ================|------------------------
                        #                   @
                        # ================|------------------------
                        # predicted label | label' or O
                        # ================|------------------------
                        match prev_given_state.label == prev_pred_state.label, prev_given_state.pos_begin == prev_pred_state.pos_begin:
                            case True, True:
                                result_bin[prev_given_state.label]["CORRECT"] += 1
                            case True, False:
                                _mark_error(
                                    prev_given_state.label,
                                    "WRONG_SPAN",
                                    result_bin,
                                    errors,
                                    ID = sent_ID,
                                )
                            case False, True:
                                _mark_error(
                                    prev_given_state.label,
                                    "WRONG_LABEL",
                                    result_bin,
                                    errors,
                                    ID = sent_ID,
                                )
                            case _:
                                _mark_error(
                                    prev_given_state.label,
                                    "WRONG_LABEL_SPAN",
                                    result_bin,
                                    errors,
                                    ID = sent_ID,
                                )
                    case _:
                        pass

        score_per_feat = {
            key: _calc_spanwise_per_feat_result(val)
            for key, val
            in result_bin.items()
        }

        return {
            "score_spanwise_details": { key: dict(val.items()) for key, val in result_bin.items() },
            "score_spanwise": score_per_feat,
            "score_spanwise_F1_strict": (
                sum(d["F1_strict"] for d in score_per_feat.values()) 
                / len(score_per_feat)
            ),
            "score_tokenwise": classification_report(
                result_tokenwise_given, result_tokenwise_pred,
                labels = list(label2id.values()),
                target_names = list(label2id.keys()),
                output_dict = True,
                zero_division = 0,
            ),
            "error_list" : error_list,
        }

def _mark_error(label: str, error: str, result_bin: dict, error_bin: list, ID: str = "<UNKNOWN>"):
    result_bin[label][error] += 1
    error_bin.append( (label, error) )