from collections import Counter, defaultdict
import dataclasses
from dataclasses import dataclass
import itertools

from sklearn.metrics import classification_report

import datasets

import evaluate

@dataclass(slots = True)
class MetricState:
    pos_begin: int = 0
    label: str = ""
    id2label_detailed: dict = dataclasses.field(default_factory = dict)

    def __bool__(self):
        return bool(self.label)

    def has_same_span(self, other: "MetricState") -> bool:
        return self.pos_begin == other.pos_begin

    def has_same_label(self, other: "MetricState") -> bool:
        return self.label == other.label

    def tell_change(self, next: "MetricState", self_ignored = False):
        label = "O" if self_ignored else self.label
        match label, next.label:
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
        label = self.id2label_detailed[label_id][0]
        if label == self.label:
            # continue the state
            return self
        else:
            # make a new state
            return self.__class__(pos, label, self.id2label_detailed)

# https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
def _calc_result(ct: Counter[str]) -> dict[str, int | float]:
    res: dict[str, int | float] = {
        "possible_entries": ct["CORRECT"] + ct["WRONG_SPAN"] + ct["WRONG_LABEL"] + ct["WRONG_LABEL_SPAN"] + ct["MISSING"]
    }
    res["actual_entries"] = res["possible_entries"] - ct["MISSING"] + ct["SPURIOUS"]

    res["precision_strict"] = ct["CORRECT"] / res["actual_entries"] 
    res["recall_strict"] = ct["CORRECT"] / res["possible_entries"]
    res["F1_strict"] = 2 * res["precision_strict"] * res["recall_strict"] / (res["precision_strict"] + res["recall_strict"])

    correct_with_partial = ct["CORRECT"] + 0.5 * ct["WRONG_SPAN"]
    res["precision_partial"] = correct_with_partial / res["actual_entries"] 
    res["recall_partial"] = correct_with_partial / res["possible_entries"]
    res["F1_partial"] = 2 * res["precision_partial"] * res["recall_partial"] / (res["precision_partial"] + res["recall_partial"])

    return res

class ComparativeNERAccuracy(evaluate.module.Metric):
    def _info(self):
        return evaluate.EvaluationModuleInfo(
            description = "TBW",
            citation = "",
            features = datasets.Features(
            )
        )


    def _compute(
        self,
        predictions,
        references,
        *,
        input_ids,
        label2id,
        id2label_detailed,
        special_ids = [],
        normalize = True,
        sample_weight = True,
        **kwargs,
    ):
        result_bin: defaultdict[str, Counter[str]] = defaultdict()
        result_bin.default_factory = Counter

        result_tokenwise_given = []
        result_tokenwise_pred = []

        predictions = predictions.argmax(axis = 2)

        for sent, label_ids, pred_ids in zip(
            input_ids,
            references,
            predictions,
        ):
            prev_given_state = MetricState(id2label_detailed = id2label_detailed)
            current_given_state = MetricState(id2label_detailed = id2label_detailed)
            prev_pred_state = MetricState(id2label_detailed = id2label_detailed)
            current_pred_state = MetricState(id2label_detailed = id2label_detailed)

            given_consumed = False
            pred_consumed = False

            for token, label_id, pred_id in zip(sent, label_ids, pred_ids):
                if token in special_ids:
                    continue
                result_tokenwise_given.append(label_id)
                result_tokenwise_pred.append(pred_id)

            for i, (label_id, pred_id) in enumerate(
                itertools.chain(
                    zip(label_ids, pred_ids),
                    ((0, 0), ),
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

                # update masking
                given_consumed &= not prev_given_state.has_same_label(current_given_state)
                pred_consumed &= not prev_pred_state.has_same_label(current_pred_state)

                match (
                    prev_given_state.tell_change(current_given_state, given_consumed),
                    prev_pred_state.tell_change(current_pred_state, pred_consumed),
                ):
                    case "KEEP_EMPTY" | "ENTER_STATE", "CHANGE_STATE" | "END_STATE":
                        #                 |------------------------
                        # given O         | label or O
                        #                 |------------------------
                        #                   @
                        # ================|------------------------
                        # predicted label | label' or O
                        # ================|------------------------
                        result_bin[prev_pred_state.label]["SPURIOUS"] += 1
                    case "KEEP_STATE", "CHANGE_STATE" | "END_STATE":
                        # =========================================
                        # given label [TO CONSUME!]
                        # =========================================
                        #                   @
                        # ================|----------------
                        # predicted label | label' or O
                        # ================|----------------
                        if prev_given_state.label == prev_pred_state.label:
                            result_bin[prev_pred_state.label]["WRONG_SPAN"] += 1
                        else:
                            result_bin[prev_pred_state.label]["WRONG_SPAN_LABEL"] += 1
                        given_consumed = True
                    case "CHANGE_STATE" | "END_STATE", "KEEP_EMPTY" | "ENTER_STATE":
                        # ================|-------------
                        # given label     | label' or O
                        # ================|-------------
                        #                   @
                        #                 |-------------
                        # predicted O     | label' or O
                        #                 |-------------
                        result_bin[prev_given_state.label]["MISSING"] += 1
                    case "CHANGE_STATE" | "END_STATE", "KEEP_STATE":
                        # ================|------------------------
                        # given label     | label' or O
                        # ================|------------------------
                        #                   @
                        # =========================================
                        # predicted label [TO CONSUME!]
                        # =========================================
                        if prev_given_state.label == prev_pred_state.label:
                            result_bin[prev_pred_state.label]["WRONG_SPAN"] += 1
                        else:
                            result_bin[prev_pred_state.label]["WRONG_LABEL_SPAN"] += 1
                        pred_consumed = True
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
                                result_bin[prev_given_state.label]["WRONG_SPAN"] += 1
                            case False, True:
                                result_bin[prev_given_state.label]["WRONG_LABEL"] += 1
                            case _:
                                result_bin[prev_given_state.label]["WRONG_LABEL_SPAN"] += 1
                    case _:
                        pass
        return {
            "score_spanwise_details": { key: dict(val.items()) for key, val in result_bin.items() },
            "score_spanwise": {
                key: _calc_result(val) for key, val
                in result_bin.items()
            },
            "score_tokenwise": classification_report(
                result_tokenwise_given, result_tokenwise_pred,
                labels = list(label2id.values()),
                target_names = list(label2id.keys()),
                output_dict = True,
            )
        }