import json
import operator
from pathlib import Path
import re
import sys
from typing import Iterable, Optional, Sequence, Any, TypeVar

import typer
import flatten_dict
import ruamel.yaml

import numpy as np
import torch
from torch import Tensor
import torch.utils.data

import datasets
from transformers import BertConfig, BertForTokenClassification, BertJapaneseTokenizer, EvalPrediction, IntervalStrategy, TokenClassificationPipeline, Trainer, TrainingArguments
import evaluate

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch

import abct_comp_ner_utils.brackets as br

LABEL2ID_DETAILED: dict[tuple[str, str], int] = {
    ("IGNORE", ""): -100,
    ("O", ""): 0,
    ("deg", "B"): 1,
    ("prej", "B"): 2,
    ("cont", "B"): 3,
    ("diff", "B"): 4,
    ("deg", "I"): 5,
    ("prej", "I"): 6,
    ("cont", "I"): 7,
    ("diff", "I"): 8,
}
"""
A dictionary that converts comparative NER labels to the corresponding integer IDs.
The labels are pairs of a feature name (deg, prej, ...) and the beginning/intermediate indicator (B/I).
"""

LABEL2ID = {
    (f"{prefix}-{name}" if prefix else name):value
    for (name, prefix), value in LABEL2ID_DETAILED.items()
}

LABEL2ID_WO_IGNORE = {
    k:v
    for k, v in LABEL2ID.items()
    if k != "IGNORE"
}

ID2LABEL_DETAILED = {
    i: label 
    for label, i in LABEL2ID_DETAILED.items()
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

_BERT_MODEL = "cl-tohoku/bert-base-japanese-whole-word-masking"

_TOKENIZER = None
def _get_tokenizer() -> BertJapaneseTokenizer:
    global _TOKENIZER
    _TOKENIZER = _TOKENIZER or BertJapaneseTokenizer.from_pretrained(_BERT_MODEL)

    return _TOKENIZER

_ID_CLS: int = _get_tokenizer().cls_token_id or 2
_ID_SEP: int = _get_tokenizer().sep_token_id or 3

_EVALUATOR = dict()

def _get_evaluator(name: str):
    global _EVALUATOR
    _EVALUATOR[name] = _EVALUATOR.get(name, None) or evaluate.load(name)
    return _EVALUATOR[name]

_RE_FEAT_ARTIFACTS = re.compile(r"^(?P<name>[a-zA-Z]+)[0-9]?")
"""
A regex that detects extra letters in feature names.
"""

def _convert_record_to_vector_internal(
    comps: list[dict[str, Any]] | dict[str, list[Any]] | None,
    input_ids: Iterable[int],
    tokens_subwords: Iterable[str],
    output_label_vector: Tensor,
):
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
    # Make the ConLL label ID tensor
    # while aligning to subwords
    # ------
    # Define states
    pos_word: int = -1  # the pointer to original words
    current_feat: str = "O" # the current feature
    current_feat_is_start: bool = True  # whether it is the beginning of the span
    current_feat_end_pos_word: int = -1

    # Enumerate words and subwords from the retokenization in a parallel way
    #       pos_subword: position
    #       input_id: subword ID
    #       input_token: subword translated back
    for pos_subword, (input_id, input_token) in enumerate(
        zip(input_ids, tokens_subwords)
    ):
        # ------
        # Increment the word pointer
        # ------
        if input_id == _ID_SEP:
            # if it reaches the end of the sentence [SEP]
            # end thj procedure
            break
        elif input_id == _ID_CLS:
            # if it hits on the beginning of the sentence [CLS]

            # increment the pointer to words
            pos_word += 1
            continue
        elif not input_token.startswith("##"):
            # if it is not a subword
            # increment the word pointer
            pos_word += 1
        # else:
            # if its a subword
            # no incr
            # pass

        # -------
        # Sync label IDs based on subwords with that based on words
        # -------
        # if the span reach the end
        # (based on word position)
        if current_feat_end_pos_word == pos_word:
            # reset the current feature to O
            current_feat = "O"

        # inquire the current span
        # (based on word position)
        for start, end, label in feat_list:
            # count in the [CLS] offset
            start += 1
            end += 1

            if (
                label != "root" 
                and pos_word == start
                and (match := _RE_FEAT_ARTIFACTS.match(label))
            ):
                previous_feat = current_feat
                current_feat = match.group("name")
                current_feat_end_pos_word = end

                if previous_feat != current_feat:
                    current_feat_is_start = True

        # write feature in
        # (based on subword position)
        if current_feat != "O":
            if current_feat_is_start:
                output_label_vector[pos_subword] = LABEL2ID_DETAILED[current_feat, "B"]
                current_feat_is_start = False
            else:
                output_label_vector[pos_subword] = LABEL2ID_DETAILED[current_feat, "I"]
        # === END FOR feat_list ===
    # === END FOR ===

E = TypeVar("E", datasets.arrow_dataset.Example, datasets.arrow_dataset.Batch)
def convert_records_to_vectors(entry: E) -> E:
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
                (k, v.reshape( (-1, ) ))
                # Since the example contains only one record, vectors can be flattened.
                for k, v in _get_tokenizer()(
                    entry["tokens"],
                    is_split_into_words = True,
                    return_tensors = "np",
                    max_length = MAX_INPUT_LENGTH,
                    padding = "max_length",
                ).items()
            )
            # translate input_ids (subword IDs) back to Japanese
            tokens_subword = (
                _get_tokenizer()
                .convert_ids_to_tokens(entry["input_ids"])
            )
            # tokens_subword += [""] * (MAX_INPUT_LENGTH - len(tokens_subword))
            entry["token_subwords"] = tokens_subword

            label_ids = np.full((MAX_INPUT_LENGTH, ), LABEL2ID["O"])
            _convert_record_to_vector_internal(
                comps = entry["comp"],
                input_ids = entry["input_ids"],
                tokens_subwords = tokens_subword,
                output_label_vector = label_ids, # out
            )
            entry["label_ids"] = label_ids
        
        case datasets.arrow_dataset.Batch():
            entry.update(
                _get_tokenizer()(
                    entry["tokens"],
                    is_split_into_words = True,
                    return_tensors = "np",
                    max_length = MAX_INPUT_LENGTH,
                    padding = "max_length",
                )
            )
            entry["token_subwords"] = []

            batch_size = len(entry["ID"])
            label_ids: Tensor = np.full(
                (batch_size, MAX_INPUT_LENGTH),
                LABEL2ID["O"]
            )
            for i in range(batch_size):
                # translate input_ids (subword IDs) back to Japanese
                tokens_subword = (
                    _get_tokenizer()
                    .convert_ids_to_tokens(entry["input_ids"][i])
                )
                # tokens_subword += [""] * (MAX_INPUT_LENGTH - len(tokens_subword))
                entry["token_subwords"].append(tokens_subword)
                
                _convert_record_to_vector_internal(
                    comps = entry["comp"][i],
                    input_ids = entry["input_ids"][i],
                    tokens_subwords = tokens_subword,
                    output_label_vector = label_ids[i]
                )
            
            entry["label_ids"] = label_ids
        case _:
            raise TypeError
    return entry

def convert_vector_to_span(
    input: Sequence | np.ndarray | Tensor,
    labels: Sequence | np.ndarray | Tensor,
):
    """
    Convert a vector of comparative feature IDs to a span annotation.

    Arguments
    ---------
    input
        Input tokens (subwords)
    labels
        Label IDs
    """

    # result: list[dict[str, int | str]] = []
    result: dict[str, list] = {
        "start": [],
        "end": [],
        "label": [],
    }

    current_label = ID2LABEL_DETAILED[0][0]
    current_span_start: int = 0

    for loc, (input_id, label_id) in enumerate(zip(input, labels)):
        label = ID2LABEL_DETAILED[label_id][0]

        if input_id == 0:
            # reached padding
            break
        elif current_label != label:
            # label changed
            # conclude the old label
            if current_label not in ("IGNORE", "O"):
                # result.append(
                #     {
                #         "start": current_span_start,
                #         "end": loc,
                #         "label": current_label,
                #     }
                # )                
                result["start"].append(current_span_start),
                result["end"].append(loc),
                result["label"].append(current_label)
            # else:
            #     pass

            # switch to new label
            current_label = label
            current_span_start = loc

    return result

def _compute_metric(e: evaluate.Metric):
    def _run(pds: EvalPrediction):
        # ensure cache
        e.add_batch(
            predictions = pds.predictions,
            references = pds.label_ids,
        )
        
        res = e._compute(
            predictions = pds.predictions,
            references = pds.label_ids,
            input_ids = pds.inputs,
            special_ids = _get_tokenizer().all_special_ids,
            label2id = LABEL2ID,
            id2label_detailed = ID2LABEL_DETAILED,
        )

        return flatten_dict.flatten(res, reducer='path')

    return _run

def train(
    name: Optional[str] = typer.Option(
        None, "--name", "-n",
        help = "Name of this training execution."
    ),
    data_path: Optional[Path] = typer.Option(
        None, "--data",
        file_okay = False,
    ),
    evaluator_path: Optional[Path] = typer.Option(
        None, "--evaluator",
        file_okay = False,
    ),
    output_path: Path = typer.Option(
        Path("./results"), "--output",
        file_okay = False,
    ),
    upload: bool = False,
):
    # init
    _get_tokenizer()
    _eval: evaluate.Metric = _get_evaluator(
        str(evaluator_path)
        if evaluator_path
        else "abctreebank/comparative-NER-metrics"
    )

    dataset_raw = datasets.load_dataset(
        (
            str(data_path)
            if data_path
            else "abctreebank/comparative-NER-BCCWJ"
        ),
        use_auth_token = True,
        split = "train",
    )
    if not isinstance(dataset_raw, datasets.Dataset):
        raise TypeError

    dataset_raw = dataset_raw.map(
        convert_records_to_vectors,
        remove_columns = dataset_raw.column_names,
    )

    # train/eval split
    dataset = dataset_raw.train_test_split(test_size = 0.1, shuffle = True)

    config = BertConfig.from_pretrained(
        _BERT_MODEL,
        id2label = ID2LABEL,
        label2id = LABEL2ID_WO_IGNORE,
    )

    model = BertForTokenClassification.from_pretrained(
        _BERT_MODEL,
        config = config
    )

    training_args = TrainingArguments(
        output_dir = str(output_path),
        num_train_epochs = 27,
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 128,
        learning_rate = 5e-5,
        warmup_steps = 200,
        weight_decay = 0,
        save_strategy = IntervalStrategy.STEPS,
        save_steps = 1000,
        do_eval = True,
        evaluation_strategy = IntervalStrategy.STEPS,
        eval_steps = 109,
        include_inputs_for_metrics = True,
        seed = 2630987289,
        logging_dir = str(output_path / "logs"),
        logging_steps= 10,
    )

    trainer = Trainer(
        model_init = lambda: model,
        args = training_args,
        compute_metrics = _compute_metric(_eval),
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()

def find_hyperparameters(
    name: Optional[str] = typer.Option(
        None, "--name",
    ),
    data_path: Optional[Path] = typer.Option(
        None, "--data",
        file_okay = False,
    ),
    evaluator_path: Optional[Path] = typer.Option(
        None, "--evaluator",
        file_okay = False,
    ),
    output_path: Path = typer.Option(
        Path("./experiments"), "--output", "-o",
        file_okay = False,
    ),
    n_trials: int = typer.Option(
        30, "--trials", "-n",
    ),
):
    # init
    _get_tokenizer()
    _eval: evaluate.Metric = _get_evaluator(
        str(evaluator_path)
        if evaluator_path
        else "abctreebank/comparative-NER-metrics"
    )

    dataset_raw = datasets.load_dataset(
        (
            str(data_path)
            if data_path
            else "abctreebank/comparative-NER-BCCWJ"
        ),
        use_auth_token = True,
        split = "train",
    )
    assert isinstance(dataset_raw, datasets.Dataset)

    dataset_raw = dataset_raw.map(
        convert_records_to_vectors,
        remove_columns = dataset_raw.column_names,
    )

    # train/eval split
    dataset = dataset_raw.train_test_split(test_size = 0.1, shuffle = True)

    config = BertConfig.from_pretrained(
        _BERT_MODEL,
        id2label = ID2LABEL,
        label2id = LABEL2ID_WO_IGNORE,
    )

    model = BertForTokenClassification.from_pretrained(
        _BERT_MODEL,
        config = config
    )

    training_args = TrainingArguments(
        output_dir = str(output_path),
        num_train_epochs = 100,
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 128,
        learning_rate = 5e-5,
        warmup_steps = 200,
        weight_decay = 0,
        save_strategy = IntervalStrategy.STEPS,
        save_steps = 1000,
        do_eval = True,
        evaluation_strategy = IntervalStrategy.STEPS,
        eval_steps = 1000,
        include_inputs_for_metrics = True,
        logging_dir = str(output_path / "logs"),
        logging_steps= 10,
    )

    trainer = Trainer(
        model_init = lambda: model,
        args = training_args,
        compute_metrics = _compute_metric(_eval),
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
    )

    tune_config = {
        "num_train_epochs": tune.qrandint(10, 100),
        "seed": tune.qrandint(0, 2 ** 32 - 1),
        "learning_rate": tune.loguniform(5e-6, 5e-3),
    }
    _F1_STRICT = "eval_score_spanwise_F1_strict"

    best_result = trainer.hyperparameter_search(
        hp_space = lambda _: tune_config,
        backend = "ray",
        n_trials = n_trials,
        compute_objective = operator.itemgetter(_F1_STRICT),
        direction = "maximize",
        local_dir = str(output_path),
        name = name or "<UNKNOWN>",
        log_to_file = True,
        scheduler = ASHAScheduler(
            metric = _F1_STRICT,
            mode = "max",
            grace_period = 5, # epochs
            max_t = 50,
            stop_last_trials = True,
            reduction_factor = 1.5,
        ),
        progress_reporter = tune.CLIReporter(
            metric_columns = ["loss", _F1_STRICT],
            max_progress_rows=10, max_report_frequency=5
        ),
        search_alg = OptunaSearch(
            metric = _F1_STRICT,
            mode = "max",
        ),
    )

    print("BEST HYPERPARAMS: ", best_result.hyperparameters)

def _transpose_dict(d: dict[str, Sequence[Any]] | None):
    if d:
        keys = d.keys()

        for elems in zip(*(d[k] for k in keys)):
            yield {k:v for k, v in zip(keys, elems)}


def _decode(tokens):
    tokens_decoded = _get_tokenizer().batch_decode(
        [t for t in tokens if t != 0],
        skip_special_tokens = True,
    )

    return [t.replace(" ", "") for t in tokens_decoded]

def predict_examples(model, examples: E):
    examples["tokens_re"] = [
        _decode(entry) for entry in examples["input_ids"]
    ]

    # feed data to the model
    predictions_raw = model.forward(
        input_ids = torch.tensor(examples["input_ids"]),
        attention_mask = torch.tensor(examples["attention_mask"]),
        token_type_ids = torch.tensor(examples["token_type_ids"]),
    ).logits

    # 
    match predictions_raw:
        case torch.Tensor():
            predictions: np.ndarray = predictions_raw.argmax(dim = 2).numpy()
        case np.ndarray():
            predictions: np.ndarray = predictions_raw.argmax(axis = 2)
        case _:
            raise TypeError
        
    examples["label_ids_predicted"] = predictions

    examples["comp_predicted"] = [
        convert_vector_to_span(i, p)
        for i, p in zip(examples["input_ids"], predictions)
    ]
    
    return examples

def test(
    model_path: Optional[Path] = typer.Option(
        None, "--model", "-m",
    ),
    test_path: Optional[Path] = typer.Option(
        None, "--test-data", "-t",
    ),
    test_revision: str = typer.Option(
        "main", "--test-revision",
    ),
    evaluator_path: Optional[Path] = typer.Option(
        None, "--evaluator",
        file_okay = False,
    ),
    dump: Optional[Path] = typer.Option(
        None, "--dump",
        file_okay = True,
        dir_okay = False,
    ),
    batch_size: Optional[int] = typer.Option(
        128, "--batch-size", "-b",
    ),
):
    model = BertForTokenClassification.from_pretrained(
        str(model_path)
        if model_path
        else "abctreebank/comparative-NER"
    )
 
    if not isinstance(model, BertForTokenClassification):
        raise TypeError(f"model: {type(model)} is not an instance of BertForTokenClassification")

    dataset = datasets.load_dataset(
        (
            str(test_path)
            if test_path
            else "abctreebank/comparative-NER-BCCWJ"
        ),
        revision = test_revision,
        use_auth_token = True,
        split = "test",
    )
    if not isinstance(dataset, datasets.Dataset):
        raise TypeError(f"dataset: {type(dataset)} is not an instance of Dataset")

    dataset = dataset.map(
        convert_records_to_vectors,
        batched = batch_size is not None,
        batch_size = batch_size or 1,
    )

    model.eval()

    def _predict(
        examples: datasets.arrow_dataset.Example | datasets.arrow_dataset.Batch
    ):
        predictions_raw = model.forward(
            input_ids = torch.tensor(examples["input_ids"]),
            attention_mask = torch.tensor(examples["attention_mask"]),
            token_type_ids = torch.tensor(examples["token_type_ids"]),
            return_dict = True,
        ).logits
        match predictions_raw:
            case torch.Tensor():
                predictions: np.ndarray = predictions_raw.argmax(dim = 2).numpy()
            case np.ndarray():
                predictions: np.ndarray = predictions_raw.argmax(axis = 2)
            case _:
                raise TypeError
        examples["label_ids_predicted"] = predictions


        examples["comp_predicted"] = [
            convert_vector_to_span(i, p)
            for i, p in zip(examples["input_ids"], predictions)
        ]

        return examples

    dataset = dataset.map(
        _predict,
        batched = batch_size is not None,
        batch_size = batch_size or 1,
    )

    _eval: evaluate.Metric = _get_evaluator(
        str(evaluator_path)
        if evaluator_path
        else "abctreebank/comparative-NER-metrics"
    )
    _eval.add_batch(
        predictions = dataset["label_ids_predicted"],
        references = dataset["label_ids"],
        ID = dataset["ID"],
        input_ids = dataset["input_ids"],
    )
    res = _eval.compute(
        predictions = None,
        references = None,
        ID = None,
        input_ids = None,
        special_ids = _get_tokenizer().all_special_ids,
        label2id = LABEL2ID,
        id2label_detailed = ID2LABEL_DETAILED,
    )

    if res:
        dataset = dataset.add_column("error_list", res["error_list"])
        del res["error_list"]
        json.dump(res, sys.stdout)
    else:
        raise RuntimeError("Evaluator returns a null result")

    if dump:
        with open(dump, "w") as h_dump:
            yaml = ruamel.yaml.YAML()

            dump_output = [
                {
                    "ID": entry["ID"],
                    "reference_linear": br.linearlize_annotation(
                        entry["tokens"],
                        entry["comp"],
                        # list(_transpose_dict(entry["comp"])),
                        ID = entry["ID"],
                    ),
                    "prediction_linear": br.linearlize_annotation(
                        [
                            w for w in entry["token_subwords"]
                            if w not in {'[PAD]'}
                        ],
                        list(_transpose_dict(entry["comp_predicted"])),
                        ID = entry["ID"],
                    ),
                    "errors": entry["error_list"],
                }
                for entry in dataset
            ]

            yaml.dump(dump_output, h_dump)

def predict(
    model_path: str,
):
    pipeline = TokenClassificationPipeline(
        model = BertForTokenClassification.from_pretrained(
            model_path
        ),
        tokenizer = _get_tokenizer(),
        task = "comparative-NER",
        aggregation_strategy = "simple",
    )

    print(pipeline(list(sys.stdin)))