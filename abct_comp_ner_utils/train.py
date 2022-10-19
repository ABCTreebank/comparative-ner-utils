import itertools
import json
import operator
from pathlib import Path
import re
import sys
from typing import Optional, Sequence, Any

import typer
import flatten_dict

import numpy as np
import torch
import torch.utils.data

import datasets
from transformers import BertConfig, BertForTokenClassification, BertJapaneseTokenizer, EvalPrediction, IntervalStrategy, TokenClassificationPipeline, Trainer, TrainingArguments
import evaluate

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch

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

_MAX_LENGTH = 256
_BERT_MODEL = "cl-tohoku/bert-base-japanese-whole-word-masking"

_TOKENIZER = None
def _get_tokenizer() -> BertJapaneseTokenizer:
    global _TOKENIZER
    _TOKENIZER = _TOKENIZER or BertJapaneseTokenizer.from_pretrained(_BERT_MODEL)

    return _TOKENIZER


_EVALUATOR = dict()

def _get_evaluator(name: str):
    global _EVALUATOR
    _EVALUATOR[name] = _EVALUATOR.get(name, None) or evaluate.load(name)
    return _EVALUATOR[name]

_RE_FEAT_ARTIFACTS = re.compile(r"^(?P<name>[a-zA-Z]+)[0-9]?")

def _add_vectors(entry: datasets.arrow_dataset.Example):
    for k, v in _get_tokenizer()(
        entry["tokens"],
        is_split_into_words = True,
        return_tensors = "pt", 
        max_length = _MAX_LENGTH,
        padding = "max_length",
    ).items():
        entry[k] = torch.reshape(v, (-1, ) )

    ID_CLS: int = _get_tokenizer().cls_token_id or 2
    ID_SEP: int = _get_tokenizer().sep_token_id or 3
    labels = list(itertools.repeat(LABEL2ID["O"], _MAX_LENGTH))
    pos_word: int = -1
    current_feat: str = "O"
    current_feat_is_start: bool = True
    current_feat_end_pos_word: int = -1
    
    ent_comps = entry["comp"] or None
    
    match ent_comps:
        case dict():
            feat_list: list[tuple[int, int, str]] = list(
                zip(
                    ent_comps["start"],
                    ent_comps["end"],
                    ent_comps["label"],
                )
            )
        case list():
            feat_list = [
                (e["start"], e["end"], e["label"])
                for e in ent_comps
            ]
        case None:
            feat_list = []
        case _:
            raise TypeError
    
    for pos_subword, (input_id, input_token) in enumerate(
        zip(
            entry["input_ids"],
            _get_tokenizer().convert_ids_to_tokens(
                entry["input_ids"]
            ),
        )
    ):
        if input_id == ID_SEP:
            # if it reaches end of sentence
            # end procedure
            break
        elif input_id == ID_CLS:
            pos_word += 1
            continue
        elif not input_token.startswith("##"):
            pos_word += 1

        # if the span reach the end
        # (based on word position)
        if current_feat_end_pos_word == pos_word:
            # reset the current feature to O
            current_feat = "O"

        # inquire the current span
        # (based on word position)
        for start, end, label in feat_list:
            # count [CLS] in
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
                labels[pos_subword] = LABEL2ID_DETAILED[current_feat, "B"]
                current_feat_is_start = False
            else:
                labels[pos_subword] = LABEL2ID_DETAILED[current_feat, "I"]

    entry["label_ids"] = labels
    return entry


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
    assert isinstance(dataset_raw, datasets.Dataset)

    dataset_raw = dataset_raw.map(
        _add_vectors,
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
        _add_vectors,
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


def test(
    model_path: Optional[Path] = typer.Option(
        None, "--model", "-m",
    ),
    test_path: Optional[Path] = typer.Option(
        None, "--test-data", "-t",
    ),
    evaluator_path: Optional[Path] = typer.Option(
        None, "--evaluator",
        file_okay = False,
    ),
    dump: Optional[Path] = typer.Option(
        None, "--dump",
        file_okay = True,
        dir_okay = False,
    )
):
    model = BertForTokenClassification.from_pretrained(
        str(model_path)
        if model_path
        else "abctreebank/comparative-NER"
    )
 
    assert(isinstance(model, BertForTokenClassification))

    dataset = datasets.load_dataset(
        (
            str(test_path)
            if test_path
            else "abctreebank/comparative-NER-BCCWJ"
        ),
        use_auth_token = True,
        split = "test",
    )
    assert(isinstance(dataset, datasets.Dataset))
    dataset = dataset.map(
        _add_vectors,
    )

    model.eval()

    def _make_spans(
        input: Sequence | np.ndarray,
        pred: Sequence | np.ndarray
    ):
        result = {
            "start": [],
            "end": [],
            "label": [],
        }

        current_label = ID2LABEL_DETAILED[0][0]
        current_span_start: int = 0

        for loc, (input_id, label_id) in enumerate(zip(input, pred)):
            label = ID2LABEL_DETAILED[label_id][0]

            if input_id == 0:
                # reached padding
                break
            elif current_label != label:
                # label changed
                # conclude the old label
                if current_label not in ("IGNORE", "O"):
                    result["start"].append(current_span_start)
                    result["end"].append(loc)
                    result["label"].append(current_label)
                else:
                    pass

                # switch to new label
                current_label = label
                current_span_start = loc

        return result

    def _decode(tokens):
        tokens_decoded = _get_tokenizer().batch_decode(
            [t for t in tokens if t != 0],
            skip_special_tokens = True,
        )

        return [t.replace(" ", "") for t in tokens_decoded]

    def _predict(
        examples: datasets.arrow_dataset.Example | datasets.arrow_dataset.Batch
    ):
        examples["tokens_re"] = [
            _decode(entry) for entry in examples["input_ids"]
        ]

        predictions_raw = model.forward(
            input_ids = torch.tensor(examples["input_ids"]),
            attention_mask = torch.tensor(examples["attention_mask"]),
            token_type_ids = torch.tensor(examples["token_type_ids"]),
        ).logits
        match predictions_raw:
            case torch.Tensor():
                predictions: np.ndarray = predictions_raw.argmax(dim = 2).numpy()
            case np.ndarray():
                predictions: np.ndarray = predictions_raw.argmax(axis = 2)
            case _:
                raise TypeError
        examples["prediction"] = predictions


        examples["comp_predicted"] = [
            _make_spans(i, p)
            for i, p in zip(examples["input_ids"], predictions)
        ]
        
        return examples

    dataset = dataset.map(
        _predict,
        batched = True,
        batch_size = 128,
    )

    if dump:
        with open(dump, "w") as h_dump:
            for entry in dataset:
                json.dump(
                    {
                        "ID": entry["ID"],
                        "tokens": entry["tokens"],
                        "comp": list(_transpose_dict(entry["comp"])),
                    },
                    h_dump,
                    ensure_ascii = False,
                )
                h_dump.write("\n")

                json.dump(
                    {
                        "ID": f"{entry['ID']}_predicted",
                        "tokens": entry["tokens_re"],
                        "comp": list(_transpose_dict(entry["comp_predicted"]))
                        ,
                    },
                    h_dump,
                    ensure_ascii = False,
                )
                h_dump.write("\n")

    _eval: evaluate.Metric = _get_evaluator(
        str(evaluator_path)
        if evaluator_path
        else "abctreebank/comparative-NER-metrics"
    )

    res = _eval._compute(
        predictions = dataset["prediction"],
        references = dataset["label_ids"],
        input_ids = dataset["input_ids"],
        special_ids = _get_tokenizer().all_special_ids,
        label2id = LABEL2ID,
        id2label_detailed = ID2LABEL_DETAILED,
    )

    json.dump(res, sys.stdout)

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