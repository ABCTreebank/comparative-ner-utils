import itertools
from pathlib import Path
import re
import sys
from typing import Optional

import typer

import torch
import torch.utils.data

import datasets
from transformers import BertConfig, BertForTokenClassification, BertJapaneseTokenizer, EvalPrediction, IntervalStrategy, TokenClassificationPipeline, Trainer, TrainingArguments
import evaluate

LABEL2ID_DETAILED = {
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

_RE_FEAT_ARTIFACTS = re.compile(r"^(?P<name>[a-zA-Z]+)[0-9]")

def train(
    data_path: Optional[Path] = typer.Option(
        None, "--data",
        file_okay = False,
    ),
    evaluator_path: Optional[Path] = typer.Option(
        None, "--evaluator",
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
    
    def _add_vectors(entry: datasets.arrow_dataset.Example):
        labels = list(itertools.repeat("O", _MAX_LENGTH))

        if (ent_comps := entry["comp"]):
            for s, e, l in zip(
                ent_comps["start"],
                ent_comps["end"],
                ent_comps["label"],
            ) if ent_comps["start"] else ():
                if l == "root":
                    continue
                elif (match := _RE_FEAT_ARTIFACTS.match(l)):
                    # remove artifacts
                    l = match.group("name")

                s += 1
                e += 1
                # initial padding tokenを考慮

                labels[s] = f"B-{l}"
                for i in range(s + 1, e):
                    labels[i] =  f"I-{l}"
        else:
            pass

        for k, v in _get_tokenizer()(
            entry["tokens"],
            is_split_into_words = True,
            return_tensors = "pt", 
            max_length = _MAX_LENGTH,
            padding = "max_length",
        ).items():
            entry[k] = torch.reshape(v, (-1, ) )

        entry["label_ids"] =  [LABEL2ID[l] for l in labels]

        return entry
    # === END DEF _add_vectors ===

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
        output_dir='./results',
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
        seed = 453123,
    )

    def _compute_metric(pds: EvalPrediction) -> dict:
        # ensure cache
        _eval.add_batch(
            predictions = pds.predictions,
            references = pds.label_ids,
        )
        
        return _eval._compute(
            predictions = pds.predictions,
            references = pds.label_ids,
            input_ids = pds.inputs,
            special_ids = _get_tokenizer().all_special_ids,
            label2id = LABEL2ID,
            id2label_detailed = ID2LABEL_DETAILED,
        ) or {}
    
    trainer = Trainer(
        model = model,
        args = training_args,
        compute_metrics = _compute_metric,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()


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