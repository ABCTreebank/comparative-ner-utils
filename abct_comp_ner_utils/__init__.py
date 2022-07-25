from collections import defaultdict, deque
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Literal, Optional, Sequence

import typer

import huggingface_hub.hf_api 
import datasets

import abct_comp_ner_utils.train

app = typer.Typer()

@app.callback()
def app_init(
    token: Optional[str] = typer.Option(
        None,
    )
):
    if token:
        huggingface_hub.hf_api.set_access_token(token)
        typer.echo("HF Hub login succeed", err = True)

app.command("train")(abct_comp_ner_utils.train.train)
app.command("predict")(abct_comp_ner_utils.train.predict)

@app.command("upload-data")
def upload_data(
    data_path: Path = typer.Argument(
        ...,
        dir_okay = True,
        file_okay = False,
    ),
    private: bool = False,
    branch: Optional[str] = None,
):
    os.chdir(data_path)

    dataset = datasets.load_dataset(str(data_path))

    dataset.push_to_hub(
        "abctreebank/comparative-NER-BCCWJ",
        private = private,
        branch = branch,
        token = huggingface_hub.hf_api.read_from_credential_store()[1]
    )

@app.command("split")
def split_data(
    test_ratio: float = typer.Option(
        0.1, min = 0, max = 1,
    ),
    output_dir: Path = typer.Option(
        Path("."),
        file_okay = False,
        dir_okay = True,
    ),
    random_state: int = 2022_07_23,
):
    """
    Split data into train and test data.

    The input data is given via STDIN with records separated by a line break.
    """

    # Set random seed
    random.seed(random_state)

    # Load data
    data: list[str] = list(sys.stdin)

    # Shuffle data
    random.shuffle(data)

    # Split data
    split_point = round(len(data) * test_ratio)
    data_test = data[:split_point]
    data_train = data[split_point:]

    # Ensure output dir
    os.makedirs(output_dir, exist_ok = True)

    # Output
    with open(output_dir / "train.jsonl", "w") as h_train:
        h_train.writelines(data_train)

    with open(output_dir / "test.jsonl", "w") as h_test:
        h_test.writelines(data_test)

_FEAT = tuple[str, Literal["start", "end"]]

def dict_to_bracket(datum: dict[str, Any]):
    comp_feats: Iterable[dict[str, Any]] = datum["comp"] or tuple()
    feats_pos: defaultdict[int, deque[_FEAT]] = defaultdict(deque)

    for feat in comp_feats:
        match feat:
            case {"start": b, "end": e, "label": l}:
                feats_pos[b].append(
                    (l, "start")
                )
                feats_pos[e - 1].appendleft(
                    (l, "end")
                )
            case _:
                raise ValueError(
                    f"Illegal comparative feature {feat} "
                    f"in {datum.get('ID', '<UNKNOWN>')}"
                )

    def _mod_token(token: str, feats: Iterable[_FEAT]) -> str:
        for feat in feats:
            match feat:
                case label, "start":
                    token = f"[{token}"
                case label, "end":
                    token = f"{token}]{label}"
                case _:
                    raise ValueError(
                        f"Illegal comparative feature tuple {feat}"
                    )
        return token

    token_bred = ' '.join(
        _mod_token(t, feats_pos[idx])
        for idx, t in enumerate(datum['tokens'])
    )

    return f"{datum['ID']} {token_bred}\n"

_RE_BR_OPEN = re.compile(r"^\[(?P<rem>.*)")
_RE_BR_CLOSE = re.compile(r"(?P<rem>.*)\](?P<label>[a-z0-9]+)$")

def _parse_br(token: str) -> tuple[int, Any]:
    pos_begin = 0
    pos_end = len(token)
    br_closed_list = []

    while (match := _RE_BR_OPEN.search(token, pos_begin)):
        token = match.group("rem")
        pos_begin += 1
    br_open_count = pos_begin

    while (match := _RE_BR_CLOSE.search(token, pos_begin, pos_end)):
        token =  match.group("rem")
        label = match.group("label")

        br_closed_list.append(label)
        pos_end -= len(label)
        pos_end -= 1

    return br_open_count, br_closed_list

def bracket_to_dict(datum: str) -> dict[str, Any]:
    token_list_raw = datum.split()
    ID, token_list_raw = token_list_raw[0], token_list_raw[1:]
    token_list = []

    for idx, token in enumerate(token_list_raw):
        open_count, close_labels = _parse_br(token)

    return {
        "ID": ID,

    }

@app.command("jsonl2br")
def jsonl_to_bracket():
    """
    Convert comparative JSON data into texts with brackets.

    The input data is given via STDIN in the JSONL format.
    """
    sys.stdout.writelines(
        dict_to_bracket(json.loads(line))
        for line in sys.stdin
    )

@app.command("br2jsonl")
def bracket_to_jsonl():
    pass