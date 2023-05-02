import json
import os
from pathlib import Path
import random
import sys
import dataclasses
from typing import Any, Optional

import ruamel.yaml
import typer

import numpy as np
import huggingface_hub.hf_api
import datasets

import abct_comp_ner_utils.train
import abctk.obj.comparative as aoc

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
app.command("find-hyper")(abct_comp_ner_utils.train.find_hyperparameters)
app.command("predict")(abct_comp_ner_utils.train.predict)
app.command("test")(abct_comp_ner_utils.train.test)

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

    dataset = datasets.load_dataset(".")

    dataset.push_to_hub(
        "abctreebank/comparative-NER-BCCWJ",
        private = private,
        branch = branch,
        token = huggingface_hub.hf_api.read_from_credential_store()[1]
    )

@app.command("split-by-ID-list")
def split_data_by_ID_list(
    id_list_file: typer.FileText = typer.Argument(
        ...,
        exists = True,
        help = """
            File containing test IDs separated by line breaks.
        """
    ),
    output_dir: Path = typer.Option(
        Path("."),
        file_okay = False,
        dir_okay = True,
    )
) -> None:
    """
    Split data into train and test data by specified IDs.

    The input data is given in the JSONL format.
    """

    data: list[dict[str, Any]] = list(json.loads(r) for r in sys.stdin)
    test_IDs: set[str] = set(i.strip() for i in id_list_file)

    data_train = [
        record for record in data 
        if record["ID"] not in test_IDs
    ]
    data_test = [
        record for record in data 
        if record["ID"] in test_IDs
    ]

    # Ensure output dir
    os.makedirs(output_dir, exist_ok = True)

    # Output
    with open(output_dir / "train.jsonl", "w") as h_train:
        h_train.writelines(
            json.dumps(r, ensure_ascii = False) + "\n"
            for r in data_train
        )

    with open(output_dir / "test.jsonl", "w") as h_test:
        h_test.writelines(
            json.dumps(r, ensure_ascii = False) + "\n"
            for r in data_test
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
    Randomly split data into train and test data.

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


@app.command("comp2br")
def comp_to_bracket():
    """
    Convert comparative JSON data into texts with brackets.

    The input data is given via STDIN in the JSONL format.
    """
    sys.stdout.writelines(
        aoc.CompRecord.to_brackets_full(json.loads(line))
        for line in sys.stdin
    )

@app.command("br2comp")
def bracket_to_comp(
    input_type: str = typer.Option(
        "txt",
        "--input-type", "-i",
    ),
    output_type: str = typer.Option(
        "jsonl",
        "--output-type", "-o",
    ),
    to_dice: bool = typer.Option(
        False,
        "--dice",
    ),
):
    """
    Convert comparative data in text format into JSON.

    The input data is given via STDIN.
    Each line corresponds to exactly one example.
    """
    input_type = input_type.lower()
    output_type = output_type.lower()

    yaml = None

    if input_type == "txt":
        records = aoc.CompRecord.read_bracket_annotation_file(sys.stdin)
    elif input_type == "yaml":
        yaml = yaml or ruamel.yaml.YAML()
        records = (
            aoc.CompRecord.from_brackets(
                line = record["annot"],
                ID = record["ID"],
                comments = record.get("comments"),
                ID_v1 = record.get("ID_v1"),
            )
            for record in yaml.load(sys.stdin)
        )
    elif input_type == "jsonl":
        records = (
            aoc.CompRecord.from_brackets(
                line = record["annot"],
                ID = record["ID"],
                comments = record.get("comments"),
                ID_v1 = record.get("ID_v1"),
            )
            for line in sys.stdin
            for record in (json.loads(line), )
        )
    else:
        raise ValueError(f"{input_type} is an invalid input file type")

    if to_dice:
        records = (rec.dice() for rec in records)

    if output_type == "jsonl":
        for rec in records:
            json.dump(dataclasses.asdict(rec), sys.stdout, ensure_ascii = False)
            sys.stdout.write("\n")
    elif output_type == "yaml":
        records_asdict = tuple(dataclasses.asdict(rec) for rec in records)
        yaml = yaml or ruamel.yaml.YAML()
        yaml.representer.add_representer(
            np.int64, 
            lambda repr, i: repr.represent_scalar("tag:yaml.org,2002:int", str(i))
        )

        yaml.dump(records_asdict, sys.stdout)
    else:
        raise ValueError(f"{output_type} is an invalid output file type")
