from collections import defaultdict, deque
import attr
from enum import Enum, auto
from typing import Iterable, Literal, Any, TextIO, Iterator
import re

_FEAT = tuple[str, Literal["start", "end"]]
_LABEL_WEIGHT = defaultdict(lambda: 0)
_LABEL_WEIGHT["root"] = -100

def dict_to_bracket(datum: dict[str, Any]):
    comp_feats: Iterable[dict[str, Any]] = datum["comp"] or tuple()
    feats_pos: defaultdict[int, deque[_FEAT]] = defaultdict(deque)

    for feat in sorted(
        comp_feats, 
        key = lambda x: _LABEL_WEIGHT[x["label"]]
    ):
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
_RE_TOKEN_BR_CLOSE = re.compile(r"^(?P<token>[^\]]+)\](?P<feat>[a-z0-9]+)(?P<rem>.*)$")

def bracket_to_dict(line: str):
    line_split = line.split(" ")
    ID, tokens = line_split[0], line_split[1:]

    res_token_list = []
    comp_dict_list = []
    stack_br_open: list[int] = []

    for i, token in enumerate(tokens):
        while token.startswith("["):
            stack_br_open.append(i)
            token = token[1:]

        while (match := _RE_TOKEN_BR_CLOSE.search(token)):
            start = stack_br_open.pop()
            comp_dict_list.append(
                {
                    "start": start,
                    "end": i + 1,
                    "label": match.group("feat")
                }
            )
            token = match.group("token") + match.group("rem")

        res_token_list.append(token)

    return {
        "ID": ID,
        "tokens": res_token_list,
        "comp": comp_dict_list,
    }

_RE_COMMENT = re.compile(f"^//")

@attr.s(slots = True, auto_attribs = True)
class TextAnalysisFileParseRecord:
    data_ref: dict[str, Any]
    data_predict: dict[str, Any]
    annot_labels: set[str]
    cont_role: str
    comments: list[str]

    def get_ID(self) -> str:
        return self.data_ref["ID"]

class TextAnalysisFileParserState(Enum):
    DATA_REF = auto()
    DATA_PREDICT = auto()
    ANNOT_LABEL = auto()
    CONT_ROLE = auto()
    COMMENT = auto()

def parse_test_analysis_file(stream: TextIO) -> Iterator[TextAnalysisFileParseRecord]:
    state: TextAnalysisFileParserState = TextAnalysisFileParserState.DATA_REF
    record: dict[str, Any] = {}

    line = stream.readline()
    while line:
        # only when the line is not empty
        if (line := line.strip()):
            match state:
                case TextAnalysisFileParserState.DATA_REF:
                    if record:
                        # yield the previous record
                        yield TextAnalysisFileParseRecord(**record)

                    # clear and init the record and parse the reference data
                    record = {
                        "data_ref": bracket_to_dict(line),
                        "comments": [],
                    }

                    # transition
                    state = TextAnalysisFileParserState.DATA_PREDICT

                case TextAnalysisFileParserState.DATA_PREDICT:
                    # parse the prediction
                    record["data_predict"] = bracket_to_dict(line)

                    # transition
                    state = TextAnalysisFileParserState.ANNOT_LABEL

                case TextAnalysisFileParserState.ANNOT_LABEL:
                    # parse annotation labels
                    if (match := _RE_COMMENT.match(line)):
                        record["annot_labels"] = set(
                            item.strip()
                            for item in line[match.end():].split(",")
                        )
                        # transition
                        state = TextAnalysisFileParserState.CONT_ROLE
                    else:
                        raise ValueError(
                            f"Expect a comment line for labels, but got: {line}"
                        )

                case TextAnalysisFileParserState.CONT_ROLE:
                    # parse the role of the contrast
                    if (match := _RE_COMMENT.match(line)):
                        record["cont_role"] = line[match.end():].strip()

                        # transition
                        state = TextAnalysisFileParserState.COMMENT
                    else:
                        raise ValueError(
                            f"Expect a comment line for the contrast role, but got: {line}"
                        )

                case TextAnalysisFileParserState.COMMENT:
                    # parse the role of the contrast
                    if (match := _RE_COMMENT.match(line)):
                        record["comments"].append(line[match.end():].strip())

                        # no transition
                    else:
                        # transition
                        state = TextAnalysisFileParserState.DATA_REF
                        # skip the line feeding and continue immediately
                        continue
                case _:
                    raise TypeError(f"The state cannot be {state} of type {type(state)}")

        # read the next line
        line = stream.readline()
    # === END WHILE ===

    if record:
        # yield the last record
        yield TextAnalysisFileParseRecord(**record)