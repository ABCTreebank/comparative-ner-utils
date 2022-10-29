from collections import defaultdict, deque
import attr
from enum import Enum, auto
from typing import Iterable, Any, TextIO, Iterator
import re

class BracketPart(Enum):
    START = auto()
    END = auto()

_LABEL_WEIGHT = defaultdict(lambda: 0)
_LABEL_WEIGHT["root"] = -100

def _mod_token(
    token: str,
    feats: Iterable[tuple[str, BracketPart]]
) -> str:
    for feat in feats:
        match feat:
            case label, BracketPart.START:
                token = f"[{token}"
            case label, BracketPart.END:
                token = f"{token}]{label}"
            case _:
                raise ValueError(
                    f"Illegal comparative feature tuple {feat}"
                )
    return token

def linearlize_annotation(
    tokens: Iterable[str],
    comp: Iterable[dict[str, Any]],
    ID: str = "<UNKNOWN>"
) -> str:
    """
    linearlize a comparative NER annotation.

    Examples
    --------
    >>> linearlize_annotation(
    ...     tokens = ["太郎", "花子", "より", "賢い"],
    ...     comp = [{"start": 1, "end": 2, "label": "prej"}],
    ... )
    "太郎 [花子 より]prej 賢い"
    """

    feats_pos: defaultdict[
        int,
        deque[tuple[str, BracketPart]]
    ] = defaultdict(deque)

    for feat in sorted(
        comp,
        key = lambda x: _LABEL_WEIGHT[x["label"]]
    ):
        match feat:
            case {"start": b, "end": e, "label": l}:
                feats_pos[b].append( (l, BracketPart.START) )
                feats_pos[e - 1].append( (l,  BracketPart.END) )
            case _:
                raise ValueError(
                    f"Illegal comparative feature {feat} "
                    f"in Record ID {ID}"
                )
            
    return ' '.join(
        _mod_token(t, feats_pos[idx])
        for idx, t in enumerate(tokens)
    )

def dict_to_bracket(datum: dict[str, Any]):
    token_bred = linearlize_annotation(
        datum["tokens"],
        datum["comp"],
        ID = datum["ID"]
    )
    return f"{datum['ID']} {token_bred}\n"

_RE_TOKEN_BR_CLOSE = re.compile(r"^(?P<token>[^\]]+)\](?P<feat>[a-z0-9]+)(?P<rem>.*)$")
def delinearlize_annotation(
    line: str
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Parse a linearlized a comparative NER annotation.

    Examples
    --------
    >>> delinearlize_annotation(
    ...     "太郎 [花子 より]prej 賢い"
    ... )
    (
        ["太郎", "花子", "より", "賢い"],
        [{"start": 1, "end": 2, "label": "prej"}],
    )
    """

    tokens = line.split(" ")

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

    return res_token_list, comp_dict_list

def bracket_to_dict(line: str):
    line_split = line.split(" ", 1)
    ID, text = line_split[0], line_split[1]

    tokens, comp = delinearlize_annotation(text)

    return {
        "ID": ID,
        "tokens": tokens,
        "comp": comp,
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