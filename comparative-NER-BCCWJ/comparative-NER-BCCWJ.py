import json

import datasets

_DESCRIPTION = """\
Annotation data of Japanese comparative constructions.

The constructions include the -yori-{∅/ka/mo} and the -{to/ni}kurabe-te construction.
Texts are originated from the BCCWJ corpus.
"""

class ComparativeNERBCCWJ(datasets.GeneratorBasedBuilder):
    """\
    Annotation data of Japanese comparative constructions in the BCCWJ corpus.
    """

    def _info(self):
        return datasets.DatasetInfo(
            description = _DESCRIPTION,
            features = datasets.Features(
                {
                    "ID": datasets.Value("string"),
                    "tokens": datasets.Sequence(
                        datasets.Value("string")
                    ),
                    "comp": datasets.Sequence(
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                            "label": datasets.Value("string"),
                        }
                    )
                }
            ),
            supervised_keys = None,
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        import os
        print(os.getcwd())
        return [
            datasets.SplitGenerator(
                name = datasets.Split.TRAIN,
                gen_kwargs = {
                    "filepath": f"./train.jsonl",
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name = datasets.Split.TEST,
                gen_kwargs = {
                    "filepath": f"./test.jsonl",
                    "split": "test",
                }
            ),
        ]
    
    def _generate_examples(self, filepath, split):
        with open(filepath) as f:
            for line in f:
                if not line.strip():
                    continue

                entry: dict = json.loads(line)

                # del entry["text"]
                yield entry["ID"], entry