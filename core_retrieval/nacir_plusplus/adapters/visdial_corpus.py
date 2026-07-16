"""NACIR++ plug-and-play VisDial/PlugIR data adapter.

This module wraps the original `Corpus` and `Queries` torch datasets used in
the main scripts without changing their logic. It simply separates the dataset
specific code from the orchestrator so it can be reused when building
corpus_vectors and RetrievalSession objects for the pipeline.

This is the dataset-specific layer. Each method or dataset can provide its own
adapter as long as it produces corpus_vectors [N, D] and a list of
RetrievalSession objects that match schema.py.
"""

import json
import os
from typing import Callable, List

import torch


class Corpus(torch.utils.data.Dataset):
    """Load the list of image paths from the corpus JSON without changing logic."""

    def __init__(self, data_dir: str, corpus_path: str, preprocessor: Callable):
        with open(corpus_path) as f:
            self.corpus = [os.path.join(data_dir, p) for p in json.load(f)]
        self.preprocessor = preprocessor
        self.path2id = {p: i for i, p in enumerate(self.corpus)}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path: str) -> int:
        return self.path2id[path]

    def __getitem__(self, i):
        return {"id": i, "image": self.preprocessor(self.corpus[i])}


class Queries(torch.utils.data.Dataset):
    """Read dialog entries by round (`dialog_length`) without changing logic."""

    def __init__(self, queries_path: str, data_dir: str, sep_token: str = ", ", split: bool = True):
        with open(queries_path) as f:
            self.queries = json.load(f)
        self.dialog_length = None
        self.data_dir = data_dir
        self.sep_token = sep_token
        self.split = split

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        target_path = os.path.join(self.data_dir, self.queries[i]["img"])
        text = (
            self.queries[i]["dialog"][self.dialog_length]
            if self.split
            else self.sep_token.join(self.queries[i]["dialog"][: self.dialog_length + 1])
        )
        return {"text": text, "target_path": target_path}


def load_corpus_paths(data_dir: str, corpus_path: str) -> List[str]:
    with open(corpus_path) as f:
        return [os.path.join(data_dir, p) for p in json.load(f)]
