"""
NACIR++ Plug-and-Play — VisDial/PlugIR Data Adapter
=======================================================
Bọc lại `Corpus` và `Queries` (torch Dataset) gốc trong main.py, KHÔNG đổi
logic, chỉ tách nó ra khỏi orchestrator để có thể tái sử dụng riêng khi bạn
build corpus_vectors / sessions cho Pipeline.

Đây là phần "đặc thù dataset" — mỗi phương pháp/dataset khác sẽ có adapter
tương ứng của riêng nó (chỉ cần cuối cùng sinh ra corpus_vectors [N,D] và
danh sách RetrievalSession theo schema.py).
"""

import json
import os
from typing import Callable, List

import torch


class Corpus(torch.utils.data.Dataset):
    """Giữ nguyên logic gốc: load danh sách path ảnh trong corpus JSON."""

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
    """Giữ nguyên logic gốc: đọc dialog theo từng round (dialog_length)."""

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
