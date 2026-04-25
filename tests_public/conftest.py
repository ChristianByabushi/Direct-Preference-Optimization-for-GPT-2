from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TinyTokenizer:
    """Whitespace tokenizer for unit tests.

    It exposes the small subset of the Hugging Face tokenizer API used by the
    homework starter code.
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = text.strip().split()
        ids: List[int] = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


@pytest.fixture()
def tiny_tokenizer() -> TinyTokenizer:
    return TinyTokenizer()
