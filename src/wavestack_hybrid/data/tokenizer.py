"""Tokenizer wrapper built on top of tiktoken."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken


@dataclass
class TokenizerWrapper:
    """Thin wrapper providing encode/decode helpers."""

    name: str = "gpt2"

    def __post_init__(self):
        self._tokenizer = tiktoken.get_encoding(self.name)
        self.pad_id = self._tokenizer.eot_token

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)
