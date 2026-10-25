"""Thin wrapper around the GPT-2 byte-pair encoder from ``tiktoken``.

The notebook called ``tiktoken.get_encoding("gpt2")`` inline in three different
places. Wrapping it keeps the loading lazy (so importing this module never hits
the network) and gives the rest of the codebase a single place to ask about
vocabulary size and token pieces.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import tiktoken

ENCODING_NAME = "gpt2"

# The notebook passed ``allowed_special={""}`` which is just an empty set, so no
# special tokens are ever whitelisted. Kept explicit here because silently
# allowing <|endoftext|> would change encoding behaviour.
ALLOWED_SPECIAL: set[str] = set()


class GPT2Tokenizer:
    """Encode/decode text with the GPT-2 BPE vocabulary."""

    def __init__(self, encoding_name: str = ENCODING_NAME) -> None:
        self.encoding_name = encoding_name
        self._enc = None

    @property
    def encoder(self) -> tiktoken.Encoding:
        if self._enc is None:
            self._enc = tiktoken.get_encoding(self.encoding_name)
        return self._enc

    @property
    def vocab_size(self) -> int:
        """Vocabulary size including the 50257 end-of-text slot."""
        return self.encoder.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text, allowed_special=ALLOWED_SPECIAL)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.encoder.decode(list(token_ids))

    def token_piece(self, token_id: int) -> str:
        """The raw byte-level piece for a single token id."""
        return self.encoder.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")

    def token_pieces(self, token_ids: Sequence[int]) -> list[str]:
        """One string piece per token id, so the UI can colour tokens individually."""
        return [self.token_piece(t) for t in token_ids]

    def end_of_text_id(self) -> int:
        return self.encoder.eot_token


@lru_cache(maxsize=1)
def get_tokenizer() -> GPT2Tokenizer:
    """Process-wide tokenizer singleton (loading the BPE ranks is not free)."""
    return GPT2Tokenizer()