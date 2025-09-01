import pickle
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.merge_2_priority: dict[tuple[bytes, bytes], int] = {
            merge: priority for priority, merge in enumerate(merges)
        }  # lower priority = merge first
        self.special_tokens = special_tokens
        if special_tokens is not None:
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token_bytes
        self.reverse_vocab: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "rb") as f:
            vocab: dict[int, bytes] = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges: list[tuple[bytes, bytes]] = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        # pretokenize first
        pretokens: list[bytes] = self._pretokenize(text)
        # encode by applying merges
        token_ids = []
        for pretoken in pretokens:
            token_ids.extend(self._apply_merges_new(pretoken))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory.
        """
        for text in iterable:
            # encode returns a list[int]
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        symbols: list[bytes] = [self.vocab[token_id] for token_id in ids]
        # concate all the symbols
        symbols_concat: bytes = b"".join(symbols)
        return symbols_concat.decode(encoding="utf-8", errors="replace")

    def _pretokenize(self, text: str) -> list[bytes]:
        """
        'apple<|endoftext|>Jobs'
        =>
        [b'apple', b'<|endoftext|>', b'Jobs']
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens: list[bytes] = []
        if self.special_tokens is not None:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)

            pattern = re.compile(
                "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
            )
            parts = re.split(pattern, text)
            pretokens: list[bytes] = []
            for part in parts:
                if part in self.special_tokens:
                    pretoken = part.encode("utf-8")  # str to bytes
                    pretokens.append(pretoken)
                else:
                    re_iter = re.finditer(PAT, part)
                    for s in re_iter:
                        pretoken = s.group(0).encode("utf-8")  # str to bytes
                        pretokens.append(pretoken)
        else:
            re_iter = re.finditer(PAT, text)
            for s in re_iter:
                pretoken: bytes = s.group(0).encode("utf-8")  # str to bytes
                pretokens.append(pretoken)
        # print("tlu7 pretokens, ", pretokens)
        return pretokens

    def _apply_merges_new(self, pretoken: bytes) -> list[int]:
        """
        Apply merges to a single pretoken
        b'the'
        vocab: {0: a, 1: th, 2: e, ...}
        =>
        [1, 2]
        """
        # special token handling
        if self.special_tokens and pretoken.decode("utf-8") in self.special_tokens:
            return [self.reverse_vocab[pretoken]]
        symbols: list[bytes] = [bytes([symbol]) for symbol in pretoken]  # [t h e]
        if len(symbols) == 1:
            return [self.reverse_vocab[symbols[0]]]

        can_merge = True

        while can_merge:
            # go through the list of merge to identify mergeable adjacent pair
            can_merge = False
            pairs: list[tuple[bytes, bytes]] = list(zip(symbols[:-1], symbols[1:]))
            can_merge_pairs = [
                (pair, self.merge_2_priority[pair])
                for pair in pairs
                if pair in self.merge_2_priority
            ]
            # print("can_merge_pairs, ", can_merge_pairs)

            if len(can_merge_pairs) > 0:
                new_symbols = []
                can_merge = True
                to_merge: tuple[bytes, bytes] = min(
                    can_merge_pairs, key=lambda p: p[1]
                )[0]
                i = 0
                while i < len(symbols):
                    if (
                        i < len(symbols) - 1
                        and (symbols[i], symbols[i + 1]) == to_merge
                    ):  # to merge
                        new_symbols.append(symbols[i] + symbols[i + 1])  # new bigram
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
                # print("tlu7 ... symbols, ", symbols)

        # print("symbols, ", symbols)
        token_ids: list[int] = [self.reverse_vocab[symbol] for symbol in symbols]
        return token_ids
