"""
Zero-RAM Disk-Streaming Dataset Iterator.
Streams training tokens directly from disk line-by-line with dynamic tokenization
and batch collation, avoiding loading the dataset into RAM.
"""

import os
import json
from typing import Iterator, Dict, Any, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class StreamingDatasetIterator:
    """
    Disk-streaming dataset reader that yields micro-batches of token IDs.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_seq_len: int = 512,
        pad_token_id: int = 163839,
        eos_token_id: int = 163586,
        bos_token_id: int = 163584,
        tokenizer_dir: Optional[str] = None,
    ):
        self.jsonl_path = jsonl_path
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tokenizer_dir = tokenizer_dir
        self._tokenizer = None
        self._tokenizer_loaded = False

    def _get_tokenizer(self):
        """
        Load Kimi K3's own tiktoken-based tokenizer.

        Without it the token ids mean nothing to the model: the byte fallback below maps
        each UTF-8 byte to `byte + 100`, which has no relation to the 163840-entry
        vocabulary, so any loss computed from it is meaningless.
        """
        if self._tokenizer_loaded:
            return self._tokenizer

        self._tokenizer_loaded = True
        if not self.tokenizer_dir or not os.path.isdir(self.tokenizer_dir):
            return None

        try:
            import sys
            if self.tokenizer_dir not in sys.path:
                sys.path.insert(0, self.tokenizer_dir)
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_dir, trust_remote_code=True
            )
        except Exception as exc:
            print(f"[!] Kimi K3 tokenizer could not be loaded ({type(exc).__name__}: {exc});"
                  f" falling back to byte ids, which makes the loss meaningless.")
            self._tokenizer = None
        return self._tokenizer

    def _simple_tokenize(self, text: str) -> List[int]:
        """Tokenize with the model's own tokenizer, or fall back to raw bytes."""
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            ids = tokenizer.encode(text)[: self.max_seq_len - 2]
            return [self.bos_token_id] + list(ids) + [self.eos_token_id]

        tokens = [self.bos_token_id]
        byte_tokens = [int(b) + 100 for b in text.encode("utf-8")[: self.max_seq_len - 2]]
        tokens.extend(byte_tokens)
        tokens.append(self.eos_token_id)
        return tokens

    def iterate_samples(self) -> Iterator[Dict[str, Any]]:
        """Yield individual parsed JSON objects from disk."""
        if not os.path.exists(self.jsonl_path):
            return
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue

    def get_batches(
        self,
        batch_size: int = 1,
        as_torch: bool = True,
        device: str = "cpu",
    ) -> Iterator[Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]]:
        """
        Yields (input_ids, target_ids) micro-batches.
        """
        current_batch_tokens = []

        for sample in self.iterate_samples():
            text = sample.get("formatted_text", "")
            if not text:
                text = f"{sample.get('instruction', '')} {sample.get('input', '')} {sample.get('output', '')}"
            token_ids = self._simple_tokenize(text)
            
            # Truncate / Pad to max_seq_len
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            else:
                token_ids = token_ids + [self.pad_token_id] * (self.max_seq_len - len(token_ids))

            current_batch_tokens.append(token_ids)

            if len(current_batch_tokens) == batch_size:
                arr = np.array(current_batch_tokens, dtype=np.int64)
                input_ids = arr[:, :-1]
                target_ids = arr[:, 1:]

                if as_torch and HAS_TORCH:
                    t_in = torch.from_numpy(input_ids)
                    t_target = torch.from_numpy(target_ids)
                    if device != "cpu" and torch.cuda.is_available():
                        t_in = t_in.to(device)
                        t_target = t_target.to(device)
                    yield t_in, t_target
                else:
                    yield input_ids, target_ids

                current_batch_tokens = []

        if current_batch_tokens:
            arr = np.array(current_batch_tokens, dtype=np.int64)
            input_ids = arr[:, :-1]
            target_ids = arr[:, 1:]
            if as_torch and HAS_TORCH:
                t_in = torch.from_numpy(input_ids)
                t_target = torch.from_numpy(target_ids)
                if device != "cpu" and torch.cuda.is_available():
                    t_in = t_in.to(device)
                    t_target = t_target.to(device)
                yield t_in, t_target
            else:
                yield input_ids, target_ids
