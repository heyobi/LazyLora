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

    def _tokenize_sample(self, sample: Dict[str, Any]) -> List[int]:
        ids, _ = self.tokenize_with_prefix(sample)
        return ids

    def tokenize_with_prefix(self, sample: Dict[str, Any]):
        """
        One dataset record -> (token ids, prompt length).

        Instruction records go through the model's own chat template, a tag format
        (`<|open|>message role="user"<|sep|> ... <|end_of_msg|>`), not ChatML. The template
        renders the assistant turn as an empty think block followed by the response, and
        appends a fresh assistant header + `<|open|>think<|sep|>` at the very end; that
        trailing generation prompt is stripped. `prompt length` is the length of the
        generation prompt (system + user + assistant header + think open): everything the
        model is given at inference. Tokens before it are not trained on.
        """
        tokenizer = self._get_tokenizer()
        instruction = (sample.get("instruction") or "").strip()
        user_input = (sample.get("input") or "").strip()
        output = (sample.get("output") or "").strip()

        if tokenizer is not None and instruction and output:
            user_content = f"{instruction}\n\n{user_input}".strip() if user_input else instruction
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            try:
                full = list(tokenizer.apply_chat_template(messages, tokenize=True))
                prefix = list(tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True))
                # the template's trailing generation prompt is the same suffix as `prefix` ends with
                tail = self._assistant_header_len(tokenizer, prefix)
                if tail and full[-tail:] == prefix[-tail:]:
                    full = full[:-tail]
                plen = len(prefix) if full[:len(prefix)] == prefix else 0
                return full[: self.max_seq_len], min(plen, self.max_seq_len)
            except Exception:
                pass

        text = sample.get("formatted_text", "")
        if not text:
            text = " ".join(p for p in (instruction, user_input, output) if p)
        return self._simple_tokenize(text), 0

    _hdr_len = None

    def _assistant_header_len(self, tokenizer, prefix: List[int]) -> int:
        """Token count of `<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`."""
        if self._hdr_len is None:
            try:
                hdr = tokenizer.encode('<|open|>message role="assistant"<|sep|><|open|>think<|sep|>')
                self._hdr_len = len(hdr) if prefix[-len(hdr):] == list(hdr) else 0
            except Exception:
                self._hdr_len = 0
        return self._hdr_len

    def _simple_tokenize(self, text: str) -> List[int]:
        """Tokenize plain text with the model's own tokenizer, or fall back to raw bytes."""
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            ids = tokenizer.encode(text)[: self.max_seq_len - 2]
            return [self.bos_token_id] + list(ids) + [self.eos_token_id]

        tokens = [self.bos_token_id]
        byte_tokens = [int(b) + 100 for b in text.encode("utf-8")[: self.max_seq_len - 2]]
        tokens.extend(byte_tokens)
        tokens.append(self.eos_token_id)
        return tokens

    def _packed_batches(self, as_torch, device, skip_samples, mask_prompt):
        """Concatenate samples into max_seq_len sequences; targets outside the assistant
        tokens are the pad id when mask_prompt is set. One sequence per micro-batch."""
        seq: List[int] = []
        train: List[bool] = []
        n_in_seq = 0

        def emit():
            nonlocal seq, train, n_in_seq
            if len(seq) < 2:
                seq, train, n_in_seq = [], [], 0
                return None
            arr = np.array(seq[: self.max_seq_len + 1], dtype=np.int64)
            tr = np.array(train[: self.max_seq_len + 1], dtype=bool)
            input_ids = arr[None, :-1]
            target_ids = arr[None, 1:].copy()
            if mask_prompt:
                target_ids[0, ~tr[1:]] = self.pad_token_id
            seq, train, n_in_seq = [], [], 0
            if as_torch and HAS_TORCH:
                t_in, t_t = torch.from_numpy(input_ids), torch.from_numpy(target_ids)
                if device != "cpu" and torch.cuda.is_available():
                    t_in, t_t = t_in.to(device), t_t.to(device)
                return t_in, t_t
            return input_ids, target_ids

        for sample in self.iterate_samples():
            if self.samples_consumed < skip_samples:
                self.samples_consumed += 1
                continue
            ids, plen = self.tokenize_with_prefix(sample)
            ids = ids[: self.max_seq_len]
            if len(seq) + len(ids) > self.max_seq_len + 1 and seq:
                out = emit()
                if out is not None:
                    yield out
            seq.extend(ids)
            train.extend([False] * min(plen, len(ids)) + [True] * max(0, len(ids) - plen))
            n_in_seq += 1
            self.samples_consumed += 1
        if seq:
            out = emit()
            if out is not None:
                yield out

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
        skip_samples: int = 0,
        pack: bool = False,
        mask_prompt: bool = False,
    ) -> Iterator[Tuple[Union["torch.Tensor", np.ndarray], Union["torch.Tensor", np.ndarray]]]:
        """
        Yields (input_ids, target_ids) micro-batches.

        `skip_samples` skips that many leading samples, so a run resumed from a checkpoint
        continues with the data it had not seen. `self.samples_consumed` counts the samples
        that went into the batches yielded so far (including skipped ones).

        `pack=True` concatenates consecutive samples into one sequence of up to max_seq_len
        tokens (the engine's cost is per sweep, not per token, so a 1024-token step should
        carry ~3 samples rather than one padded one). `mask_prompt=True` sets the targets of
        prompt positions to the pad id, so only the assistant's tokens are trained on.
        """
        current_batch_tokens = []
        self.samples_consumed = 0

        if pack:
            yield from self._packed_batches(as_torch, device, skip_samples, mask_prompt)
            return

        for sample in self.iterate_samples():
            if self.samples_consumed < skip_samples:
                self.samples_consumed += 1
                continue
            self.samples_consumed += 1
            token_ids, plen = self.tokenize_with_prefix(sample)
            
            # Truncate / Pad to max_seq_len
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            else:
                token_ids = token_ids + [self.pad_token_id] * (self.max_seq_len - len(token_ids))

            current_batch_tokens.append(token_ids)

            if len(current_batch_tokens) == batch_size:
                arr = np.array(current_batch_tokens, dtype=np.int64)
                input_ids = arr[:, :-1]
                target_ids = arr[:, 1:].copy()
                if mask_prompt and plen > 1:
                    target_ids[:, : plen - 1] = self.pad_token_id

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
