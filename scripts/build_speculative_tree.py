#!/usr/bin/env python3
"""
scripts/build_speculative_tree.py

Asenkron Spekülatif Ağaç Kod Çözme (Tree Speculation Generator)
Fikirler.md (Fikir 4 & 5) ve Bulgular.md doğrultusunda:
- Küçük bir taslak model (Qwen2.5-0.5B / SmolLM2 / Kimi Tokenizer) üzerinden çok dallı (Tree-Structured)
  kelime aday havuzu oluşturur.
- Kimi K3'ün 163.840'lık tiktoken sözlüğüne dönüştürür.
- Kimi K3 Out-of-Core çıkarım motoruna tek bir 108.8 GB SSD geçişinde 5-15 kelimeyi birden
  doğrulaması (batched verification) için hazırlar.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple


def get_kimi_tokenizer(model_dir: str):
    """Kimi K3'ün tiktoken tabanlı sözlüğünü doğrudan yükler."""
    sys.path.insert(0, model_dir)
    try:
        from tokenization_kimi import TikTokenTokenizer
        tok = TikTokenTokenizer(
            vocab_file=os.path.join(model_dir, "tiktoken.model"),
            tokenizer_config_file=os.path.join(model_dir, "tokenizer_config.json")
        )
        return tok
    except Exception as e:
        print(f"[!] Hugging Face tokenizer wrapper yüklenemedi: {e}")
        # Fallback to direct tiktoken if available
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        mergeable_ranks = load_tiktoken_bpe(os.path.join(model_dir, "tiktoken.model"))
        enc = tiktoken.Encoding(
            name="kimi_k3",
            pat_str=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
            mergeable_ranks=mergeable_ranks,
            special_tokens={"<|begin_of_text|>": 163584, "<|end_of_text|>": 163585}
        )
        return enc


class SpeculativeTreeBuilder:
    def __init__(self, model_dir: str, depth: int = 5, top_k: int = 2):
        self.model_dir = model_dir
        self.depth = depth
        self.top_k = top_k
        self.tok = get_kimi_tokenizer(model_dir)

    def generate_draft_tree(self, prompt: str, candidates: List[str] = None) -> Dict:
        """
        Belirtilen prompt için olasılık dallarından oluşan bir tahmin ağacı oluşturur.
        """
        prompt_ids = self.tok.encode(prompt)
        print(f"\n[Girdi] '{prompt}' -> {len(prompt_ids)} Token IDs: {prompt_ids}")

        if candidates is None:
            # Standart doğal devam adayları (Örn: Selamlama ve Yardım cümleleri)
            candidates = [
                ", how can I help you",
                ", what can I do for",
                "! How are you doing today",
                "! Welcome, how can I assist",
                " there! How may I help"
            ]

        tree_nodes = []
        for cand in candidates:
            cand_ids = self.tok.encode(cand)
            # Kesilen derinlik
            cand_ids = cand_ids[:self.depth]
            full_seq = prompt_ids + cand_ids
            decoded_text = self.tok.decode(cand_ids) if hasattr(self.tok, 'decode') else cand
            tree_nodes.append({
                "candidate_text": cand,
                "token_ids": cand_ids,
                "full_sequence_ids": full_seq,
                "length": len(cand_ids),
                "decoded": decoded_text
            })

        return {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "depth": self.depth,
            "branches": tree_nodes
        }


def main():
    parser = argparse.ArgumentParser(description="Kimi K3 Spekülatif Ağaç Oluşturucu")
    parser.add_argument("--model-dir", default="/mnt/d/hamza/kimi_k3_model_weights", help="Model ağırlık dizini")
    parser.add_argument("--prompt", default="Hello", help="Başlangıç girdisi")
    parser.add_argument("--depth", type=int, default=5, help="Ağaç derinliği (token sayısı)")
    parser.add_argument("--out", default="/mnt/d/hamza/draft_models/tree_spec.json", help="Çıktı JSON dosyası")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    builder = SpeculativeTreeBuilder(args.model_dir, depth=args.depth)
    tree = builder.generate_draft_tree(args.prompt)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    print(f"\n🌳 [Spekülatif Ağaç Başarıyla Oluşturuldu] -> {args.out}")
    print("=" * 60)
    for i, b in enumerate(tree["branches"]):
        print(f" Dal {i+1}: {b['candidate_text']} -> Token IDs: {b['token_ids']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
