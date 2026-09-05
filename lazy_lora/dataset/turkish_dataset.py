"""
Turkish Translation and Instruction Dataset Generator & Formatter for Kimi K3.
Prepares bilingual instruction pairs, Turkish linguistic adaptation tasks,
and chat template formatting with Kimi K3 special tokens.
"""

import os
import json
from typing import List, Dict, Any, Generator, Optional


# Curated high-fidelity English-Turkish translation & instruction seeds
SEED_TURKISH_DATASET: List[Dict[str, str]] = [
    {
        "instruction": "Aşağıdaki İngilizce metni doğal, akıcı ve terminolojiye uygun bir Türkçe ile çeviriniz.",
        "input": "Mixture-of-Experts (MoE) models scale parameter capacity while keeping per-token computational cost bounded by routing tokens to a subset of specialized experts.",
        "output": "Uzmanlar Karışımı (MoE) modelleri, belirteçleri (token'ları) yalnızca belirli uzmanların bir alt kümesine yönlendirerek, belirteç başına hesaplama maliyetini sabit tutarken parametre kapasitesini ölçeklendirir.",
    },
    {
        "instruction": "Aşağıdaki metni Türkçeye çeviriniz ve teknik terimleri açıklayınız.",
        "input": "Low-Rank Adaptation (LoRA) freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.",
        "output": "Düşük Dereceli Uyarlama (LoRA), önceden eğitilmiş model ağırlıklarını dondurur ve Transformer mimarisinin her katmanına eğitilebilir düşük dereceli ayrıştırma matrisleri enjekte eder.",
    },
    {
        "instruction": "Verilen cümleyi profesyonel bir üslupla Türkçeye çeviriniz.",
        "input": "Out-of-core training enables fine-tuning models that exceed physical GPU memory capacity by streaming parameters directly from high-speed NVMe storage.",
        "output": "Çekirdek-dışı (Out-of-core) eğitim, parametreleri doğrudan yüksek hızlı NVMe depolamadan aktararak fiziksel GPU bellek kapasitesini aşan modellerin ince ayarını mümkün kılar.",
    },
    {
        "instruction": "Aşağıdaki soruyu Türkçe olarak yanıtlayınız.",
        "input": "What is the primary advantage of activation checkpointing in deep neural network training?",
        "output": "Derin yapay sinir ağlarının eğitiminde aktivasyon kontrol noktalarının (activation checkpointing) temel avantajı, tüm ara katman aktivasyonlarını bellekte tutmak yerine geriye yayılım (backprop) sırasında yalnızca gerektiğinde yeniden hesaplayarak VRAM/RAM tüketimini dramatik ölçüde azaltmasıdır.",
    },
    {
        "instruction": "Aşağıdaki teknik paragrafı Türkçeye çeviriniz.",
        "input": "Asynchronous prefetching and double buffering allow I/O operations across the PCIe bus to overlap seamlessly with GPU kernel execution.",
        "output": "Eşzamansız önceden yükleme (asynchronous prefetching) ve çift arabelleğe alma (double buffering), PCIe veri yolu üzerindeki G/Ç işlemlerinin GPU çekirdek hesaplamalarıyla kesintisiz bir şekilde örtüşmesini sağlar.",
    },
]


class TurkishDatasetManager:
    """
    Manages Turkish dataset preparation, synthesis, formatting, and disk caching.
    """

    def __init__(self, dataset_dir: Optional[str] = None):
        from lazy_lora.core.config import default_dataset_dir
        self.dataset_dir = dataset_dir or default_dataset_dir()
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.train_file = os.path.join(self.dataset_dir, "turkish_instruct_train.jsonl")
        self.val_file = os.path.join(self.dataset_dir, "turkish_instruct_val.jsonl")

    @staticmethod
    def format_kimi_prompt(instruction: str, user_input: str = "", response: str = "") -> str:
        """
        Plain-text rendering of one sample.

        Kimi K3 does not use ChatML: its chat format is a tag syntax built from its own
        control tokens (`<|open|>message role="user"<|sep|> ... <|end_of_msg|>`). Writing
        `<|im_start|>` markers here would feed the model literal text it has never seen in
        that arrangement, so the real template is applied at tokenization time through the
        tokenizer's `apply_chat_template`, and this field stays plain text for readability
        and for the byte-level fallback path.
        """
        user_content = f"{instruction}\n\n{user_input}".strip() if user_input else instruction.strip()
        return f"{user_content}\n\n{response}".strip() if response else user_content

    def generate_seed_dataset(self, target_samples: int = 100) -> str:
        """
        Populate train JSONL with seed and augmented bilingual samples.
        """
        samples = []
        # Expand seed samples with linguistic variations
        templates = [
            ("İngilizce metni Türkçeye çeviriniz:", "Bu metin şöyledir: "),
            ("Lütfen bu içeriği akıcı bir Türkçe ile tercüme edin:", ""),
            ("Translate to natural Turkish:", ""),
        ]

        count = 0
        while len(samples) < target_samples:
            for base_item in SEED_TURKISH_DATASET:
                tmpl_inst, prefix = templates[count % len(templates)]
                formatted_item = {
                    "instruction": tmpl_inst if count % 2 == 0 else base_item["instruction"],
                    "input": f"{prefix}{base_item['input']}".strip(),
                    "output": base_item["output"],
                    "formatted_text": self.format_kimi_prompt(
                        base_item["instruction"],
                        base_item["input"],
                        base_item["output"],
                    )
                }
                samples.append(formatted_item)
                count += 1
                if len(samples) >= target_samples:
                    break

        # Save to D: drive
        with open(self.train_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        return self.train_file

    def generate_curated_samples(self, target_samples: int = 100) -> str:
        """Alias for generate_seed_dataset."""
        return self.generate_seed_dataset(target_samples=target_samples)


if __name__ == "__main__":
    mgr = TurkishDatasetManager()
    path = mgr.generate_seed_dataset(50)
    print(f"Generated Turkish instruction dataset at: {path}")
