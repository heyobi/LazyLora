"""
Configuration definitions for LazyLoRA Engine:
- Model Architecture Config (Kimi K3 & MoE specifications)
- LoRA Hyperparameters
- Out-of-Core Streaming & Buffer Allocation
- Path Management (Strict D: drive enforcement)
- Training & Optimization parameters
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KimiK3ArchitectureConfig:
    """Exact architectural parameters of Kimi K3."""
    num_hidden_layers: int = 93
    hidden_size: int = 7168
    intermediate_size: int = 33792
    moe_intermediate_size: int = 3072
    routed_expert_hidden_size: int = 3584
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    vocab_size: int = 163840
    pad_token_id: int = 163839
    bos_token_id: int = 163584
    eos_token_id: int = 163586
    rms_norm_eps: float = 1e-5
    first_k_dense_replace: int = 1
    routed_scaling_factor: float = 1.0
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    activation_func: str = "situ"
    dtype: str = "bfloat16"


@dataclass
class LoRAConfig:
    """LoRA Low-Rank Adaptation configuration."""
    r: int = 16                           # Low rank dimension
    lora_alpha: int = 32                  # Scaling factor
    lora_dropout: float = 0.05            # Dropout rate
    target_modules: List[str] = field(    # Target matrices for LoRA insertion
        default_factory=lambda: [
            "gate_proj",                  # MoE Expert Gate
            "up_proj",                    # MoE Expert Up
            "down_proj",                  # MoE Expert Down
            "q_proj",                     # Attention Query
            "v_proj",                     # Attention Value
        ]
    )
    bias: str = "none"                    # No bias training
    init_lora_weights: bool = True        # Kaiming uniform for A, zero for B


@dataclass
class StreamingConfig:
    """Out-of-Core I/O and Memory Buffer Management."""
    device: str = "cuda:0"                # Target device (cuda:0 or cpu)
    max_vram_mb: float = 4608.0           # Strict VRAM cap (under 6GB physical)
    max_ram_gb: float = 6.0               # Strict RAM cap (under 16GB physical)
    async_prefetch: bool = True           # Overlap I/O with compute
    ring_buffer_depth: int = 2            # Double-buffering for expert streaming
    mmap_mode: bool = True                # Memory-mapped shard reading
    use_pinned_memory: bool = True        # Pinned host RAM for zero-latency PCIe DMA
    cuda_streams: int = 2                 # Number of concurrent CUDA streams


@dataclass
class PathConfig:
    """Storage & Cache Path Configuration. All scratch and caches routed to D: drive."""
    base_model_dir: str = "/mnt/d/hamza/kimi_k3_model_weights"
    workspace_dir: str = "/mnt/d/hamza/LazyLora_Workspace"
    activation_cache_dir: str = "/mnt/d/hamza/LazyLora_Workspace/activations"
    checkpoints_dir: str = "/mnt/d/hamza/LazyLora_Workspace/checkpoints"
    dataset_dir: str = "/mnt/d/hamza/LazyLora_Workspace/datasets"
    cache_dir: str = "/mnt/d/hamza/LazyLora_Workspace/cache"

    def ensure_directories(self) -> None:
        """Create all required scratch and cache directories on D: drive."""
        for path in [
            self.workspace_dir,
            self.activation_cache_dir,
            self.checkpoints_dir,
            self.dataset_dir,
            self.cache_dir,
        ]:
            os.makedirs(path, exist_ok=True)


@dataclass
class TrainingConfig:
    """Hyperparameters for Turkish Translation Fine-Tuning."""
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 512
    warmup_steps: int = 50
    max_steps: int = 2000
    save_steps: int = 100
    logging_steps: int = 1
    eval_steps: int = 50
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    grad_clip_norm: float = 1.0


@dataclass
class LazyLoraConfig:
    """Root configuration aggregating all sub-configs."""
    model: KimiK3ArchitectureConfig = field(default_factory=KimiK3ArchitectureConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> LazyLoraConfig:
    """Returns initialized and validated default configuration."""
    cfg = LazyLoraConfig()
    return cfg
