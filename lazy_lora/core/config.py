"""
Configuration definitions for LazyLoRA Engine:
- Model Architecture Config (Kimi K3 & MoE specifications)
- LoRA Hyperparameters
- Out-of-Core Streaming & Buffer Allocation
- Path Management (environment-driven, see default_*_dir)
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
    # Kimi Linear interleaves two attention types. `full_attn_layers` is stored exactly as
    # config.json lists it (1-based); everything else is a KDA linear-attention layer.
    full_attn_layers: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
                                 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93]
    )
    # KDA uses the same head geometry as the rest of the model
    # (linear_attn_config: num_heads 96, head_dim 128).
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0
    mla_use_output_gate: bool = True
    # Cross-layer block residuals: every attn_res_block_size layers the residual stream is
    # pushed onto a bank and restarted, and the bank is mixed back in via a learned softmax.
    attn_res_block_size: int = 12
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    activation_func: str = "situ"
    dtype: str = "bfloat16"

    def is_kda_layer(self, layer_idx: int) -> bool:
        """True when layer_idx uses KDA linear attention rather than full MLA."""
        return (layer_idx + 1) not in set(self.full_attn_layers)


@dataclass
class LoRAConfig:
    """LoRA Low-Rank Adaptation configuration."""
    r: int = 16                           # Low rank dimension
    lora_alpha: int = 32                  # Scaling factor
    lora_dropout: float = 0.0             # Dropout: off. The backward recomputes the layer, and a
                                          # fresh random mask there would not match the forward's.
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
    device: str = "cpu"                   # Target device (cpu or cuda:0)
    max_vram_mb: float = 4608.0           # Strict VRAM cap (under 6GB physical)
    max_ram_gb: float = 4.5               # Strict RAM cap (7.6 GB physical on this machine, no swap wanted)
    async_prefetch: bool = True           # Overlap I/O with compute
    ring_buffer_depth: int = 2            # Double-buffering for expert streaming
    mmap_mode: bool = True                # Memory-mapped shard reading
    use_pinned_memory: bool = True        # Pinned host RAM for zero-latency PCIe DMA
    cuda_streams: int = 2                 # Number of concurrent CUDA streams


def _env_path(name: str, default: str) -> str:
    """A path from the environment when set, else the default for this machine."""
    return os.path.abspath(os.path.expanduser(os.environ.get(name, default)))


# Defaults for the current machine (native Ubuntu, 5 September 2026 onwards):
#   HDD  /mnt/disk2tb  (NTFS, ntfs3)   checkpoint shards, the long-lived workspace, logs
#   NVMe /mnt/nvme     (ext4)          activation ring buffer and checkpoints (fast scratch)
# Every location can be overridden with an environment variable, so nothing in the code
# base names a drive letter or a WSL mount any more.
DEFAULT_MODEL_DIR = "/mnt/disk2tb/hamza/kimi_k3_model_weights"
DEFAULT_WORKSPACE_DIR = "/mnt/disk2tb/hamza/LazyLora_Workspace"
DEFAULT_FAST_SCRATCH_DIR = "/mnt/nvme/lazylora"
DEFAULT_REFERENCE_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "kimi-k3-in-c"
)


def default_model_dir() -> str:
    return _env_path("LAZYLORA_MODEL_DIR", DEFAULT_MODEL_DIR)


def default_workspace_dir() -> str:
    return _env_path("LAZYLORA_WORKSPACE_DIR", DEFAULT_WORKSPACE_DIR)


def default_fast_scratch_dir() -> str:
    return _env_path("LAZYLORA_FAST_SCRATCH_DIR", DEFAULT_FAST_SCRATCH_DIR)


def default_cache_dir() -> str:
    return _env_path("LAZYLORA_CACHE_DIR", os.path.join(default_workspace_dir(), "cache"))


def default_activation_dir() -> str:
    return _env_path("LAZYLORA_ACTIVATION_DIR", os.path.join(default_fast_scratch_dir(), "activations"))


def default_checkpoints_dir() -> str:
    return _env_path("LAZYLORA_CHECKPOINTS_DIR", os.path.join(default_fast_scratch_dir(), "checkpoints"))


def default_dataset_dir() -> str:
    return _env_path("LAZYLORA_DATASET_DIR", os.path.join(default_workspace_dir(), "datasets"))


def default_reference_fixtures_dir() -> str:
    """The op fixtures shipped with kimi-k3-in-c (sibling checkout of this repository)."""
    return _env_path(
        "LAZYLORA_REF_FIXTURES",
        os.path.join(os.path.normpath(DEFAULT_REFERENCE_REPO), "tests", "fixtures", "ops"),
    )


class MissingTensorError(RuntimeError):
    """A tensor the model needs is not on disk and synthetic substitution is not allowed."""


def synthetic_enabled() -> bool:
    """True when LAZYLORA_ALLOW_SYNTHETIC=1 (mock tests); never raises."""
    return os.environ.get("LAZYLORA_ALLOW_SYNTHETIC", "") == "1"


def synthetic_allowed(what: str) -> bool:
    """
    Gate for every synthetic (random / all-ones) substitute in the engine.

    The costliest mistakes in this project came from loaders that quietly produced random
    weights when a tensor was not found: the code kept running and printed plausible
    numbers that meant nothing (fake attention, random routers, 518 invented experts,
    zero-byte shards that a scan reported as clean). Substitution is therefore an error
    unless LAZYLORA_ALLOW_SYNTHETIC=1 is set, which only the mock test suite does.
    """
    if os.environ.get("LAZYLORA_ALLOW_SYNTHETIC", "") == "1":
        return True
    raise MissingTensorError(
        f"{what} is not on disk. Refusing to substitute a synthetic tensor: the result would "
        f"run but be meaningless. Check the checkpoint (scripts/check_shards.py) and the "
        f"tensor name; set LAZYLORA_ALLOW_SYNTHETIC=1 only for mock tests."
    )


@dataclass
class PathConfig:
    """Storage & cache paths. Defaults come from the environment, see default_*_dir()."""
    base_model_dir: str = field(default_factory=default_model_dir)
    workspace_dir: str = field(default_factory=default_workspace_dir)
    activation_cache_dir: str = field(default_factory=default_activation_dir)
    checkpoints_dir: str = field(default_factory=default_checkpoints_dir)
    dataset_dir: str = field(default_factory=default_dataset_dir)
    cache_dir: str = field(default_factory=default_cache_dir)
    reference_fixtures_dir: str = field(default_factory=default_reference_fixtures_dir)

    def ensure_directories(self) -> None:
        """Create every writable scratch and cache directory."""
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
