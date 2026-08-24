# 🚀 LazyLoRA: Out-of-Core MoE LoRA Fine-Tuning Engine for Kimi K3

**LazyLoRA** is an ultra-low-memory out-of-core fine-tuning engine engineered to train LoRA adapters on massive Mixture-of-Experts (MoE) models—specifically **Moonshot AI's Kimi K3** (2.78-trillion parameters, 93 layers, 896 fine-grained experts + 2 shared experts, ~1.56 TB safetensors weights)—on consumer-grade hardware.

---

## 🖥️ Target Hardware & Environment Profile
- **GPU**: NVIDIA GeForce GTX 980 Ti (6 GB VRAM, Maxwell GM200, CC 5.2, ~2816 CUDA Cores)
- **CPU**: AMD Ryzen 5 3600 (6 Cores / 12 Threads)
- **Host RAM**: 16 GB Physical (~7.7 GB allocated in WSL2 + Swap)
- **Mass Storage**: D: Drive NVMe/SSD (1.86 TB total, ~1.2 TB free space)
- **C: Drive Isolation Guard**: Strict **ZERO-WRITE POLICY** on C: drive (only ~8.5 GB free space). All caches, temporary files, activations, checkpoints, and datasets are strictly routed to `/mnt/d/hamza/LazyLora_Workspace/`.

---

## ⚡ Core Mathematical Optimizations & Architecture

### 1. Out-of-Core LoRA Parameter Footprint
In standard LoRA ($W = W_0 + \frac{\alpha}{r} B A$), the frozen base weights $W_0$ require zero gradient storage and zero optimizer states.
- Trainable matrices $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$ with rank $r=16$ occupy only **$\sim 228\text{ KB}$** per projection.
- Total LoRA parameters across all 93 layers occupy **$< 250\text{ MB}$**, permanently fitting in GPU VRAM alongside the optimizer states!

### 2. Expert-Wise Dynamic Streaming & Pipelining (ES-MoE Pattern)
- For every token in layer $l$, the router selects top-16 out of 896 experts.
- **Selective Loading**: The $(896 - 16) = 880$ inactive experts are **never read from disk or loaded into RAM/VRAM**.
- **Double Buffering / Asynchronous Prefetch**: While GPU computes GEMM for layer $l$, a background worker pre-fetches layer $l+1$'s active expert tensors from NVMe storage into pinned host RAM / GPU memory via CUDA streams.

### 3. Disk-Backed Activation Ring Buffer on D: Drive
- Deep 93-layer backpropagation without OOM: Layer boundary hidden states $h_l$ ($l = 0 \dots 92$) are streamed to a memory-mapped binary ring buffer on D: drive (`/mnt/d/hamza/LazyLora_Workspace/activations/`).
- Forward pass runs $l = 0 \to 92$, caching $h_l$.
- Backward pass runs $l = 92 \to 0$, reading $h_l$, evaluating analytical LoRA gradients $\nabla_A L, \nabla_B L$, and propagating loss gradients downwards with bounded memory footprint.

---

## 📂 Project Architecture

```
LazyLora/
├── Gorev.txt                              # User task requirement
├── PlanVeGorev.txt                        # System requirements & paper references
├── lazy_lora/                             # Core Python/CUDA engine
│   ├── core/
│   │   ├── config.py                      # Training & hardware hyperparameters
│   │   ├── lora_layer.py                  # LoRA linear adapter module
│   │   ├── moe_router.py                  # Kimi K3 MoE top-16 router & gating
│   │   └── situ_activation.py             # SiTU & SiTU-GLU activation
│   ├── streaming/
│   │   ├── mmap_loader.py                 # Fast mmap safetensors shard reader
│   │   ├── expert_streamer.py             # Dynamic expert-wise streaming & prefetch
│   │   ├── trunk_streamer.py              # Layer-wise dense trunk streamer
│   │   └── activation_ring_buffer.py      # D: drive activation ring buffer
│   ├── trainer/
│   │   ├── lazy_trainer.py                # Sequential layer-wise forward & backward engine
│   │   ├── optimizer.py                   # Low-memory LoRA AdamW optimizer
│   │   └── loss.py                        # CrossEntropyLoss with label smoothing
│   ├── dataset/
│   │   ├── turkish_dataset.py             # Turkish instruction & translation corpus processor
│   │   └── stream_dataset.py              # Zero-RAM disk-streaming dataset iterator
│   ├── monitor/
│   │   ├── dashboard.py                   # Rich visual live terminal UI / monitor
│   │   └── metrics.py                     # VRAM, RAM, disk I/O, ETA stats collector
│   ├── profiler/
│   │   └── hardware.py                    # Hardware analyzer (CPU, GPU, RAM, Disks)
│   └── tests/
│       ├── test_hardware_profiler.py      # Test system specs & disk safety
│       ├── test_moe_routing.py            # Test K3 router invariants & top-16 math
│       ├── test_lora_gradient.py          # Test LoRA forward/backward gradient math
│       ├── test_streaming_loader.py       # Test mmap & expert ring buffer
│       └── test_synthetic_lazy_train.py   # Full synthetic mock training step test
├── scripts/
│   ├── run_profile.sh                     # Hardware audit launcher
│   ├── run_mock_tests.sh                  # Pre-training verification test runner
│   └── train_lazy_lora.sh                 # Production training runner
└── README.md
```

---

## 🛠️ Usage & Verification

### 1. Run Hardware & Storage Safety Audit
```bash
bash scripts/run_profile.sh
```

### 2. Run Comprehensive Pre-Training Test Suite
Before starting the real training, execute the mock test suite to verify math, routing, gradients, disk ring buffers, and memory bounds:
```bash
bash scripts/run_mock_tests.sh
```

### 3. Production Training (When Download Completes)
Once all 96 shards of Kimi K3 finish downloading to `/mnt/d/hamza/kimi_k3_model_weights`, launch training:
```bash
bash scripts/train_lazy_lora.sh
```
