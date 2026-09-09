"""
LazyLoRA: Ultra-Memory-Efficient Out-of-Core MoE LoRA Training Engine.
Designed for training LoRA adapters on massive Mixture-of-Experts models (Kimi K3)
under strict consumer hardware constraints: an i7-7700HQ laptop with 7.6 GB of RAM,
a GTX 1050 with 2 GB, and the 1.56 TB checkpoint on an external USB disk.
"""

__version__ = "0.1.0"
__author__ = "Ibrahim Polat"
