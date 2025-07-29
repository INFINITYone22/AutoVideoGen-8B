import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class T2VConfig:
    # Video specifications
    video_length_sec: int = 10
    fps: int = 30
    total_frames: int = 300  # video_length_sec * fps
    resolution: int = 512
    latent_resolution: int = 64  # 8x compression
    patch_size: int = 4
    tokens_per_frame: int = 256  # (64//4)**2
    total_tokens: int = 76800  # 300 * 256
    
    # Advanced model architecture
    embed_dim: int = 4096
    num_layers: int = 40
    num_heads: int = 32
    head_dim: int = 128  # embed_dim // num_heads
    
    # SwiGLU FFN (more efficient than standard FFN)
    ffn_hidden_dim: int = 11008  # ~2.7x embed_dim (SwiGLU optimal ratio)
    use_swiglu: bool = True
    use_gelu_activation: bool = True  # For non-SwiGLU components
    
    # Advanced attention mechanisms
    use_rope: bool = True  # Rotary Position Embedding
    rope_theta: float = 10000.0
    attention_dropout: float = 0.1
    use_flash_attention: bool = True
    use_grouped_query_attention: bool = True
    num_kv_heads: int = 8  # For GQA
    
    # Memory optimization
    temporal_context_frames: int = 16
    use_gradient_checkpointing: bool = True
    use_activation_checkpointing: bool = True
    max_memory_gb: float = 80.0
    context_window: int = 8192
    
    # VQ-VAE improvements
    vocab_size: int = 16384
    codebook_beta: float = 0.25  # Commitment loss weight
    use_ema_codebook: bool = True
    ema_decay: float = 0.99
    codebook_l2_norm: bool = True
    
    # Tokenization
    text_vocab_size: int = 50257
    max_text_length: int = 77
    
    # Training optimization
    batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4  # Slightly higher for better convergence
    min_learning_rate: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    
    # Advanced training techniques
    use_mixed_precision: bool = True
    precision: str = "bf16"  # Better than fp16 for stability
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Loss configuration
    reconstruction_weight: float = 1.0
    commitment_weight: float = 0.25
    perceptual_weight: float = 0.1
    temporal_consistency_weight: float = 0.5
    
    # Sampling parameters
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    typical_p: float = 0.95
    use_mirostat: bool = False
    
    # Special tokens
    bos_token_id: int = 16384
    eos_token_id: int = 16385
    pad_token_id: int = 16386
    mask_token_id: int = 16387
    total_vocab_size: int = 67000  # vocab_size + text_vocab_size + special tokens
    
    def __post_init__(self):
        assert self.embed_dim % self.num_heads == 0
        if self.use_grouped_query_attention:
            assert self.num_heads % self.num_kv_heads == 0 