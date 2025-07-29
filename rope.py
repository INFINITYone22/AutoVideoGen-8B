import torch
import torch.nn as nn
from typing import Tuple

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding - superior to absolute positional encoding"""
    
    def __init__(self, dim: int, max_seq_len: int = 100000, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, 
                           seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_cos_sin(seq_len, q.device, q.dtype)
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed 