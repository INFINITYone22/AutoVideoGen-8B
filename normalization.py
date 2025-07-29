import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from config import T2VConfig

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - more stable than LayerNorm"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

class SwiGLU(nn.Module):
    """SwiGLU activation function - more efficient than GELU for large models"""
    
    def __init__(self, config: T2VConfig):
        super().__init__()
        self.hidden_dim = config.ffn_hidden_dim
        
        # SwiGLU requires 3 linear layers instead of 2
        self.gate_proj = nn.Linear(config.embed_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.embed_dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, config.embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(Wx) ⊙ Vx where ⊙ is element-wise multiplication
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class GELU(nn.Module):
    """High-precision GELU implementation"""
    
    def __init__(self, use_approx: bool = False):
        super().__init__()
        self.use_approx = use_approx
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_approx:
            # Faster approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            ))
        else:
            # Exact implementation
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0))) 