import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from config import T2VConfig
from normalization import RMSNorm, SwiGLU, GELU
from rope import RotaryEmbedding
from vqvae import AdvancedVQVAEEncoder, AdvancedVQVAEDecoder

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention - more efficient than standard MHA"""
    
    def __init__(self, config: T2VConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Grouped projections
        self.q_proj = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # RoPE
        if config.use_rope:
            self.rope = RotaryEmbedding(self.head_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                use_cache: bool = False, past_key_value: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key-values for caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Apply RoPE if enabled
        if hasattr(self, 'rope'):
            q, k = self.rope.apply_rotary_pos_emb(q, k, k.size(2))
        
        # Expand K,V for grouped attention
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -float('inf'))
        
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        # Return cache if requested
        present_key_value = (k, v) if use_cache else None
        
        return output, present_key_value

class EnhancedTransformerBlock(nn.Module):
    """Advanced transformer block with modern optimizations"""
    
    def __init__(self, config: T2VConfig):
        super().__init__()
        self.config = config
        
        # Pre-normalization with RMSNorm
        self.input_layernorm = RMSNorm(config.embed_dim)
        self.post_attention_layernorm = RMSNorm(config.embed_dim)
        
        # Attention
        self.self_attn = GroupedQueryAttention(config)
        
        # Feed-forward network
        if config.use_swiglu:
            self.mlp = SwiGLU(config)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.embed_dim, config.ffn_hidden_dim, bias=False),
                GELU(use_approx=config.use_gelu_activation),
                nn.Linear(config.ffn_hidden_dim, config.embed_dim, bias=False)
            )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False, past_key_value: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        
        # Pre-norm + self-attention + residual
        residual = x
        x = self.input_layernorm(x)
        attn_output, present_key_value = self.self_attn(
            x, attention_mask=attention_mask, use_cache=use_cache, past_key_value=past_key_value
        )
        x = residual + attn_output
        
        # Pre-norm + FFN + residual  
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        
        return x, present_key_value

class EnhancedAutoregressiveT2V(nn.Module):
    """Production-ready autoregressive T2V model with all optimizations"""
    
    def __init__(self, config: T2VConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings with proper initialization
        self.embed_tokens = nn.Embedding(config.total_vocab_size, config.embed_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final norm and output projection
        self.norm = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Enhanced VQ-VAE
        self.vq_encoder = AdvancedVQVAEEncoder(config)
        self.vq_decoder = AdvancedVQVAEDecoder(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Gradient checkpointing
        if config.use_gradient_checkpointing:
            self.gradient_checkpointing_enable()
    
    def _init_weights(self, module):
        """Initialize weights using best practices"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask with sliding window"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # Apply sliding window
        if self.config.temporal_context_frames > 0:
            context_tokens = self.config.temporal_context_frames * self.config.tokens_per_frame
            for i in range(seq_len):
                start_idx = max(0, i - context_tokens + 1)
                mask[i, :start_idx] = 0
        
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False, past_key_values: Optional[List[Tuple]] = None) -> Dict[str, Any]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self.create_causal_mask(seq_len, device)
        
        # Initialize cache
        present_key_values = [] if use_cache else None
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.config.use_gradient_checkpointing and self.training:
                hidden_states, present_key_value = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, use_cache, past_key_value
                )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states, attention_mask=attention_mask, 
                    use_cache=use_cache, past_key_value=past_key_value
                )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final normalization and projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'past_key_values': present_key_values,
            'hidden_states': hidden_states
        } 