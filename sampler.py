import torch
import torch.nn.functional as F
from typing import Optional

class AdvancedSampler:
    """Advanced sampling strategies for high-quality generation"""
    
    @staticmethod
    def top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k sampling"""
        if k <= 0:
            return logits
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Create mask
        mask = torch.full_like(logits, -float('inf'))
        mask.scatter_(-1, top_k_indices, top_k_values)
        
        return mask
    
    @staticmethod
    def top_p_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus (top-p) sampling"""
        if p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff point
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create mask
        mask = torch.full_like(logits, -float('inf'))
        keep_indices = sorted_indices[~sorted_indices_to_remove]
        keep_logits = sorted_logits[~sorted_indices_to_remove]
        mask.scatter_(-1, keep_indices, keep_logits)
        
        return mask
    
    @staticmethod
    def typical_sampling(logits: torch.Tensor, tau: float = 0.95) -> torch.Tensor:
        """Typical sampling based on conditional entropy"""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
        
        # Calculate surprisal for each token
        surprisal = -torch.log(probs + 1e-10)
        
        # Find tokens with surprisal close to entropy
        diff = torch.abs(surprisal - entropy)
        
        # Sort by difference to entropy
        sorted_diffs, sorted_indices = torch.sort(diff)
        
        # Find cumulative probability mass
        sorted_probs = probs.gather(-1, sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff
        cutoff_index = torch.searchsorted(cumulative_probs, tau)
        cutoff_index = torch.clamp(cutoff_index, 1, sorted_indices.size(-1) - 1)
        
        # Create mask
        mask = torch.full_like(logits, -float('inf'))
        for i in range(logits.size(0)):
            keep_indices = sorted_indices[i, :cutoff_index[i]]
            keep_logits = logits[i].gather(0, keep_indices)
            mask[i].scatter_(0, keep_indices, keep_logits)
        
        return mask 