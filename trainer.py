import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import os
from typing import Dict, Optional
from config import T2VConfig
from transformer import EnhancedAutoregressiveT2V

class AdvancedTrainer:
    """Production-ready training loop with all optimizations"""
    
    def __init__(self, model: EnhancedAutoregressiveT2V, config: T2VConfig):
        self.model = model
        self.config = config
        
        # Setup distributed training
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if self.world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            self.model = DDP(model.cuda(), device_ids=[self.local_rank])
        
        # Optimizer with proper settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.max_steps,
            eta_min=config.min_learning_rate
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Metrics tracking
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def calculate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate comprehensive loss with multiple components"""
        
        text_tokens = batch['text_tokens']  # [batch, max_text_length]
        video_frames = batch['video_frames']  # [batch, num_frames, 3, H, W]
        
        batch_size, num_frames = video_frames.shape[:2]
        total_losses = {}
        
        # Encode video frames to tokens
        video_tokens_list = []
        vq_losses_list = []
        
        for frame_idx in range(num_frames):
            frame = video_frames[:, frame_idx]
            tokens, vq_losses = self.model.vq_encoder(frame)
            video_tokens_list.append(tokens)
            vq_losses_list.append(vq_losses)
        
        # Stack video tokens: [batch, num_frames * tokens_per_frame]
        video_tokens = torch.stack(video_tokens_list, dim=1)
        video_tokens = video_tokens.view(batch_size, -1)
        
        # Create autoregressive training sequence
        input_sequence = torch.cat([
            text_tokens,
            torch.full((batch_size, 1), self.config.bos_token_id, device=text_tokens.device),
            video_tokens[:, :-1]  # Shifted by 1 for autoregressive training
        ], dim=1)
        
        target_sequence = video_tokens
        
        # Forward pass through transformer
        with autocast(enabled=self.config.use_mixed_precision, dtype=torch.bfloat16):
            outputs = self.model(input_sequence)
            logits = outputs['logits']
            
            # Extract logits for video prediction (skip text portion)
            video_start_idx = text_tokens.size(1) + 1
            video_logits = logits[:, video_start_idx:, :]
            
            # Language modeling loss
            lm_loss = F.cross_entropy(
                video_logits.reshape(-1, video_logits.size(-1)),
                target_sequence.view(-1),
                ignore_index=self.config.pad_token_id
            )
            
            # VQ-VAE losses (averaged across frames)
            vq_loss = torch.stack([losses['vq_loss'] for losses in vq_losses_list]).mean()
            commitment_loss = torch.stack([losses['commitment_loss'] for losses in vq_losses_list]).mean()
            
            # Reconstruction loss (sample a few frames for efficiency)
            recon_losses = []
            sample_frames = min(4, num_frames)  # Sample up to 4 frames
            frame_indices = torch.randperm(num_frames)[:sample_frames]
            
            for idx in frame_indices:
                frame_tokens = video_tokens_list[idx]
                target_frame = video_frames[:, idx]
                decoder_output = self.model.vq_decoder(frame_tokens, target_frame)
                recon_losses.extend(list(decoder_output['losses'].values()))
            
            recon_loss = torch.stack(recon_losses).mean() if recon_losses else torch.tensor(0.0)
            
            # Temporal consistency loss (encourage smooth transitions)
            temporal_loss = torch.tensor(0.0, device=logits.device)
            if num_frames > 1:
                for i in range(num_frames - 1):
                    curr_tokens = video_tokens_list[i]
                    next_tokens = video_tokens_list[i + 1]
                    # Encourage similarity between adjacent frames
                    curr_embed = self.model.embed_tokens(curr_tokens)
                    next_embed = self.model.embed_tokens(next_tokens)
                    temporal_loss += F.mse_loss(curr_embed, next_embed)
                temporal_loss /= (num_frames - 1)
            
            # Combined loss
            total_loss = (
                lm_loss + 
                self.config.commitment_weight * vq_loss +
                self.config.reconstruction_weight * recon_loss +
                self.config.temporal_consistency_weight * temporal_loss
            )
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'reconstruction_loss': recon_loss,
            'temporal_loss': temporal_loss
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient accumulation"""
        
        self.model.train()
        
        # Calculate losses
        losses = self.calculate_loss(batch)
        total_loss = losses['total_loss'] / self.config.gradient_accumulation_steps
        
        # Backward pass with mixed precision
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                # Unscale gradients and clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.step += 1
        
        # Convert losses to float for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def save_checkpoint(self, path: str, extra_state: Dict = None):
        """Save training checkpoint"""
        
        state = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
            
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, path)
        print(f"Checkpoint saved to {path}")
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop"""
        
        print(f"Starting training for {self.config.max_steps} steps")
        
        while self.step < self.config.max_steps:
            for batch in train_dataloader:
                if self.step >= self.config.max_steps:
                    break
                
                # Move batch to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Training step
                losses = self.train_step(batch)
                
                # Logging
                if self.step % 100 == 0:
                    print(f"Step {self.step}: {losses}")
                
                # Evaluation
                if self.step % self.config.eval_interval == 0 and eval_dataloader is not None:
                    eval_loss = self.evaluate(eval_dataloader)
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint("best_model.pt")
                
                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.step}.pt")
            
            self.epoch += 1
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader) -> float:
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in eval_dataloader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            losses = self.calculate_loss(batch)
            total_loss += losses['total_loss'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Evaluation loss: {avg_loss:.4f}")
        return avg_loss

# Training usage
def main():
    config = T2VConfig()
    model = EnhancedAutoregressiveT2V(config)
    trainer = AdvancedTrainer(model, config)
    
    # Your dataloaders here
    # train_dataloader = create_train_dataloader(config)
    # eval_dataloader = create_eval_dataloader(config)
    
    # trainer.train(train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main() 