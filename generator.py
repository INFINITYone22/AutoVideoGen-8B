import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List
from torch.cuda.amp import autocast
from config import T2VConfig
from transformer import EnhancedAutoregressiveT2V
from sampler import AdvancedSampler

class EnhancedVideoGenerator:
    """Production-ready video generation with advanced sampling"""
    
    def __init__(self, model: EnhancedAutoregressiveT2V, config: T2VConfig):
        self.model = model
        self.config = config
        self.sampler = AdvancedSampler()
        
    @torch.no_grad()
    def generate_video(self, 
                      text_prompt: str,
                      max_new_tokens: Optional[int] = None,
                      temperature: float = 0.8,
                      top_k: int = 50,
                      top_p: float = 0.9,
                      typical_p: float = 0.95,
                      use_typical_sampling: bool = True,
                      num_frames: Optional[int] = None,
                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate video with advanced sampling strategies
        
        Args:
            text_prompt: Text description of the video
            max_new_tokens: Maximum tokens to generate (default: full video)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter  
            typical_p: Typical sampling parameter
            use_typical_sampling: Whether to use typical sampling
            num_frames: Number of frames to generate
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing generated video and metadata
        """
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Tokenize text prompt (simplified - use proper tokenizer)
        text_tokens = self._encode_text(text_prompt)
        text_tokens = torch.tensor([text_tokens], device=device)
        
        # Initialize sequence
        input_ids = torch.cat([
            text_tokens,
            torch.tensor([[self.config.bos_token_id]], device=device)
        ], dim=1)
        
        # Determine generation length
        if max_new_tokens is None:
            if num_frames is not None:
                max_new_tokens = num_frames * self.config.tokens_per_frame
            else:
                max_new_tokens = self.config.total_tokens
        
        # Generation loop with caching
        past_key_values = None
        generated_frames = []
        frame_tokens_buffer = []
        
        progress_callback = lambda step, total: print(f"Generating token {step}/{total}")
        
        for step in range(max_new_tokens):
            # Manage memory with sliding window
            current_seq_len = input_ids.size(1)
            if current_seq_len > self.config.context_window + self.config.max_text_length:
                # Truncate sequence but keep text context
                keep_text = input_ids[:, :self.config.max_text_length]
                keep_video = input_ids[:, -self.config.context_window:]
                input_ids = torch.cat([keep_text, keep_video], dim=1)
                past_key_values = None  # Reset cache after truncation
            
            # Forward pass
            with autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(
                    input_ids if past_key_values is None else input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values
                )
            
            logits = outputs['logits'][:, -1, :]  # Last token predictions
            past_key_values = outputs['past_key_values']
            
            # Apply sampling strategies
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                logits = self.sampler.top_k_sampling(logits, top_k)
            
            # Apply typical sampling or nucleus sampling
            if use_typical_sampling and typical_p < 1.0:
                logits = self.sampler.typical_sampling(logits, typical_p)
            elif top_p < 1.0:
                logits = self.sampler.top_p_sampling(logits, top_p)
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Collect frame tokens
            frame_tokens_buffer.append(next_token.item())
            
            # Check if we have a complete frame
            if len(frame_tokens_buffer) == self.config.tokens_per_frame:
                # Decode frame
                frame_tokens = torch.tensor([frame_tokens_buffer], device=device)
                decoder_output = self.model.vq_decoder(frame_tokens)
                generated_frames.append(decoder_output['reconstructed'])
                frame_tokens_buffer = []
            
            # Progress callback
            if step % 100 == 0:
                progress_callback(step, max_new_tokens)
        
        # Stack all frames
        if generated_frames:
            video_tensor = torch.stack(generated_frames, dim=1)  # [1, num_frames, 3, H, W]
            video_tensor = video_tensor.squeeze(0)  # [num_frames, 3, H, W]
        else:
            video_tensor = torch.empty(0, 3, self.config.resolution, self.config.resolution)
        
        return {
            'video': video_tensor,
            'num_frames': len(generated_frames),
            'resolution': self.config.resolution,
            'fps': self.config.fps,
            'text_prompt': text_prompt,
            'generation_params': {
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'typical_p': typical_p,
                'use_typical_sampling': use_typical_sampling
            }
        }
    
    def _encode_text(self, text: str) -> List[int]:
        """Enhanced text encoding - integrate with proper tokenizer"""
        # This is a placeholder - integrate with actual tokenizer
        # For example: transformers.AutoTokenizer
        
        words = text.lower().split()
        # Simple hash-based encoding (replace with real tokenizer)
        tokens = [hash(word) % self.config.text_vocab_size for word in words]
        
        # Pad/truncate to max length
        if len(tokens) > self.config.max_text_length:
            tokens = tokens[:self.config.max_text_length]
        else:
            tokens.extend([self.config.pad_token_id] * (self.config.max_text_length - len(tokens)))
        
        return tokens
    
    def save_video(self, video_tensor: torch.Tensor, output_path: str, fps: int = 30):
        """Save video tensor to file"""
        try:
            import imageio
            
            # Convert tensor to numpy: [3, T, H, W] -> [T, H, W, 3]
            video_np = video_tensor.permute(1, 2, 3, 0).numpy()
            
            # Normalize to [0, 255]
            video_np = ((video_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Save as MP4
            with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
                for frame in video_np:
                    writer.append_data(frame)
            
            print(f"Video saved to {output_path}")
            
        except ImportError:
            print("imageio not available, saving as numpy array")
            np.save(output_path.replace('.mp4', '.npy'), video_tensor.numpy())

# Usage example
def generate_sample_video():
    """Example usage of the enhanced video generator"""
    
    config = T2VConfig()
    model = EnhancedAutoregressiveT2V(config)
    
    # Load trained weights
    # checkpoint = torch.load("trained_model.pt", map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    generator = EnhancedVideoGenerator(model, config)
    
    # Generate video
    result = generator.generate_video(
        text_prompt="A robot dancing in a park with colorful flowers",
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        use_typical_sampling=True,
        typical_p=0.95,
        num_frames=120,  # 4 seconds at 30 FPS
        seed=42
    )
    
    # Save video
    generator.save_video(
        result['video'], 
        "generated_robot_dance.mp4", 
        fps=config.fps
    )
    
    print(f"Generated {result['num_frames']} frames")
    print(f"Video resolution: {result['resolution']}x{result['resolution']}")

if __name__ == "__main__":
    generate_sample_video() 