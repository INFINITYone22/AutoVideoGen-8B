import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from config import T2VConfig
from normalization import GELU

class VectorQuantizer(nn.Module):
    """Advanced Vector Quantizer with EMA updates and improved loss"""
    
    def __init__(self, config: T2VConfig):
        super().__init__()
        self.num_embeddings = config.vocab_size
        self.embedding_dim = 512
        self.beta = config.codebook_beta
        self.use_ema = config.use_ema_codebook
        self.decay = config.ema_decay
        self.eps = 1e-5
        
        # Codebook embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
        if self.use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(self.num_embeddings))
            self.register_buffer('ema_w', torch.Tensor(self.num_embeddings, self.embedding_dim))
            self.ema_w.data.normal_()
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            inputs: [batch, height, width, dim] or [batch, seq_len, dim]
        Returns:
            quantized: quantized tensor
            indices: quantization indices
            losses: dictionary of losses
        """
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find closest encodings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # One-hot encoding
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # EMA update
        if self.training and self.use_ema:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                   (1 - self.decay) * torch.sum(encodings, 0)
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps) * n
            )
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # Calculate losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        if self.use_ema:
            loss = self.beta * e_latent_loss
        else:
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.beta * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        losses = {
            'vq_loss': loss,
            'commitment_loss': e_latent_loss,
            'codebook_usage': (encodings.sum(0) > 0).sum().float() / self.num_embeddings
        }
        
        return quantized, encoding_indices.view(input_shape[:-1]), losses

class AdvancedVQVAEEncoder(nn.Module):
    """Enhanced VQ-VAE encoder with residual connections and attention"""
    
    def __init__(self, config: T2VConfig):
        super().__init__()
        self.config = config
        
        # Enhanced convolutional blocks with residual connections
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(3, 128, stride=2),      # 512 -> 256
            self._make_conv_block(128, 256, stride=2),    # 256 -> 128
            self._make_conv_block(256, 512, stride=2),    # 128 -> 64
        ])
        
        # Spatial attention for better feature extraction
        self.spatial_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Final projection to quantization dimension
        self.final_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.norm = nn.GroupNorm(32, 512)
        self.act = GELU(use_approx=True)
        
        # Quantizer
        self.quantizer = VectorQuantizer(config)
    
    def _make_conv_block(self, in_channels: int, out_channels: int, stride: int = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1),
            nn.GroupNorm(32, out_channels),
            GELU(use_approx=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            GELU(use_approx=True),
            # Residual connection if dimensions match
            nn.Identity() if in_channels == out_channels and stride == 1 else
            nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch, 3, 512, 512] - RGB frames
        Returns:
            token_ids: [batch, 256] - quantized tokens
            losses: dictionary of VQ losses
        """
        batch_size = x.size(0)
        
        # Apply convolutional blocks
        for i, block in enumerate(self.conv_blocks):
            if i == 0:
                h = block[:-1](x)  # All layers except residual
            else:
                residual = h
                h_new = block[:-1](h)
                # Add residual connection
                if h.shape == h_new.shape:
                    h = h_new + residual
                else:
                    h = h_new + block[-1](residual)
        
        # Apply final processing
        h = self.act(self.norm(self.final_conv(h)))  # [batch, 512, 64, 64]
        
        # Reshape for quantization: [batch, 64, 64, 512]
        h = h.permute(0, 2, 3, 1)
        
        # Apply spatial attention
        h_flat = h.view(batch_size, -1, 512)  # [batch, 4096, 512]
        h_attn, _ = self.spatial_attention(h_flat, h_flat, h_flat)
        h = h_attn.view(batch_size, 64, 64, 512)
        
        # Vector quantization
        quantized, indices, vq_losses = self.quantizer(h)
        
        # Convert to patch tokens (4x4 patches -> 16x16 = 256 tokens)
        patches = quantized.view(batch_size, 16, 4, 16, 4, 512)
        patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
        patches = patches.view(batch_size, 256, 4*4*512)
        
        # Average pool each patch to get final token representation
        token_features = patches.mean(dim=2)  # [batch, 256, 512]
        
        # Get token IDs by re-quantizing the averaged features
        _, token_ids, _ = self.quantizer(token_features)
        
        return token_ids.squeeze(-1), vq_losses  # [batch, 256]

class AdvancedVQVAEDecoder(nn.Module):
    """Enhanced decoder with upsampling and attention mechanisms"""
    
    def __init__(self, config: T2VConfig):
        super().__init__()
        self.config = config
        
        # Embedding lookup
        self.embedding = nn.Embedding(config.vocab_size, 512)
        
        # Initial projection
        self.initial_conv = nn.Conv2d(512, 512, 3, padding=1)
        
        # Upsampling blocks with attention
        self.deconv_blocks = nn.ModuleList([
            self._make_deconv_block(512, 256, scale_factor=2),  # 64 -> 128
            self._make_deconv_block(256, 128, scale_factor=2),  # 128 -> 256  
            self._make_deconv_block(128, 64, scale_factor=2),   # 256 -> 512
        ])
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            GELU(use_approx=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Perceptual loss network (frozen pretrained features)
        self.perceptual_net = self._build_perceptual_net()
    
    def _make_deconv_block(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            GELU(use_approx=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            GELU(use_approx=True)
        )
    
    def _build_perceptual_net(self):
        """Build a lightweight perceptual loss network"""
        # Use first few layers of a pretrained network for perceptual loss
        class SimplePerceptualNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            
            def forward(self, x):
                return self.features(x)
        
        net = SimplePerceptualNet()
        # Freeze parameters
        for param in net.parameters():
            param.requires_grad = False
        return net
    
    def forward(self, token_ids: torch.Tensor, target_image: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            token_ids: [batch, 256] - quantized token IDs
            target_image: [batch, 3, 512, 512] - target for perceptual loss
        Returns:
            Dictionary containing reconstructed image and losses
        """
        batch_size = token_ids.size(0)
        
        # Embed tokens
        h = self.embedding(token_ids)  # [batch, 256, 512]
        
        # Reshape to spatial grid: 256 tokens -> 16x16 spatial layout
        h = h.view(batch_size, 16, 16, 512).permute(0, 3, 1, 2)  # [batch, 512, 16, 16]
        
        # Upsample to 64x64
        h = F.interpolate(h, size=64, mode='bilinear', align_corners=False)
        h = self.initial_conv(h)
        
        # Apply deconvolution blocks
        for block in self.deconv_blocks:
            h = block(h)
        
        # Final output
        reconstructed = self.final_conv(h)  # [batch, 3, 512, 512]
        
        # Calculate losses if target provided
        losses = {}
        if target_image is not None:
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, target_image)
            losses['reconstruction_loss'] = recon_loss
            
            # Perceptual loss
            with torch.no_grad():
                target_features = self.perceptual_net(target_image)
            recon_features = self.perceptual_net(reconstructed)
            perceptual_loss = F.mse_loss(recon_features, target_features)
            losses['perceptual_loss'] = perceptual_loss
        
        return {
            'reconstructed': reconstructed,
            'losses': losses
        } 