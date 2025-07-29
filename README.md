# Enhanced Autoregressive Text-to-Video Model

A production-ready implementation of an advanced autoregressive text-to-video model with state-of-the-art optimizations and modern techniques.

## 🚀 Key Features

### **Architecture Improvements**
- **RMSNorm** instead of LayerNorm for better stability
- **SwiGLU activation** alongside GELU for optimal performance
- **Grouped Query Attention** for memory efficiency
- **Rotary Position Embedding (RoPE)** for better positional understanding
- **Gradient checkpointing** for memory optimization

### **VQ-VAE Enhancements**
- **EMA codebook updates** for stable quantization
- **Perceptual loss** for better visual quality
- **Commitment loss** with proper weighting
- **Spatial attention** in encoder for better feature extraction

### **Training Optimizations**
- **Mixed precision training** with BFloat16
- **Distributed training** support
- **Advanced loss formulation** with multiple components
- **Proper gradient clipping** and learning rate scheduling
- **Comprehensive checkpointing** system

### **Inference Features**
- **Multiple sampling strategies** (top-k, top-p, typical sampling)
- **KV caching** for efficient generation
- **Memory management** with sliding window
- **Progress tracking** and error handling

## 📁 Project Structure

```
├── config.py              # Enhanced configuration with advanced settings
├── normalization.py       # RMSNorm, SwiGLU, and GELU implementations
├── rope.py               # Rotary Position Embedding
├── vqvae.py              # Enhanced VQ-VAE with advanced features
├── transformer.py        # Enhanced transformer with modern techniques
├── trainer.py            # Advanced training loop with mixed precision
├── sampler.py            # Advanced sampling strategies
├── generator.py          # Production-ready video generation
├── main.py               # Main entry point and CLI
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd enhanced-t2v-model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 🎯 Usage

### **Training**

```bash
# Basic training
python main.py --mode train --train_data /path/to/training/data

# Training with evaluation
python main.py --mode train \
    --train_data /path/to/training/data \
    --eval_data /path/to/eval/data \
    --checkpoint /path/to/checkpoint.pt
```

### **Video Generation**

```bash
# Generate video with default settings
python main.py --mode generate \
    --prompt "A robot dancing in a park with colorful flowers" \
    --checkpoint /path/to/trained_model.pt

# Generate with custom parameters
python main.py --mode generate \
    --prompt "A cat playing with a ball" \
    --checkpoint /path/to/trained_model.pt \
    --output custom_video.mp4 \
    --num_frames 180 \
    --temperature 0.7
```

### **Model Evaluation**

```bash
python main.py --mode evaluate \
    --checkpoint /path/to/trained_model.pt \
    --eval_data /path/to/eval/data
```

## ⚙️ Configuration

The model uses a comprehensive configuration system with sensible defaults:

```python
from config import T2VConfig

# Use default configuration
config = T2VConfig()

# Customize specific parameters
config.embed_dim = 2048
config.num_layers = 24
config.learning_rate = 1e-4
config.batch_size = 2
```

### **Key Configuration Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 4096 | Model embedding dimension |
| `num_layers` | 40 | Number of transformer layers |
| `num_heads` | 32 | Number of attention heads |
| `vocab_size` | 16384 | VQ-VAE vocabulary size |
| `resolution` | 512 | Video resolution |
| `fps` | 30 | Frames per second |
| `learning_rate` | 2e-4 | Learning rate |
| `batch_size` | 1 | Training batch size |

## 🔧 Advanced Features

### **Mixed Precision Training**

The model automatically uses mixed precision training when available:

```python
config.use_mixed_precision = True
config.precision = "bf16"  # Better than fp16 for stability
```

### **Distributed Training**

Support for multi-GPU training:

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 main.py --mode train

# Multi-node training
torchrun --nnodes=2 --nproc_per_node=4 main.py --mode train
```

### **Advanced Sampling**

Multiple sampling strategies for high-quality generation:

```python
# Top-k sampling
result = generator.generate_video(
    text_prompt="A beautiful sunset",
    top_k=50,
    temperature=0.8
)

# Nucleus sampling
result = generator.generate_video(
    text_prompt="A beautiful sunset",
    top_p=0.9,
    temperature=0.8
)

# Typical sampling
result = generator.generate_video(
    text_prompt="A beautiful sunset",
    use_typical_sampling=True,
    typical_p=0.95,
    temperature=0.8
)
```

## 📊 Model Architecture

### **Transformer Architecture**
- **40 layers** with 4096 embedding dimension
- **Grouped Query Attention** with 32 heads (8 KV heads)
- **SwiGLU feed-forward** with 11008 hidden dimension
- **RMSNorm** for stable training
- **Rotary Position Embedding** for better positional understanding

### **VQ-VAE Architecture**
- **3-stage encoder** with residual connections
- **Spatial attention** for better feature extraction
- **EMA codebook updates** for stable quantization
- **Perceptual loss** for visual quality
- **16x16 patch tokens** per frame

### **Training Features**
- **Gradient checkpointing** for memory efficiency
- **Mixed precision** with BFloat16
- **Distributed training** support
- **Comprehensive loss** with multiple components
- **Advanced scheduling** with cosine annealing

## 🎨 Generation Examples

The model can generate high-quality videos from text prompts:

```
"A robot dancing in a park with colorful flowers"
"A cat playing with a ball in a sunny garden"
"A car driving through a futuristic city at night"
"A butterfly flying over a field of flowers"
```

## 🔬 Technical Details

### **Memory Optimization**
- **Gradient checkpointing** reduces memory usage by ~50%
- **Mixed precision** reduces memory usage by ~50%
- **Sliding window attention** for long sequences
- **Efficient KV caching** during generation

### **Training Stability**
- **RMSNorm** provides better gradient flow
- **Proper weight initialization** using best practices
- **Gradient clipping** prevents exploding gradients
- **Learning rate scheduling** for stable convergence

### **Generation Quality**
- **Multiple sampling strategies** for diverse outputs
- **Temporal consistency loss** for smooth videos
- **Perceptual loss** for visual quality
- **Advanced attention mechanisms** for better understanding

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on modern transformer architectures
- Inspired by state-of-the-art text-to-video models
- Uses best practices from the research community

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note:** This is a research implementation. For production use, additional testing, optimization, and deployment considerations should be made. 