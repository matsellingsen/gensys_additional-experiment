# Transformer Architecture: Technical Implementation Guide

## Table of Contents
1. [Module Dependencies](#module-dependencies)
2. [Class Hierarchy](#class-hierarchy)
3. [Component Implementations](#component-implementations)
4. [Detailed Function Specifications](#detailed-function-specifications)
5. [Design Decisions and Rationale](#design-decisions-and-rationale)

---

## Module Dependencies

### Core Imports
```python
import torch                    # Tensor operations, autograd
import torch.nn as nn          # Neural network layers (Linear, Embedding, Dropout, etc.)
import torch.nn.functional as F # Functional API (softmax, dropout, etc.)
import math                     # Mathematical functions (sqrt, pi, log)
import numpy as np             # Numerical computations (optional, for initialization)
```

### PyTorch Components Used

| Component | Purpose | Alternative |
|-----------|---------|-------------|
| `nn.Embedding` | Token embeddings (vocab → d_model) | Manual weight matrix |
| `nn.Linear` | Dense transformations | Custom weight + bias |
| `nn.LayerNorm` | Normalization | BatchNorm (not recommended) |
| `nn.Dropout` | Regularization | DropConnect, DropBlock |
| `nn.ReLU` | Activation in FFN | GELU, SiLU |
| `torch.softmax` | Attention weights | Other normalization |
| `torch.matmul` | Matrix multiplication | `@` operator |
| `nn.ModuleList` | Container for layers | Python list (not recommended) |

---

## Class Hierarchy

### Overall Architecture

```
Transformer (nn.Module)
├── Encoder (nn.Module)
│   ├── token_embedding (nn.Embedding)
│   ├── positional_encoding (nn.Embedding or tensor)
│   ├── dropout (nn.Dropout)
│   └── encoder_layers (nn.ModuleList)
│       └── [0..N] TransformerBlock
│           ├── MultiHeadAttention
│           ├── LayerNorm
│           ├── FFN (nn.Sequential)
│           └── LayerNorm
│
├── Decoder (nn.Module)
│   ├── token_embedding (nn.Embedding)
│   ├── positional_encoding (nn.Embedding or tensor)
│   ├── dropout (nn.Dropout)
│   ├── decoder_layers (nn.ModuleList)
│   │   └── [0..N] DecoderBlock
│   │       ├── MaskedMultiHeadAttention (self-attention)
│   │       ├── LayerNorm
│   │       ├── MultiHeadAttention (cross-attention)
│   │       ├── LayerNorm
│   │       ├── FFN
│   │       └── LayerNorm
│   └── linear_output (nn.Linear)
│
└── Supporting modules:
    ├── MultiHeadAttention (nn.Module)
    │   ├── query_layer (nn.Linear)
    │   ├── key_layer (nn.Linear)
    │   ├── value_layer (nn.Linear)
    │   └── output_layer (nn.Linear)
    │
    ├── TransformerBlock (nn.Module)
    │   ├── MultiHeadAttention
    │   ├── LayerNorm [×2]
    │   └── FFN
    │
    └── DecoderBlock (nn.Module)
        ├── MultiHeadAttention [×2]
        ├── LayerNorm [×3]
        └── FFN
```

---

## Component Implementations

### 1. Positional Encoding

#### Implementation Strategy
**Method**: Sinusoidal positional encoding (non-learnable)

```python
def get_sinusoid_table(max_len, emb_dim):
    """Generate sinusoidal positional encoding matrix."""
    sinusoid_table = torch.zeros(max_len, emb_dim)
    
    # Position indices: [0, 1, 2, ..., max_len-1]
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    # Dimension indices: [0, 1, 2, ..., emb_dim-1]
    div_term = torch.exp(
        torch.arange(0, emb_dim, 2).float() * 
        -(math.log(10000.0) / emb_dim)
    )
    
    sinusoid_table[:, 0::2] = torch.sin(position * div_term)    # Even indices
    sinusoid_table[:, 1::2] = torch.cos(position * div_term)    # Odd indices
    
    return sinusoid_table
```

#### Mathematical Formulation
- **Even dimensions (2i)**: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- **Odd dimensions (2i+1)**: `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

#### PyTorch Specific Choices
- **`torch.arange(0, emb_dim, 2)`**: Slicing pattern to handle even/odd dims separately
- **`unsqueeze(1)`**: Expand position to shape [max_len, 1] for broadcasting
- **`exp(...* -(math.log(10000.0) / emb_dim))`**: Numerically stable alternative to division by 10000^(2i/d_model)
- **Slice assignment `[:, 0::2]`**: In-place assignment for memory efficiency

#### Integration in Forward Pass
```python
# Option 1: Store as registered buffer (no gradients)
self.register_buffer('positional_encoding', get_sinusoid_table(max_len, emb_dim))

# Option 2: Use nn.Embedding (learnable, less common)
self.positional_encoding = nn.Embedding(max_len, emb_dim)
```

---

### 2. Multi-Head Attention

#### Class Structure
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Typically: 512 / 8 = 64
        
        # Linear layers for Q, K, V projections
        self.query_layer = nn.Linear(emb_dim, num_heads * head_dim)
        self.key_layer = nn.Linear(emb_dim, num_heads * head_dim)
        self.value_layer = nn.Linear(emb_dim, num_heads * head_dim)
        
        # Output projection
        self.output_layer = nn.Linear(num_heads * head_dim, emb_dim)
```

#### Forward Pass Implementation
```python
def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)
    
    # 1. Linear projections to multiple heads
    Q = self.query_layer(query)              # [B, T_q, emb_dim]
    K = self.key_layer(key)                  # [B, T_k, emb_dim]
    V = self.value_layer(value)              # [B, T_v, emb_dim]
    
    # 2. Reshape for multi-head attention
    # [B, T, emb_dim] → [B, T, num_heads, head_dim] → [B, num_heads, T, head_dim]
    Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    # 3. Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, num_heads, T_q, T_k]
    scores = scores / math.sqrt(self.head_dim)     # Scaling by √d_k
    
    # 4. Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 5. Softmax over key dimension
    attention_weights = torch.softmax(scores, dim=-1)  # [B, num_heads, T_q, T_k]
    
    # 6. Apply to values
    context = torch.matmul(attention_weights, V)  # [B, num_heads, T_q, head_dim]
    
    # 7. Concatenate heads
    context = context.transpose(1, 2).contiguous()  # [B, T_q, num_heads, head_dim]
    context = context.view(batch_size, -1, self.emb_dim)  # [B, T_q, emb_dim]
    
    # 8. Final output projection
    output = self.output_layer(context)
    
    return output
```

#### PyTorch Specific Details
- **`.view()` vs `.reshape()`**: `view()` requires contiguous tensors (faster), while `reshape()` copies if needed
- **`.transpose(1, 2)`**: Swaps dimensions 1 and 2 for multi-head arrangement
- **`.contiguous()`**: Ensures memory layout is contiguous before view() operation
- **`masked_fill(mask == 0, float('-inf'))`**: Sets mask=0 positions to -inf before softmax → 0 after softmax
- **`torch.softmax(dim=-1)`**: Softmax over last dimension (keys)
- **Math scaling**: `/ math.sqrt(head_dim)` prevents attention weights from becoming too small when d_k is large

---

### 3. Transformer Block (Encoder Layer)

#### Structure
```python
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()
        
        # Self-attention
        self.mha = MultiHeadAttention(emb_dim, num_heads)
        
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(emb_dim, eps=1e-6)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
```

#### Forward Pass with Residual Connections
```python
def forward(self, x, mask=None):
    # Sub-layer 1: Multi-head self-attention with residual
    attn_output = self.mha(x, x, x, mask)           # Self-attention (Q=K=V)
    attn_output = self.dropout(attn_output)
    x = self.layernorm1(x + attn_output)             # Residual + LayerNorm (post-norm)
    
    # Sub-layer 2: Feed-forward with residual
    ffn_output = self.ffn(x)
    ffn_output = self.dropout(ffn_output)
    x = self.layernorm2(x + ffn_output)              # Residual + LayerNorm
    
    return x
```

#### Key Design Choices
- **Post-norm** (`LayerNorm(x + residual)`): Standard in modern implementations (more stable)
- **Pre-norm** (`x + LayerNorm(residual)`): Alternative (requires tuning)
- **Layer normalization epsilon=1e-6**: Numerical stability in softmax/gradients
- **ReLU activation**: Standard choice (could use GELU for better performance)
- **FFN expansion factor**: Typically 4× (512 → 2048 → 512)

---

### 4. Encoder Stack

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_layers, num_heads, 
                 forward_dim, dropout, max_len):
        super().__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_encoding = nn.Embedding.from_pretrained(
            get_sinusoid_table(max_len, emb_dim), 
            freeze=True  # Don't train positional encoding
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, dropout, forward_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len]
        seq_len = x.size(1)
        
        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        # Combine token + positional embeddings
        x = self.token_embedding(x) + self.position_encoding(positions)
        x = self.dropout(x)
        
        # Pass through each transformer block
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        return x  # [batch_size, seq_len, emb_dim]
```

#### PyTorch Specific Details
- **`nn.Embedding.from_pretrained()`**: Load pre-computed positional encodings without training
- **`freeze=True`**: Registers buffer instead of parameter (no gradients)
- **`nn.ModuleList`**: Proper container for variable number of layers (ensures parameters are registered)
- **Position indices**: `torch.arange()` creates sequential positions [0, 1, 2, ...] on same device as input

---

### 5. Decoder Block

```python
class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        
        # Masked self-attention
        self.masked_mha = MultiHeadAttention(emb_dim, num_heads)
        
        # Cross-attention (decoder attends to encoder)
        self.cross_attention = MultiHeadAttention(emb_dim, num_heads)
        
        # Three layer norms
        self.layernorm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(emb_dim, eps=1e-6)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Sub-layer 1: Masked self-attention on decoder
        masked_attn = self.masked_mha(x, x, x, tgt_mask)  # Q=K=V=decoder states
        masked_attn = self.dropout(masked_attn)
        x = self.layernorm1(x + masked_attn)
        
        # Sub-layer 2: Cross-attention to encoder
        cross_attn = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        # Q from decoder, K and V from encoder
        cross_attn = self.dropout(cross_attn)
        x = self.layernorm2(x + cross_attn)
        
        # Sub-layer 3: Feed-forward
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        x = self.layernorm3(x + ffn_out)
        
        return x
```

#### Key Implementation Details
- **Masked self-attention**: `tgt_mask` is causal (lower triangular)
- **Cross-attention**: Query from decoder, Keys/Values from encoder
- **Source mask**: Applied in cross-attention to ignore encoder padding
- **Target mask**: Applied in self-attention to prevent looking ahead

---

### 6. Masking Functions

#### Source Padding Mask
```python
def create_src_mask(src):
    """
    Prevent attention to padding tokens in encoder.
    
    Returns mask shape: [batch_size, 1, 1, src_len]
    """
    # src shape: [batch_size, src_len]
    # (src != PAD_ID) gives [batch_size, src_len] boolean matrix
    src_mask = (src != PAD_ID).unsqueeze(1).unsqueeze(2)
    # Unsqueeze twice to get [batch_size, 1, 1, src_len]
    
    return src_mask  # True for real tokens, False for padding
```

#### Target Causal Mask
```python
def create_tgt_mask(tgt, tgt_len):
    """
    Prevent decoder from attending to future tokens (causal masking).
    Also prevents attention to padding tokens.
    
    Returns mask shape: [batch_size, 1, tgt_len, tgt_len]
    """
    batch_size = tgt.size(0)
    
    # Causal mask (lower triangular): position i can attend to positions ≤ i
    # torch.tril() creates lower triangular matrix
    causal_mask = torch.tril(
        torch.ones((tgt_len, tgt_len), device=tgt.device)
    )  # Shape: [tgt_len, tgt_len]
    
    # Padding mask: [batch_size, 1, 1, tgt_len]
    padding_mask = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)
    
    # Expand causal mask to batch dimension: [batch_size, 1, tgt_len, tgt_len]
    causal_mask = causal_mask.expand(batch_size, 1, tgt_len, tgt_len)
    
    # Combine: causal AND padding
    # Both must be True (1.0) for attention to be allowed
    mask = causal_mask * padding_mask
    
    return mask
```

#### Mask Application in Attention
```python
if mask is not None:
    # Before softmax: set masked positions to -inf
    scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## Detailed Function Specifications

### Key Mathematical Operations

| Operation | PyTorch Code | Dimensions | Purpose |
|-----------|-------------|-----------|---------|
| Scaled dot-product | `Q @ K.T / √d_k` | [B, h, T, T] | Compute attention scores |
| Softmax | `softmax(scores, dim=-1)` | [B, h, T, T] | Normalize attention |
| Context | `attn @ V` | [B, h, T, d] | Weight values by attention |
| Linear transform | `nn.Linear(in, out)` | [*, in] → [*, out] | Project dimensions |

### Numerical Stability Considerations

```python
# 1. Attention scaling prevents exploding/vanishing gradients
scores /= math.sqrt(head_dim)  # Critical for training stability

# 2. Layer normalization with epsilon prevents division by zero
nn.LayerNorm(dim, eps=1e-6)

# 3. Masked fill uses -inf before softmax (converts to 0 after)
scores.masked_fill(mask == 0, float('-inf'))
# Better than setting to 0 directly (would still have softmax contribution)

# 4. Dropout position (after softmax, before matmul)
attention_weights = torch.softmax(scores, dim=-1)
attention_weights = torch.dropout(attention_weights, p=dropout, training=True)
```

---

## Design Decisions and Rationale

### 1. Embedding Dimension Divisibility
```python
# Must divide evenly by num_heads
head_dim = emb_dim // num_heads
# If emb_dim=512, num_heads=8 → head_dim=64 ✓
# If emb_dim=512, num_heads=7 → head_dim=73 ✗ (some info lost)
```

### 2. FFN Expansion Factor
```python
# Typical: forward_dim = 4 * emb_dim
forward_dim = 4 * 512  # 2048
# Creates bottleneck: 512 → 2048 → 512
# Increases model capacity for non-linear transformations
```

### 3. Dropout Strategy
```python
# Applied in three places:
1. After embedding + positional encoding
2. After attention weights (before matmul with values)
3. After FFN output
# Prevents co-adaptation of units
```

### 4. Initialization
```python
# PyTorch default: Kaiming uniform for Linear layers
# nn.Linear weight: U(-√k, √k) where k = 1/in_features
# This is sufficient for Transformers
```

### 5. Gradient Flow
```python
# Residual connections allow gradients to bypass layers
# y = x + f(x)
# dy/dx = 1 + df/x
# Without residuals: dy/dx = df/x (can vanish with many layers)
```

### 6. Post-Norm vs Pre-Norm
```python
# Post-norm (implemented above):
# x = LayerNorm(x + f(x))
# More stable for deep networks
# Alternative Pre-norm:
# x = x + f(LayerNorm(x))
# Slightly different learning dynamics
```

---

## Parameter Counts and Memory

### Typical Configuration (from implementation)
```
EMB_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 8
FORWARD_DIM = 256

Parameters per encoder block:
- Attention: (128 * 128 * 3) + (128 * 128) = 49,664
- FFN: (128 * 256) + (256 * 128) = 65,536
- LayerNorms: 256 (2 × 128)
Total per block: ~115,500

Total for 2-layer encoder + decoder: ~231,000 parameters
Plus embeddings: vocab_size * 128
```

### Memory During Forward Pass
```
Activation memory = batch_size × seq_len × emb_dim × num_layers
Attention matrices = batch_size × num_heads × seq_len × seq_len (quadratic!)
```

---

## Common Implementation Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Using `.reshape()` before `.contiguous()` | Breaks gradient flow | Use `.contiguous().view()` |
| Mask shape mismatch | Broadcasting errors | Verify: [B, 1, T, T] format |
| Positional encoding not on device | Device mismatch errors | Use `device=x.device` |
| Not masking before softmax | Learns to attend to masked positions | Set to -inf before softmax |
| Large attention scores | Vanishing gradients | Always divide by √d_k |
| Improper residual connections | Training instability | Use residual in all blocks |
| Forgetting padding mask in cross-attention | Attends to encoder padding | Apply src_mask in cross-attention |

---

## Optimization Techniques

### Efficient Implementations
1. **Fused attention**: Combine multiple operations to reduce memory
2. **Sparse attention**: Only attend to local positions (for long sequences)
3. **Flash Attention**: I/O aware algorithm (external library)
4. **Quantization**: Reduce precision (int8/fp16) for inference

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits = model(src, tgt)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Conclusion

This implementation demonstrates:
- **Modular design**: Clear separation of concerns (attention, FFN, blocks, stacks)
- **PyTorch best practices**: Proper use of nn.Module, parameter registration, device handling
- **Numerical stability**: Scaling, normalization, careful masking
- **Extensibility**: Easy to modify for variants (RoPE encodings, different activations, etc.)

The class hierarchy supports both research experimentation and production deployment while maintaining clarity and maintainability.
