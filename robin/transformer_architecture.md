# Transformer Architecture Implementation

## Overview

Lets go thorugh the complete Transformer model implementation which is built from scratch using PyTorch. The Transformer architecture, introduced in "Attention is All You Need" (Vaswani et al., 2017), revolutionized sequence-to-sequence modeling by replacing recurrent connections with self-attention mechanisms.

### Complete Architecture

The Transformer we built follows the original encoder-decoder structure:

``` 
Input Sequence → Encoder → Encoded Representation → Decoder → Output Sequence
```

DISPLAY COMPLETE ARCHITECTURE HERE AND POINT OUT THE FOLLOWING COMPONENTS
**Key Components:** 
- **Embeddings**: Convert discrete tokens to continuous vector representations
- **Positional Encoding**: Inject sequence order information (Transformers have no inherent notion of position)
- **Multi-Head Attention**: Allow the model to jointly attend to information from different representation subspaces
- **Feed-Forward Networks**: Apply non-linear transformations to each position independently
- **Layer Normalization**: Stabilize training by normalizing activations
- **Encoder Stack**: Multiple encoder layers that process the input sequence
- **Decoder Stack**: Multiple decoder layers that generate the output sequence autoregressively
- **Masking**: Prevent attention to future positions (decoder) and padding tokens

---

## Building Blocks (in order of implementation)

### 1. Positional Encoding

**Concept**: Unlike RNNs, Transformers process all positions in parallel and have no inherent understanding of sequence order. Positional encodings add position-specific patterns to embeddings.

**Implementation**: Uses sinusoidal functions with different frequencies:
- Even dimensions: `sin(pos / 10000^(2i/d_model))`
- Odd dimensions: `cos(pos / 10000^(2i/d_model))`

**Technical Advantages**:
- **Deterministic**: No learnable parameters, works for any sequence length
- **Extrapolation**: Can handle sequences longer than those seen during training
- **Unique patterns**: Each position gets a unique encoding
- **Relative position awareness**: The model can learn to attend by relative positions due to the sinusoidal properties (linear combinations of encodings can represent offsets)

---

### 2. Multi-Head Attention

**Concept**: The core innovation of Transformers. Instead of computing attention once, the mechanism splits the representation into multiple "heads," each learning different aspects of relationships between tokens.

**Mechanism**:
1. **Linear projections**: Input is projected to Query (Q), Key (K), and Value (V) matrices
2. **Scaled dot-product attention**: `Attention(Q,K,V) = softmax(QK^T / √d_k)V`
3. **Multiple heads**: Run this process in parallel with different learned projections
4. **Concatenate & project**: Combine all heads and apply a final linear transformation

**Technical Advantages**:
- **Multiple representation subspaces**: Different heads can capture different types of relationships (e.g., syntactic vs. semantic)
- **Parallel computation**: All positions processed simultaneously (unlike sequential RNNs)
- **Scalability**: Computational complexity is O(n²d) where n is sequence length
- **Interpretability**: Attention weights can be visualized to understand what the model focuses on
- **Scaling factor (√d_k)**: Prevents dot products from growing too large, which would push softmax into regions with vanishing gradients

---

### 3. Encoder Block

**Concept**: A single encoder layer that processes the input through self-attention and feed-forward networks with residual connections.

**Structure**:
1. **Multi-head self-attention**: Allows each position to attend to all positions in the input
2. **Add & Normalize**: Residual connection + Layer Normalization
3. **Position-wise Feed-Forward**: Two linear transformations with ReLU activation
4. **Add & Normalize**: Another residual connection + Layer Normalization

**Technical Advantages**:
- **Residual connections**: Enable gradient flow through deep networks, allowing stacking of many layers
- **Layer normalization**: Stabilizes training by normalizing across the feature dimension
- **Position-wise FFN**: Applies the same transformation independently to each position, allowing the model to process information locally after global attention
- **Self-attention parallelization**: All positions processed simultaneously rather than sequentially

---

### 4. Encoder Stack

**Concept**: Stack multiple encoder blocks to create increasingly abstract representations of the input sequence.

**Structure**: N identical encoder layers applied sequentially, where the output of one layer becomes the input to the next.

**Technical Advantages**:
- **Hierarchical representations**: Lower layers capture simple patterns, higher layers capture complex abstractions
- **Information flow**: Residual connections allow information to flow from input directly to any layer
- **Depth benefits**: Deeper networks can model more complex functions without sequential computation bottlenecks (unlike deep RNNs)

---

### 5. Decoder Block

**Concept**: Similar to encoder blocks but with additional cross-attention to incorporate encoder outputs and causal masking for autoregressive generation.

**Structure**:
1. **Masked multi-head self-attention**: Attends only to previous positions in the output sequence
2. **Add & Normalize**
3. **Cross-attention**: Queries from decoder attend to encoder outputs (Keys and Values)
4. **Add & Normalize**
5. **Position-wise Feed-Forward**
6. **Add & Normalize**

**Technical Advantages**:
- **Causal masking**: Prevents the decoder from "cheating" by looking at future tokens during training
- **Cross-attention**: Allows the decoder to focus on relevant parts of the input sequence for each output token
- **Autoregressive generation**: Can generate sequences token-by-token at inference time
- **Parallel training**: Despite autoregressive inference, training can be parallelized by masking

---

### 6. Decoder Stack

**Concept**: Stack multiple decoder blocks, similar to the encoder, but with the added complexity of both self-attention and cross-attention.

**Structure**: N identical decoder layers, each attending to previous decoder positions and the encoder output.

**Technical Advantages**:
- **Progressive refinement**: Each layer refines the output representation
- **Long-range dependencies**: Multiple layers of attention can capture dependencies across the entire sequence
- **Encoder-decoder alignment**: Cross-attention in each layer allows fine-grained alignment between input and output

---

### 7. Complete Transformer Model

**Concept**: Combines encoder and decoder stacks with embeddings, positional encodings, and a final linear projection.

**Full Pipeline**:
1. **Source embedding + positional encoding** → Encoder stack → Encoded representation
2. **Target embedding + positional encoding** → Decoder stack (with cross-attention to encoder)
3. **Linear projection** → Vocabulary-sized logits
4. **Softmax** (during training/inference) → Probability distribution over vocabulary

**Technical Advantages**:
- **No recurrence**: Enables parallel training across all positions
- **Constant path length**: Information flows directly between any two positions (O(1) rather than O(n) in RNNs)
- **Flexible attention**: Can model arbitrary dependencies without sequential bottlenecks
- **Scalability**: Scales better to longer sequences than RNNs (though quadratic in sequence length)
- **Transfer learning**: Pre-trained Transformers can be fine-tuned for downstream tasks

---

## Key Design Decisions

### Masking

**Source Padding Mask**: Prevents attention to padding tokens in the input sequence.
- Shape: `[batch_size, 1, 1, source_length]`
- Ensures padding tokens don't influence representations

**Target Padding Mask**: Prevents attention to padding tokens in the output.
- Shape: `[batch_size, 1, target_length, target_length]`

**Causal Mask (Look-ahead Mask)**: Prevents attending to future tokens during training.
- Combined with target padding mask
- Lower triangular matrix: position i can only attend to positions ≤ i
- Enables parallel training while maintaining autoregressive property

### Dropout

Applied after:
- Attention weights (before multiplication with values)
- Feed-forward network outputs
- Residual connections

**Advantages**: Prevents overfitting and improves generalization.

---

## Computational Considerations

**Complexity**:
- Self-attention: O(n²·d) where n = sequence length, d = model dimension
- Feed-forward: O(n·d²)
- For long sequences, self-attention dominates computational cost

**Memory**:
- Attention matrices: O(n²) per head
- Activations: O(n·d·layers)

**Training Efficiency**:
- Fully parallelizable across sequence positions
- Batch processing across multiple sequences
- No sequential dependencies within layers (unlike RNNs)

---

## Summary

The Transformer architecture achieves state-of-the-art performance on sequence-to-sequence tasks through:

1. **Self-attention mechanisms** that capture long-range dependencies without recurrence
2. **Multi-head attention** that learns diverse representations
3. **Positional encodings** that inject order information
4. **Residual connections and layer normalization** that enable deep architectures
5. **Parallel processing** that dramatically speeds up training
6. **Flexible attention patterns** through masking

This implementation demonstrates how these components work together to create a powerful, efficient, and scalable sequence modeling architecture that has become the foundation for modern NLP systems (BERT, GPT, T5, etc.).
