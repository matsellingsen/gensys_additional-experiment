import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        # TODO
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        # Might look redundant, because: self.num_heads * self.head_dim == emb_dim
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.linear_value = nn.Linear(self.emb_dim, self.num_heads * self.head_dim)
        self.linear_key = nn.Linear(self.emb_dim, self.num_heads * self.head_dim)
        self.linear_query = nn.Linear(self.emb_dim, self.num_heads * self.head_dim)
        
        self.linear_output = nn.Linear(self.num_heads * self.head_dim, self.emb_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        v_len = value.size(1)
        emb_dim = query.size(2)

        assert emb_dim == self.emb_dim, (
            f"Expected emb_dim={self.emb_dim}, but got {emb_dim}"
        )

        # Reminder: PyTorch tensors are backed by
        # 1 flat, contiguous array in memory
        # + shape metadata
        # + stride metadat
        # `view` creates a new interpretation of the same memory (no data copied),
        # giving it a different logical structure.
    
        # 1. Linear projections
        Q = self.linear_query(query)
        K = self.linear_key(key)
        V = self.linear_value(value)

        # 2. Split embedding dimension into multiple heads
        # Change the "view" of the data (no computation, only reshaping):
        # [batch, seq_len, emb_dim]
        # -> split emb_dim into (num_heads × head_dim)
        # -> [batch, seq_len, num_heads, head_dim]
        # This is valid because emb_dim == num_heads * head_dim
        # Example: [2, 4, 8] with 2 heads -> [2, 4, 2, 4]
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, k_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, v_len, self.num_heads, self.head_dim)

        # Rearrange tensor dimensions so that attention is computed per head:
        # Move num_heads in front of seq_len, because torch.matmul operates
        # on the last two dimensions and treats all earlier ones as batch dimensions.
        # [batch, seq_len, num_heads, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # 3. Scaled dot-product attention
        # Transpose keys so that matrix multiplication works:
        # [batch, heads, seq_len, head_dim]
        # -> [batch, heads, head_dim, seq_len]
        # This allows each query vector to dot-product with all key vectors
        K_t = K.transpose(-2, -1)

        key_out = torch.matmul(Q, K_t)
        key_out = key_out / math.sqrt(self.head_dim)

        # 4. Mask (optional)
        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -1e20)

        # 5. Softmax over key dimension (which tokens to attend to)
        attention = torch.softmax(key_out, dim=-1)

        # 6. Weighted sum of values
        out = torch.matmul(attention, V)

        # 7. Combine heads back into a single embedding
        # Move heads back behind sequence dimension:
        # [batch, heads, seq_len, head_dim]
        # -> [batch, seq_len, heads, head_dim]
        out = out.permute(0, 2, 1, 3)

        # Ensure memory is contiguous before reshaping
        out = out.contiguous()

        # Concatenate all heads:
        # [batch, seq_len, heads, head_dim]
        # -> [batch, seq_len, emb_dim]
        # (this is where the "concat" of heads happens)
        out = out.view(batch_size, q_len, self.emb_dim)

        # 8. Final linear projection
        out = self.linear_output(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.forward_dim = forward_dim

        self.multihead_attention = MultiHeadAttention(self.emb_dim, self.num_heads)
        
        self.norm1 = nn.LayerNorm(self.emb_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.emb_dim, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(self.emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, self.emb_dim)
        )

    def forward(self, query, key, value, attention_mask):
        # 1. Multi-head attention
        attn_out = self.multihead_attention(query, key, value, attention_mask)

        # 2. Skip connection + normalization
        query_with_attention = (attn_out + query)
        query_with_attention = self.dropout(query_with_attention)
        query_with_attention = self.norm1(query_with_attention)
        
        # 3. Feed-forward network
        ffn_out = self.ffn(query_with_attention)

        # 4. Skip connection + normalization
        query_with_ffn_out = (ffn_out + query_with_attention)
        query_with_ffn_out = self.dropout(query_with_ffn_out)
        query_with_ffn_out = self.norm2(query_with_ffn_out)

        block_out = query_with_ffn_out

        return block_out

class CrossAttentionBlock(nn.Module):
    """
    Decoder cross-attention + feed-forward block.
    Explicitly separates encoder–decoder attention
    from self-attention for clarity.
    """
    def __init__(
        self,
        emb_dim,
        num_heads,
        forward_dim,
        dropout,
    ):
        super().__init__()

        self.cross_attention = MultiHeadAttention(
            emb_dim,
            num_heads,
        )

        self.feedforward_network = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim),
        )

        self.layer_norm_attention = nn.LayerNorm(emb_dim, eps=1e-6)
        self.layer_norm_feedforward = nn.LayerNorm(emb_dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_sequence,
        encoder_keys,
        encoder_values,
        source_mask,
    ):
        # Encoder–decoder attention
        attention_output = self.cross_attention(
            decoder_sequence,
            encoder_keys,
            encoder_values,
            source_mask,
        )

        attention_residual = self.layer_norm_attention(
            self.dropout(attention_output + decoder_sequence)
        )

        # Feed-forward
        feedforward_output = self.feedforward_network(attention_residual)

        output = self.layer_norm_feedforward(
            self.dropout(feedforward_output + attention_residual)
        )

        return output


def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()

        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        # Positional encodings (sinusoid, frozen)
        pos_table = get_sinusoid_table(max_len + 1, emb_dim)
        self.position_embedding = nn.Embedding.from_pretrained(
            pos_table,
            freeze=True
        )

        # Make a dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    forward_dim=forward_dim
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, source_sequence, source_mask):
        batch_size, sequence_length = source_sequence.shape

        assert sequence_length < self.position_embedding.num_embeddings, (
            "Source sequence length exceeds maximum positional encoding length"
        )


        # Create position indices [1..seq_len] for each batch element
        positions = torch.arange(sequence_length)                # [seq_len]
        positions = positions.unsqueeze(0)               # add batch dim -> [1, seq_len]
        positions = positions.expand(batch_size, sequence_length)  # repeat for batch -> [batch, seq_len]
        positions = positions + 1                         # reserve 0 for [PAD]
        positions = positions.to(source_sequence.device)                # move to same device (cpu/gpu) as input

        # Embeddings
        token_emb = self.token_embedding(source_sequence)
        pos_emb = self.position_embedding(positions)

        # Sum + dropout
        token_pos_emb = token_emb + pos_emb
        token_pos_emb = self.dropout(token_pos_emb)

        encoder_out = token_pos_emb

        # Transformer blocks
        for layer in self.layers:
            encoder_out = layer(encoder_out, encoder_out, encoder_out, source_mask)

        return encoder_out

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        
        # Masked self-attention (decoder attends to itself)
        self.multihead_attention = MultiHeadAttention(emb_dim, num_heads)

        # LayerNorm after first skip connection
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)

        # Cross-attention + FFN (reuse TransformerBlock)
        self.cross_attention_block = CrossAttentionBlock(
            emb_dim=emb_dim,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
        )


        self.dropout = nn.Dropout(dropout)

    def forward(self, target_sequence, encoder_values, encoder_keys, source_mask, target_mask):
        # 1. Masked self-attention (decoder attends to itself)
        multihead_attention_out = self.multihead_attention(target_sequence, target_sequence, target_sequence, target_mask)

        # 2. Skip connection + normalization
        x_with_self_attn = multihead_attention_out + target_sequence
        x_with_self_attn = self.dropout(x_with_self_attn)
        x_with_self_attn = self.norm(x_with_self_attn)

        # 3. Cross-attention + FFN (encoder-decoder attention)
        out = self.cross_attention_block(
            x_with_self_attn,
            encoder_keys,
            encoder_values,
            source_mask
        )

        return out

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        # Sinusoidal positional embeddings (frozen) for length extrapolation
        pos_table = get_sinusoid_table(max_len, emb_dim)
        self.position_embedding = nn.Embedding.from_pretrained(
            pos_table,
            freeze=True
        )

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    forward_dim=forward_dim,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.output_layer = nn.Linear(emb_dim, vocab_size)

    def forward(self, target_input_sequence, encoder_out, source_mask, target_mask):
        batch_size, sequence_length = target_input_sequence.shape
        
        assert sequence_length < self.position_embedding.num_embeddings, (
            "Target sequence length exceeds maximum positional encoding length"
        )


        # Create position indices for decoder tokens
        positions = torch.arange(sequence_length)                  # [seq_len]
        positions = positions.unsqueeze(0)                 # [1, seq_len]
        positions = positions.expand(batch_size, sequence_length)  # [batch, seq_len]
        positions = positions.to(target_input_sequence.device)   # move to same device as input  

        # Token + positional embeddings
        token_emb = self.token_embedding(target_input_sequence)
        pos_emb = self.position_embedding(positions)

        decoder_out = self.dropout(token_emb + pos_emb)

        # Pass through stacked Decoder blocks
        for layer in self.layers:
            decoder_out = layer(
                decoder_out,
                encoder_out,
                encoder_out,
                source_mask,
                target_mask
            )

        # Project to vocabulary size
        logits = self.output_layer(decoder_out)

        return logits

class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        source_padding_index,
        target_padding_index,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_sequence_length=128,
    ):
        super().__init__()

        self.source_padding_index = source_padding_index
        self.target_padding_index = target_padding_index

        self.encoder = Encoder(
            vocab_size=source_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_sequence_length,
        )

        self.decoder = Decoder(
            vocab_size=target_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_sequence_length,
        )

    def create_source_padding_mask(self, source_sequence):
        device = source_sequence.device
        # (batch_size, 1, 1, src_seq_len)
        source_mask = (
            (source_sequence != self.source_padding_index)
            .unsqueeze(1)
            .unsqueeze(2)
        )
        return source_mask.to(device)

    def create_target_causal_mask(self, target_sequence):
        batch_size, target_length = target_sequence.shape

        padding_mask = (
            target_sequence != self.target_padding_index
        ).unsqueeze(1).unsqueeze(2)

        causal_mask = torch.tril(
            torch.ones(
                (target_length, target_length),
                device=target_sequence.device,
            )
        )

        return padding_mask * causal_mask


    def forward(self, source_sequence, target_input_sequence):
        # Create masks
        source_mask = self.create_source_padding_mask(source_sequence)
        target_mask = self.create_target_causal_mask(target_input_sequence)

        # Encode source sequence
        encoder_out = self.encoder(source_sequence, source_mask)

        # Decode target sequence
        out = self.decoder(
            target_input_sequence,
            encoder_out,
            source_mask,
            target_mask
        )

        return out