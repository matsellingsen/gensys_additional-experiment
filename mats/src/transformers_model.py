import os
import sys
import math
import random
import numpy as nps

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()

        # TODO
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.query_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.key_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.value_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.output_layer = nn.Linear(num_heads * self.head_dim, emb_dim)

    def forward(self, query, key, value, mask=None):
        # TODO
        #1. pass through linear layers
        query_lin = self.query_layer(query)
        key_lin = self.key_layer(key)
        value_lin = self.value_layer(value)

        #print("query_lin shape:", query_lin.shape)

        #2. reshape for multi-head attention
        batch_size = query.size(0)
        
        q_len = query.size(1)
        k_len = key.size(1)
        v_len = value.size(1)

        query_lin_reshaped = query_lin.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)


        key_lin_reshaped = key_lin.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)


        value_lin_reshaped = value_lin.view(batch_size, v_len, self.num_heads, self.head_dim).transpose(1, 2)


        #3. Matrix multiplication of Q and K^T
        key_out = torch.matmul(query_lin_reshaped, key_lin_reshaped.transpose(-2, -1))


        # 4. optional mask step
        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -1e20)

        # 5. softmax on key_out / squared root of head_dim
        scaled_scores = key_out / math.sqrt(self.head_dim)
        #print("scaled_scores shape:", scaled_scores.shape)
        attn = torch.softmax(scaled_scores, dim=-1)
        #print("attn shape:", attn.shape)

        # 6 matrix multiplication of attn and V
        attn_output = torch.matmul(attn, value_lin_reshaped)
        #print("attn_output shape:", attn_output.shape)

        # 7. reshape back to original shape
        attn_output_reshaped = attn_output.view(batch_size, q_len, self.num_heads * self.head_dim)

        # 8. Send through final output linear layer
        output = self.output_layer(attn_output_reshaped)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()

        # TODO
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.forward_dim = forward_dim
        self.layernorm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.FFN = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )

        # intitialize multi-head attention block
        self.MHA_BLOCK = MultiHeadAttention(emb_dim, num_heads)


    def forward(self, query, key, value, mask):
        # TODO
        # 1. Run QKV through multi-head attention block
        mha_output = self.MHA_BLOCK.forward(query, key, value, mask)

        #2. add & dropout & layer norm
        added = mha_output + query
        dropped = F.dropout(added, p=self.dropout, training=self.training)
        normed1 = self.layernorm1(dropped)

        #3. feed through FFN
        ffn_output = self.FFN(normed1)

        #4. add & dropout & layer norm again
        added2 = ffn_output + normed1
        dropped2 = F.dropout(added2, p=self.dropout, training=self.training)
        normed2 = self.layernorm2(dropped2)

        # 5. return output
        output = normed2
        return output

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

        # TODO
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.max_len = max_len

        #1. embeddings and positional encodings
        self.token_level_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.position_encodings = nn.Embedding.from_pretrained(get_sinusoid_table(max_len, emb_dim), freeze=True)

        #2. dropout and transformer blocks
        self.dropout_layer = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(TransformerBlock(emb_dim, num_heads, dropout, forward_dim) for _ in range(num_layers))
        

    def forward(self, x, mask):
        # TODO
        # init index position matrix with +1 right shift
        index_positions = torch.arange(1, x.size(1) + 1).unsqueeze(0).repeat(x.size(0), 1)
        x = self.token_level_embeddings(x) + self.position_encodings(index_positions)
        x = self.dropout_layer(x)

        # running through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, x, x, mask) #REMEMBER: Q = K = V = x
        
        output = x
        return output


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()

        # TODO
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.forward_dim = forward_dim

        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.MHA = MultiHeadAttention(emb_dim, num_heads)
        self.transformer_block = TransformerBlock(emb_dim, num_heads, dropout, forward_dim)

    def forward(self, x, value, key, src_mask, tgt_mask):
        # TODO

        #1. maked multi-head attention
        out_mHA = self.MHA(x, x, x, tgt_mask)

        #2. residual connection, dropout, layer norm
    
        x = x + self.dropout(out_mHA)
        x = self.layernorm(x)

        # rename for clarity
        query = x

        #3. transformer block
        output = self.transformer_block.forward(query, key, value, src_mask)

        #4. return output
        return output

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

        # TODO
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decoderblocks = nn.ModuleList(DecoderBlock(emb_dim, num_heads, forward_dim, dropout)
                                            for _ in range(num_layers))

        self.postional_encodings = nn.Embedding(max_len, emb_dim)

        self.linear_output = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        # TODO
        # init index position matrix 
        index_positions = torch.arange(0, x.size(1)).unsqueeze(0).repeat(x.size(0), 1)

        #1. embedding + positional encodings, apply dropout
        x = self.token_embedding(x) + self.postional_encodings(index_positions)
        x = self.dropout(x)

        #2. move inputs to the batched sequence
        for decoderblock in self.decoderblocks:
            x = decoderblock.forward(x, encoder_out, encoder_out, src_mask, tgt_mask)
        
        # 3. final linear layer, projecting to vocab size
        output = self.linear_output(x)
        
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()

        # TODO
        self.endcoder = Encoder(
            vocab_size=src_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_len
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_len
        )
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx


    def create_src_mask(self, src):
        device = src.device
        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len), device=device)).expand(
            batch_size, 1, tgt_len, tgt_len
        )
        return tgt_mask.to(device)

    def forward(self, src, tgt):
        # TODO
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        # encode src sequence
        encoder_out = self.endcoder.forward(src, src_mask)

        # decode tgt sequence
        decoder_out = self.decoder.forward(tgt, encoder_out, src_mask, tgt_mask)

        return decoder_out