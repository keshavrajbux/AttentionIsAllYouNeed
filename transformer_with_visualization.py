import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax normalization
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Store attention weights for visualization
        self.attention_weights = attn_probs
        
        # Apply attention to values
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads
        output = self.combine_heads(attn_output)
        
        # Final linear layer
        output = self.W_o(output)
        
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention with residual connection and normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self attention with residual connection and normalization
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross attention with residual connection and normalization
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        x = self.dropout(self.positional_encoding(x))
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.dropout(self.positional_encoding(x))
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x

class TransformerWithVisualization(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, dropout=0.1):
        super(TransformerWithVisualization, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self.return_attention_weights = False
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed source and target sequences
        src_embedded = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)
        
        # Pass through encoder and decoder
        enc_output = self.encoder(src_embedded, src_mask)
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        
        # Project to vocabulary size
        output = self.final_layer(dec_output)
        
        if self.return_attention_weights:
            # Collect attention weights from all layers
            enc_self_attention = [layer.self_attn.attention_weights for layer in self.encoder.layers]
            dec_self_attention = [layer.self_attn.attention_weights for layer in self.decoder.layers]
            dec_cross_attention = [layer.cross_attn.attention_weights for layer in self.decoder.layers]
            
            return output, {
                'encoder_self_attention': enc_self_attention,
                'decoder_self_attention': dec_self_attention,
                'decoder_cross_attention': dec_cross_attention
            }
        
        return output
    
    def set_return_attention_weights(self, value):
        """Set whether to return attention weights or not."""
        self.return_attention_weights = value
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def get_attention_weights(model, src, tgt, layer_idx=0):
    """
    Extract attention weights from a specific layer of the Transformer model.
    
    Args:
        model: A trained TransformerWithVisualization model
        src: Source tokens [batch_size, src_len]
        tgt: Target tokens [batch_size, tgt_len]
        layer_idx: Index of the layer to visualize (default: 0)
        
    Returns:
        Dictionary containing the attention weights
    """
    # Enable attention weight collection
    model.set_return_attention_weights(True)
    
    # Create masks
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).unsqueeze(0)
    
    # Forward pass to collect attention weights
    _, attention_weights = model(src, tgt, src_mask, tgt_mask)
    
    # Disable attention weight collection
    model.set_return_attention_weights(False)
    
    # Extract weights from the specified layer
    enc_self_attn = attention_weights['encoder_self_attention'][layer_idx]
    dec_self_attn = attention_weights['decoder_self_attention'][layer_idx]
    dec_cross_attn = attention_weights['decoder_cross_attention'][layer_idx]
    
    return {
        'encoder_self_attention': enc_self_attn,
        'decoder_self_attention': dec_self_attn,
        'decoder_cross_attention': dec_cross_attn
    } 