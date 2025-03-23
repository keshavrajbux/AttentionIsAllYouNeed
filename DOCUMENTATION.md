# Transformer Implementation Documentation

This document provides detailed information about the implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Modules](#modules)
   - [Multi-Head Attention](#multi-head-attention)
   - [Positional Encoding](#positional-encoding)
   - [Position-wise Feed-Forward Networks](#position-wise-feed-forward-networks)
   - [Encoder and Decoder](#encoder-and-decoder)
3. [Usage Examples](#usage-examples)
   - [Basic Model Initialization](#basic-model-initialization)
   - [Training the Model](#training-the-model)
   - [Inference](#inference)
   - [Visualization](#visualization)
4. [Advanced Configuration](#advanced-configuration)
5. [Performance Tips](#performance-tips)

## Architecture Overview

The Transformer architecture is based entirely on attention mechanisms and does not rely on recurrence or convolutions. The architecture consists of an encoder-decoder structure:

![Transformer Architecture](https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png)

- The **encoder** transforms an input sequence into a continuous representation.
- The **decoder** generates an output sequence one token at a time, attending to the encoder's output and previously generated tokens.

## Modules

### Multi-Head Attention

The multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces at different positions.

```python
# Usage example
from transformer import MultiHeadAttention

# Initialize multi-head attention with 8 heads
attention = MultiHeadAttention(d_model=512, num_heads=8)

# Forward pass
output = attention(query, key, value, mask=None)
```

#### Parameters:

- `d_model` (int): Dimensionality of the model (must be divisible by `num_heads`)
- `num_heads` (int): Number of attention heads

### Positional Encoding

Since the Transformer does not use recurrence or convolution, it needs positional encodings to make use of the order of the sequence.

```python
# Usage example
from transformer import PositionalEncoding
import torch

# Initialize positional encoding for sequences up to length 1000
pos_encoding = PositionalEncoding(d_model=512, max_seq_length=1000)

# Apply to input embeddings
x = torch.randn(32, 50, 512)  # (batch_size, seq_length, d_model)
output = pos_encoding(x)
```

#### Parameters:

- `d_model` (int): Dimensionality of the model
- `max_seq_length` (int, optional): Maximum sequence length (default: 5000)

### Position-wise Feed-Forward Networks

Each position in the sequence gets processed through the same feed-forward network.

```python
# Usage example
from transformer import PositionwiseFeedForward

# Initialize feed-forward network
ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)

# Forward pass
output = ffn(x)
```

#### Parameters:

- `d_model` (int): Input and output dimensionality
- `d_ff` (int): Inner dimensionality of the feed-forward network

### Encoder and Decoder

The encoder consists of multiple identical layers, each containing a multi-head self-attention mechanism and a position-wise feed-forward network.

The decoder is similar but also includes a multi-head attention layer that attends to the encoder's output.

```python
# Usage example
from transformer import Encoder, Decoder

# Initialize encoder
encoder = Encoder(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
)

# Initialize decoder
decoder = Decoder(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
)

# Forward pass
encoder_output = encoder(src_embedded, src_mask)
decoder_output = decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
```

## Usage Examples

### Basic Model Initialization

```python
import torch
from transformer import Transformer

# Define vocabulary sizes
src_vocab_size = 10000
tgt_vocab_size = 10000

# Initialize Transformer
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Training the Model

```python
import torch.optim as optim
import torch.nn as nn

# Initialize loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding token
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Training loop
model.train()
for epoch in range(num_epochs):
    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Create target input (shifted right) and target output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create masks
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss
        output_flat = output.contiguous().view(-1, tgt_vocab_size)
        tgt_output_flat = tgt_output.contiguous().view(-1)
        loss = criterion(output_flat, tgt_output_flat)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print loss
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### Inference

```python
def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    
    memory = model.encoder(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    
    for i in range(max_len-1):
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.decoder(ys, memory, src_mask, tgt_mask)
        prob = model.final_layer(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word).to(device)], dim=1)
        
        if next_word == end_symbol:
            break
            
    return ys

# Example usage
src_sentence = "Hello, how are you?"
src_tokens = tokenize(src_sentence)
src_indices = [src_vocab[token] for token in src_tokens]
src_tensor = torch.tensor([src_indices]).to(device)

model.eval()
with torch.no_grad():
    output_indices = greedy_decode(model, src_tensor, max_len=50, start_symbol=2)
    
output_tokens = [tgt_vocab.get_itos()[idx] for idx in output_indices[0].tolist()]
output_sentence = detokenize(output_tokens)
print(f"Translation: {output_sentence}")
```

### Visualization

```python
from transformer_with_visualization import TransformerWithVisualization, get_attention_weights
import matplotlib.pyplot as plt
import numpy as np

# Initialize model with visualization
model = TransformerWithVisualization(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
)

# Train model...

# Get attention weights
attn_weights = get_attention_weights(model, src_tensor, tgt_tensor, layer_idx=0)

# Extract encoder self-attention weights for the first head
enc_self_attn = attn_weights['encoder_self_attention'][0, 0].cpu().numpy()

# Plot attention weights
plt.figure(figsize=(10, 8))
plt.imshow(enc_self_attn, cmap='viridis')
plt.colorbar()
plt.title("Encoder Self-Attention")
plt.xlabel("Source Tokens")
plt.ylabel("Source Tokens")
plt.show()
```

## Advanced Configuration

### Learning Rate Scheduler

The paper introduces a custom learning rate scheduler with warmup:

```python
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self.learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def learning_rate(self):
        return self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))

# Usage
scheduler = NoamScheduler(optimizer, d_model=512)
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Training steps...
        optimizer.step()
        scheduler.step()
```

### Label Smoothing

Label smoothing can improve model performance:

```python
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

# Usage
criterion = LabelSmoothing(size=tgt_vocab_size, padding_idx=0, smoothing=0.1)
```

## Performance Tips

1. **Batch Size**: Use the largest batch size that fits in your memory.
2. **Mixed Precision Training**: Use mixed precision (FP16) for faster training on compatible GPUs.
3. **Gradient Accumulation**: Accumulate gradients over multiple forward/backward passes to simulate larger batch sizes.
4. **Caching**: Cache the encoder output for faster inference in machine translation tasks.
5. **Beam Search**: Use beam search instead of greedy decoding for better translation quality.

```python
# Example of gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()

for i, (src, tgt) in enumerate(train_dataloader):
    # Forward pass
    output = model(src, tgt_input, src_mask, tgt_mask)
    
    # Calculate loss
    loss = criterion(output_flat, tgt_output_flat) / accumulation_steps
    
    # Backward pass
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

For more detailed information, refer to the original paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) and the code implementation in this repository. 