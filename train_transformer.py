import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from transformer import Transformer

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1
batch_size = 32
num_epochs = 20
learning_rate = 0.0001

# Toy dataset for demonstration - simple sequence translation task
# We'll create a small synthetic dataset where the target is to reverse the source sequence
def create_dataset(vocab_size, seq_length, num_samples):
    # Generate random sequences
    src_data = torch.randint(1, vocab_size, (num_samples, seq_length))
    # Target is the reverse of the source
    tgt_data = torch.flip(src_data, dims=[1])
    return src_data, tgt_data

# Dataset parameters
vocab_size = 1000
seq_length = 20
num_train_samples = 10000
num_val_samples = 1000

# Create training and validation datasets
src_train, tgt_train = create_dataset(vocab_size, seq_length, num_train_samples)
src_val, tgt_val = create_dataset(vocab_size, seq_length, num_val_samples)

# Create data loaders
def create_batches(src, tgt, batch_size):
    num_samples = src.size(0)
    indices = torch.randperm(num_samples)
    src = src[indices]
    tgt = tgt[indices]
    
    for i in range(0, num_samples, batch_size):
        if i + batch_size <= num_samples:
            yield src[i:i+batch_size], tgt[i:i+batch_size]

# Initialize model
model = Transformer(
    src_vocab_size=vocab_size+1,  # +1 for padding token
    tgt_vocab_size=vocab_size+1,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    dropout=dropout
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler (as described in the paper)
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

scheduler = NoamScheduler(optimizer, d_model)

# Function to create masks
def create_masks(src, tgt):
    # Create padding mask for source sequence
    src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
    # Create padding and look-ahead mask for target sequence
    tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)
    
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask
    
    return src_padding_mask, tgt_mask

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Training Transformer model on {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for src_batch, tgt_batch in create_batches(src_train, tgt_train, batch_size):
        # Prepare input and output
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        
        # Create target input and output (shift right)
        tgt_input = torch.zeros_like(tgt_batch)
        tgt_input[:, 1:] = tgt_batch[:, :-1]  # Shift right
        tgt_output = tgt_batch
        
        # Create masks
        src_mask, tgt_mask = create_masks(src_batch, tgt_input)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src_batch, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss
        loss = criterion(output.contiguous().view(-1, vocab_size+1), tgt_output.contiguous().view(-1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track loss
        total_loss += loss.item()
        batch_count += 1
        
        if batch_count % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    val_batch_count = 0
    
    with torch.no_grad():
        for src_batch, tgt_batch in create_batches(src_val, tgt_val, batch_size):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            tgt_input = torch.zeros_like(tgt_batch)
            tgt_input[:, 1:] = tgt_batch[:, :-1]
            
            src_mask, tgt_mask = create_masks(src_batch, tgt_input)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            
            output = model(src_batch, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.contiguous().view(-1, vocab_size+1), tgt_batch.contiguous().view(-1))
            
            val_loss += loss.item()
            val_batch_count += 1
    
    avg_train_loss = total_loss / batch_count
    avg_val_loss = val_loss / val_batch_count
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print("Training complete!")

# Save the model
torch.save(model.state_dict(), "transformer_model.pt")

# Test with a sample
def translate(model, src, max_len=50):
    model.eval()
    src = src.to(device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    
    # Start with a tensor containing only zeros (with first token being 1)
    output = torch.zeros(src.size(0), max_len, dtype=torch.long).to(device)
    output[:, 0] = 1  # Start token
    
    for i in range(1, max_len):
        tgt_mask = model.generate_square_subsequent_mask(i).to(device)
        
        with torch.no_grad():
            out = model(src, output[:, :i], src_mask, tgt_mask)
            # Get the most likely next word
            _, next_word = torch.max(out[:, -1], dim=1)
            output[:, i] = next_word
            
            # Stop if all sequences have end token
            if (next_word == 2).all():  # Assuming 2 is the end token
                break
    
    return output

# Test translation
test_src = src_val[:1]  # Take one sample
print("Source sequence:", test_src)
translated = translate(model, test_src)
print("Translated sequence:", translated)

# Check if it's correctly reversed
print("Expected translation:", tgt_val[:1]) 