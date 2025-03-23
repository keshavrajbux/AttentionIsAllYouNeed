"""
Simple example showing how to use the Transformer for a sequence reversal task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer import Transformer

# Set random seed for reproducibility
torch.manual_seed(42)

# Simple vocabulary for demonstration
vocab = list("abcdefghijklmnopqrstuvwxyz")
char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}  # +1 because 0 is reserved for padding
idx_to_char = {idx + 1: char for idx, char in enumerate(vocab)}
idx_to_char[0] = '<pad>'  # Padding token

# Parameters
seq_length = 10
vocab_size = len(vocab) + 1  # +1 for padding
d_model = 128
num_heads = 4
num_layers = 2
d_ff = 512
batch_size = 64
num_epochs = 100

# Create synthetic dataset: the task is to reverse sequences
def create_dataset(num_samples):
    data = []
    for _ in range(num_samples):
        # Generate random sequence of characters
        seq_len = torch.randint(3, seq_length + 1, (1,)).item()
        chars = [vocab[i] for i in torch.randint(0, len(vocab), (seq_len,))]
        
        # Input: original sequence
        src = [char_to_idx[c] for c in chars]
        
        # Target: reversed sequence
        tgt = [char_to_idx[c] for c in reversed(chars)]
        
        # Pad sequences
        src = src + [0] * (seq_length - len(src))
        tgt = tgt + [0] * (seq_length - len(tgt))
        
        data.append((src, tgt))
    return data

# Create train and validation datasets
train_data = create_dataset(1000)
val_data = create_dataset(100)

# Create data loaders
def create_batches(data, batch_size):
    indices = torch.randperm(len(data))
    for i in range(0, len(data), batch_size):
        if i + batch_size <= len(data):
            batch_indices = indices[i:i+batch_size]
            src_batch = [data[j][0] for j in batch_indices]
            tgt_batch = [data[j][1] for j in batch_indices]
            yield torch.tensor(src_batch), torch.tensor(tgt_batch)

# Initialize model
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    dropout=0.1
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    batch_count = 0
    
    for src_batch, tgt_batch in create_batches(train_data, batch_size):
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        
        # Create target input (shifted right) and target output
        tgt_input = torch.zeros_like(tgt_batch)
        tgt_input[:, :-1] = tgt_batch[:, 1:]  # Shift right
        tgt_output = tgt_batch
        
        # Create masks
        src_mask = (src_batch != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = model.generate_square_subsequent_mask(tgt_batch.size(1)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src_batch, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss
        output_flat = output.contiguous().view(-1, vocab_size)
        tgt_output_flat = tgt_output.contiguous().view(-1)
        loss = criterion(output_flat, tgt_output_flat)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        batch_count += 1
    
    avg_train_loss = total_train_loss / batch_count
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    total_val_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for src_batch, tgt_batch in create_batches(val_data, batch_size):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            tgt_input = torch.zeros_like(tgt_batch)
            tgt_input[:, :-1] = tgt_batch[:, 1:]
            
            src_mask = (src_batch != 0).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_batch.size(1)).to(device)
            
            output = model(src_batch, tgt_input, src_mask, tgt_mask)
            
            output_flat = output.contiguous().view(-1, vocab_size)
            tgt_output_flat = tgt_batch.contiguous().view(-1)
            loss = criterion(output_flat, tgt_output_flat)
            
            total_val_loss += loss.item()
            batch_count += 1
    
    avg_val_loss = total_val_loss / batch_count
    val_losses.append(avg_val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')
plt.close()

# Test with a few examples
def test_model(model, input_text):
    # Convert input text to tensor
    input_chars = list(input_text.lower())
    input_indices = [char_to_idx.get(c, 0) for c in input_chars]
    
    # Pad if necessary
    if len(input_indices) < seq_length:
        input_indices = input_indices + [0] * (seq_length - len(input_indices))
    elif len(input_indices) > seq_length:
        input_indices = input_indices[:seq_length]
    
    input_tensor = torch.tensor([input_indices]).to(device)
    
    # Create source mask
    src_mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2)
    
    # Start with a tensor containing only zeros (with first token being 1)
    output = torch.zeros(1, seq_length, dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        for i in range(seq_length):
            tgt_mask = model.generate_square_subsequent_mask(i+1).to(device)
            out = model(input_tensor, output[:, :i+1], src_mask, tgt_mask)
            _, next_word = torch.max(out[:, -1], dim=1)
            output[:, i] = next_word
            
            # Stop if padding token is predicted
            if next_word.item() == 0:
                break
    
    # Convert output to text
    output_indices = output[0].cpu().numpy()
    output_chars = [idx_to_char.get(idx, '') for idx in output_indices if idx > 0]
    return ''.join(output_chars)

# Test examples
test_cases = ["hello", "transformer", "python", "attention"]
print("\nTesting the model with examples:")
for text in test_cases:
    reversed_text = text[::-1]  # Correct reversed text
    model_output = test_model(model, text)
    print(f"Input: {text}, Expected: {reversed_text}, Model Output: {model_output}")

# Save the model
torch.save(model.state_dict(), "simple_transformer.pt")
print("\nModel saved as 'simple_transformer.pt'")

print("\nExample completed! Check 'loss_curves.png' for the training and validation loss curves.") 