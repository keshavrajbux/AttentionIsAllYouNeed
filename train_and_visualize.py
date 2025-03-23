import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from transformer_with_visualization import TransformerWithVisualization, get_attention_weights

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1
batch_size = 32
num_epochs = 10
learning_rate = 0.0001

# For demonstration, let's use a very small dataset
vocab_size = 20
seq_length = 10
num_train_samples = 1000
num_val_samples = 100

# Create a simple sequence reversal dataset
def create_dataset(vocab_size, seq_length, num_samples):
    # Generate random sequences
    src_data = torch.randint(1, vocab_size, (num_samples, seq_length))
    # Target is the reverse of the source
    tgt_data = torch.flip(src_data, dims=[1])
    return src_data, tgt_data

# Create datasets
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

# Initialize model with visualization
model = TransformerWithVisualization(
    src_vocab_size=vocab_size+1,
    tgt_vocab_size=vocab_size+1,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    dropout=dropout
)

# Use a smaller model for faster training
small_model = TransformerWithVisualization(
    src_vocab_size=vocab_size+1,
    tgt_vocab_size=vocab_size+1,
    d_model=64,
    num_heads=4,
    d_ff=128,
    num_layers=2,
    dropout=dropout
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(small_model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

# Training function
def train_model(model, num_epochs, device="cpu"):
    model = model.to(device)
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for src_batch, tgt_batch in create_batches(src_train, tgt_train, batch_size):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Create target input and output
            tgt_input = torch.zeros_like(tgt_batch)
            tgt_input[:, 1:] = tgt_batch[:, :-1]
            tgt_output = tgt_batch
            
            # Create masks
            src_mask = (src_batch != 0).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src_batch, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(output.contiguous().view(-1, vocab_size+1), tgt_output.contiguous().view(-1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
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
                
                src_mask = (src_batch != 0).unsqueeze(1).unsqueeze(2).to(device)
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                
                output = model(src_batch, tgt_input, src_mask, tgt_mask)
                loss = criterion(output.contiguous().view(-1, vocab_size+1), tgt_batch.contiguous().view(-1))
                
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_train_loss = total_loss / batch_count
        avg_val_loss = val_loss / val_batch_count
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model

# Function to visualize attention weights
def visualize_attention(model, src, tgt, layer_idx=0):
    # Get attention weights
    attn_weights = get_attention_weights(model, src, tgt, layer_idx)
    
    # Create tokens for visualization
    src_tokens = [f"src_{i+1}" for i in range(src.size(1))]
    tgt_tokens = [f"tgt_{i+1}" for i in range(tgt.size(1))]
    
    # Define function to plot attention weights
    def plot_attention(weights, x_labels, y_labels, title):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(weights, cmap='viridis')
        
        # Set labels
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_yticklabels(y_labels)
        
        # Add colorbar and title
        plt.colorbar(im)
        plt.title(title)
        
        # Add values as text
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, f"{weights[i, j]:.2f}",
                               ha="center", va="center", color="white" if weights[i, j] > 0.5 else "black")
        
        plt.tight_layout()
        plt.show()
    
    # Extract weights from the first batch and head for visualization
    enc_self_attn = attn_weights['encoder_self_attention'][0, 0].cpu().numpy()
    dec_self_attn = attn_weights['decoder_self_attention'][0, 0].cpu().numpy()
    dec_cross_attn = attn_weights['decoder_cross_attention'][0, 0].cpu().numpy()
    
    # Plot the attention weights
    plot_attention(enc_self_attn, src_tokens, src_tokens, "Encoder Self-Attention")
    plot_attention(dec_self_attn, tgt_tokens, tgt_tokens, "Decoder Self-Attention")
    plot_attention(dec_cross_attn, src_tokens, tgt_tokens, "Decoder Cross-Attention")

# Main function
def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the small model
    print("Training the model...")
    trained_model = train_model(small_model, num_epochs, device)
    
    # Save the model
    torch.save(trained_model.state_dict(), "transformer_vis_model.pt")
    print("Model saved as transformer_vis_model.pt")
    
    # Test with a sample
    print("\nTesting with a sample...")
    test_src = src_val[:1].to(device)
    test_tgt = tgt_val[:1].to(device)
    
    print("Source sequence:", test_src.cpu().numpy()[0])
    print("Target sequence:", test_tgt.cpu().numpy()[0])
    
    # Visualize attention weights
    print("\nVisualizing attention weights...")
    visualize_attention(trained_model, test_src, test_tgt)
    
if __name__ == "__main__":
    main() 