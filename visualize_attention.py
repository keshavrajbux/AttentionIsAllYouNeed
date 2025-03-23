import torch
import matplotlib.pyplot as plt
import numpy as np
from transformer import Transformer

def get_attention_weights(model, src, tgt, layer_idx=0, head_idx=0):
    """
    Extract attention weights from a specific layer and head of the Transformer model.
    
    Args:
        model: A trained Transformer model
        src: Source tokens [batch_size, src_len]
        tgt: Target tokens [batch_size, tgt_len]
        layer_idx: Index of the layer to visualize (default: 0)
        head_idx: Index of the attention head to visualize (default: 0)
        
    Returns:
        Tuple of (encoder_self_attn, decoder_self_attn, decoder_cross_attn)
    """
    # This function assumes the model has been modified to return attention weights
    # For demonstration purposes, we'll show how to modify the model to return attention weights
    # In a real implementation, you would need to modify the model classes to store and return the weights
    
    # Mock attention weights for demonstration
    batch_size = src.size(0)
    src_len = src.size(1)
    tgt_len = tgt.size(1)
    
    # Encoder self-attention: [batch_size, num_heads, src_len, src_len]
    encoder_self_attn = torch.randn(batch_size, model.encoder.layers[layer_idx].self_attn.num_heads, src_len, src_len)
    encoder_self_attn = torch.softmax(encoder_self_attn, dim=-1)
    
    # Decoder self-attention: [batch_size, num_heads, tgt_len, tgt_len]
    decoder_self_attn = torch.randn(batch_size, model.decoder.layers[layer_idx].self_attn.num_heads, tgt_len, tgt_len)
    decoder_self_attn = torch.softmax(decoder_self_attn, dim=-1)
    
    # Decoder cross-attention: [batch_size, num_heads, tgt_len, src_len]
    decoder_cross_attn = torch.randn(batch_size, model.decoder.layers[layer_idx].cross_attn.num_heads, tgt_len, src_len)
    decoder_cross_attn = torch.softmax(decoder_cross_attn, dim=-1)
    
    # Extract the specified head
    encoder_self_attn = encoder_self_attn[:, head_idx, :, :]
    decoder_self_attn = decoder_self_attn[:, head_idx, :, :]
    decoder_cross_attn = decoder_cross_attn[:, head_idx, :, :]
    
    return encoder_self_attn, decoder_self_attn, decoder_cross_attn

def plot_attention_weights(weights, x_labels=None, y_labels=None, title=None):
    """
    Plot attention weights as a heatmap.
    
    Args:
        weights: Attention weights [batch_size, seq_len_q, seq_len_k]
        x_labels: Labels for the x-axis (default: None)
        y_labels: Labels for the y-axis (default: None)
        title: Title of the plot (default: None)
    """
    # Take the first example in the batch
    weights = weights[0].cpu().detach().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(weights, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90)
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Loop over data dimensions and create text annotations
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            text = ax.text(j, i, f"{weights[i, j]:.2f}",
                          ha="center", va="center", color="white" if weights[i, j] < 0.5 else "black")
    
    plt.tight_layout()
    plt.show()

def visualize_all_attention_patterns(model, src, tgt, src_tokens=None, tgt_tokens=None):
    """
    Visualize all types of attention patterns in the Transformer model.
    
    Args:
        model: A trained Transformer model
        src: Source tokens [batch_size, src_len]
        tgt: Target tokens [batch_size, tgt_len]
        src_tokens: Human-readable source tokens (default: None)
        tgt_tokens: Human-readable target tokens (default: None)
    """
    # Get attention weights
    encoder_self_attn, decoder_self_attn, decoder_cross_attn = get_attention_weights(model, src, tgt)
    
    # If token labels are not provided, use indices
    if src_tokens is None:
        src_tokens = [f"src_{i}" for i in range(src.size(1))]
    if tgt_tokens is None:
        tgt_tokens = [f"tgt_{i}" for i in range(tgt.size(1))]
    
    # Plot encoder self-attention
    plot_attention_weights(
        encoder_self_attn,
        x_labels=src_tokens,
        y_labels=src_tokens,
        title="Encoder Self-Attention"
    )
    
    # Plot decoder self-attention
    plot_attention_weights(
        decoder_self_attn,
        x_labels=tgt_tokens,
        y_labels=tgt_tokens,
        title="Decoder Self-Attention (Masked)"
    )
    
    # Plot decoder cross-attention
    plot_attention_weights(
        decoder_cross_attn,
        x_labels=src_tokens,
        y_labels=tgt_tokens,
        title="Decoder Cross-Attention"
    )

if __name__ == "__main__":
    # Example usage
    print("To use this script, you need to:")
    print("1. Train your Transformer model")
    print("2. Modify the model to store attention weights during forward pass")
    print("3. Call visualize_all_attention_patterns with your model and data")
    
    # Loading a pretrained model (for demonstration)
    print("\nExample code to visualize attention weights:")
    print("""
    # Load trained model
    model = Transformer(src_vocab_size, tgt_vocab_size)
    model.load_state_dict(torch.load('transformer_model.pt'))
    
    # Prepare a sample input
    src = torch.tensor([[1, 2, 3, 4, 5]])  # Example source sequence
    tgt = torch.tensor([[5, 4, 3, 2, 1]])  # Example target sequence
    
    # Visualize attention weights
    visualize_all_attention_patterns(model, src, tgt)
    """) 