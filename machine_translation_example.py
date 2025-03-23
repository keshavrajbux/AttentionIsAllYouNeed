import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from transformer import Transformer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# This is a simplified example to demonstrate machine translation
# In a real scenario, you would use actual language datasets

# Example sentences (English -> German)
en_sentences = [
    "Hello, how are you?",
    "I am a student.",
    "Where is the library?",
    "The book is on the table.",
    "He likes to play football.",
    "She is reading a book.",
    "Today is a beautiful day.",
    "What time is it?",
    "The cat is sleeping on the sofa.",
    "I would like a cup of coffee, please."
]

de_sentences = [
    "Hallo, wie geht es dir?",
    "Ich bin ein Student.",
    "Wo ist die Bibliothek?",
    "Das Buch liegt auf dem Tisch.",
    "Er spielt gerne Fußball.",
    "Sie liest ein Buch.",
    "Heute ist ein schöner Tag.",
    "Wie spät ist es?",
    "Die Katze schläft auf dem Sofa.",
    "Ich hätte gerne eine Tasse Kaffee, bitte."
]

# Tokenizers
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

# Build vocabularies
def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text.lower())

# Special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Build vocabularies
en_vocab = build_vocab_from_iterator(
    yield_tokens(en_sentences, en_tokenizer),
    min_freq=1,
    specials=special_symbols,
    special_first=True
)

de_vocab = build_vocab_from_iterator(
    yield_tokens(de_sentences, de_tokenizer),
    min_freq=1,
    specials=special_symbols,
    special_first=True
)

en_vocab.set_default_index(UNK_IDX)
de_vocab.set_default_index(UNK_IDX)

# Convert text to indices
def text_to_indices(text, tokenizer, vocab):
    tokens = tokenizer(text.lower())
    return [BOS_IDX] + [vocab[token] for token in tokens] + [EOS_IDX]

# Prepare dataset
train_data = []
for en, de in zip(en_sentences, de_sentences):
    train_data.append((text_to_indices(en, en_tokenizer, en_vocab), 
                      text_to_indices(de, de_tokenizer, de_vocab)))

# Create batch
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.tensor(src_sample))
        tgt_batch.append(torch.tensor(tgt_sample))
    
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    
    return src_batch, tgt_batch

# Model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_vocab_size = len(en_vocab)
tgt_vocab_size = len(de_vocab)
emb_size = 256
nhead = 4
ffn_hid_dim = 512
batch_size = 5
num_encoder_layers = 3
num_decoder_layers = 3

# Create model
transformer = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=emb_size,
    num_heads=nhead,
    d_ff=ffn_hid_dim,
    num_layers=num_encoder_layers
)

# Create DataLoader
train_iterator = DataLoader(
    train_data, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Optimizer
optimizer = torch.optim.Adam(
    transformer.parameters(), 
    lr=0.0001, 
    betas=(0.9, 0.98), 
    eps=1e-9
)

# Training function
def train_epoch(model, iterator, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    
    for src, tgt in iterator:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Create masks
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        # Compute loss
        output_flat = output.contiguous().view(-1, tgt_vocab_size)
        tgt_output_flat = tgt_output.contiguous().view(-1)
        loss = loss_fn(output_flat, tgt_output_flat)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Translation function
def translate(model, src_text, src_tokenizer, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    
    # Process source text
    src_indices = text_to_indices(src_text, src_tokenizer, src_vocab)
    src_tensor = torch.tensor([src_indices]).to(device)
    
    # Create source mask
    src_mask = (src_tensor != PAD_IDX).unsqueeze(1).unsqueeze(2)
    
    # Start with BOS token
    tgt_indices = [BOS_IDX]
    tgt_tensor = torch.tensor([tgt_indices]).to(device)
    
    for i in range(max_len):
        tgt_mask = model.generate_square_subsequent_mask(len(tgt_indices)).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
            
        # Get the next token prediction
        next_token = output.argmax(2)[:, -1].item()
        tgt_indices.append(next_token)
        
        # Update target tensor
        tgt_tensor = torch.tensor([tgt_indices]).to(device)
        
        # Stop if EOS token is predicted
        if next_token == EOS_IDX:
            break
    
    # Convert indices to tokens
    tgt_tokens = []
    for idx in tgt_indices:
        if idx in [BOS_IDX, EOS_IDX, PAD_IDX]:
            continue
        tgt_tokens.append(tgt_vocab.get_itos()[idx])
    
    return " ".join(tgt_tokens)

def main():
    print("Training machine translation model...")
    
    # Move model to device
    transformer.to(device)
    
    # Train for a few epochs
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_epoch(transformer, train_iterator, optimizer, loss_fn, device)
        print(f"Epoch: {epoch+1}, Loss: {train_loss:.4f}")
    
    # Save the model
    torch.save(transformer.state_dict(), "translation_model.pt")
    print("Model saved as translation_model.pt")
    
    # Test translation
    print("\nTesting translation:")
    for en_text in en_sentences[:3]:  # Test with first 3 sentences
        translation = translate(transformer, en_text, en_tokenizer, en_vocab, de_vocab, device)
        print(f"English: {en_text}")
        print(f"Translated: {translation}")
        print()

if __name__ == "__main__":
    # Note: You would need to install spacy and download language models first:
    # python -m spacy download en_core_web_sm
    # python -m spacy download de_core_news_sm
    print("Note: To run this script, you need to install spacy and its language models.")
    print("Run the following commands first:")
    print("pip install spacy")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download de_core_news_sm")
    print("\nThen run this script again.")
    
    # Uncomment the line below to run the main function after installing the requirements
    # main() 