# Transformer Implementation

This is a PyTorch implementation of the Transformer architecture as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Overview

The Transformer is a neural network architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions. It has become the foundation for many state-of-the-art models in natural language processing, such as BERT, GPT, and T5.

This implementation includes:

- Multi-Head Attention
- Positional Encoding
- Encoder and Decoder Stacks
- Position-wise Feed-Forward Networks
- Learning Rate Scheduler with Warmup
- Masking for Padding and Future Tokens

## Files

- `transformer.py`: The core Transformer implementation
- `train_transformer.py`: A script to train the Transformer on a toy task (sequence reversal)
- `transformer_with_visualization.py`: An extended version that captures attention weights for visualization
- `visualize_attention.py`: Utilities for visualizing attention patterns
- `train_and_visualize.py`: A script that trains a small Transformer and visualizes its attention patterns
- `machine_translation_example.py`: A simple machine translation example using the Transformer

## Usage

### Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

For the machine translation example, you'll also need to download the spacy language models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Training

To train the Transformer on a toy sequence reversal task:

```bash
python train_transformer.py
```

### Visualizing Attention

To train a small Transformer and visualize its attention patterns:

```bash
python train_and_visualize.py
```

### Machine Translation

To run the machine translation example:

```bash
python machine_translation_example.py
```

## Implementation Details

The implementation follows the paper closely:

- **Encoder**: 6 identical layers, each with:
  - Multi-head self-attention mechanism
  - Position-wise feed-forward network
  - Residual connections and layer normalization

- **Decoder**: 6 identical layers, each with:
  - Masked multi-head self-attention mechanism
  - Multi-head attention over encoder output
  - Position-wise feed-forward network
  - Residual connections and layer normalization

- **Hyperparameters**:
  - Model dimension (d_model): 512
  - Number of attention heads: 8
  - Feed-forward dimension: 2048
  - Number of encoder/decoder layers: 6
  - Dropout rate: 0.1

## Understanding the Paper

Key concepts from the "Attention Is All You Need" paper:

1. **Self-Attention**: Allows the model to focus on different parts of the input sequence when encoding each position.

2. **Multi-Head Attention**: Splits the attention mechanism into multiple heads, allowing the model to jointly attend to information from different representation subspaces.

3. **Positional Encoding**: Since the Transformer doesn't use recurrence, positional encodings are added to embed the position information of tokens in the sequence.

4. **Scaled Dot-Product Attention**: Attention function used in the paper, defined as:
   ```
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
   ```

5. **Residual Connections and Normalization**: Used to aid in training deeper networks.

## Applications

The Transformer architecture can be used for various natural language processing tasks:

1. **Machine Translation**: Translating text from one language to another
2. **Text Summarization**: Generating concise summaries of longer texts
3. **Question Answering**: Providing answers to questions based on a given context
4. **Text Generation**: Creating coherent text based on a prompt or context

## Citations

```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
``` 