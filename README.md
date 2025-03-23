# 🤖 Transformer Implementation
<div align="center">

[![GitHub](https://img.shields.io/github/license/keshavrajbux/AttentionIsAllYouNeed?color=blue)](https://github.com/keshavrajbux/AttentionIsAllYouNeed/blob/main/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png" width="600" alt="Transformer Architecture">
</p>

<p align="center">
  <i>A PyTorch implementation of the groundbreaking paper <a href="https://arxiv.org/abs/1706.03762">"Attention Is All You Need"</a> by Vaswani et al.</i>
</p>

---

## 📌 Overview

The Transformer is a revolutionary neural network architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions. It has become the foundation for many state-of-the-art models in natural language processing, such as BERT, GPT, and T5.

This implementation includes:

- ✨ **Multi-Head Attention** - Parallel attention layers for richer representations
- 📍 **Positional Encoding** - Injecting sequence order information
- 🧱 **Encoder and Decoder Stacks** - The core architecture components
- 🔄 **Position-wise Feed-Forward Networks** - Per-position feature transformation
- 📈 **Learning Rate Scheduler with Warmup** - Advanced optimization techniques
- 🎭 **Masking for Padding and Future Tokens** - Handling variable-length sequences

## 📁 Repository Structure

```bash
.
├── transformer.py                    # Core Transformer implementation
├── train_transformer.py              # Training script for sequence reversal task
├── transformer_with_visualization.py # Extended version with attention visualization
├── visualize_attention.py            # Utilities for attention visualization
├── train_and_visualize.py            # Script for training and visualizing attention
├── machine_translation_example.py    # Example of machine translation
└── requirements.txt                  # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

For the machine translation example, install the spaCy language models:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Training

Train the Transformer on a toy sequence reversal task:

```bash
python train_transformer.py
```

### Visualizing Attention

Train a small Transformer and visualize its attention patterns:

```bash
python train_and_visualize.py
```

<p align="center">
  <img src="https://miro.medium.com/max/1400/1*Ua5SyuN1rZjYJJH0Kua3Ow.png" width="500" alt="Attention Visualization">
</p>

### Machine Translation

Run the machine translation example:

```bash
python machine_translation_example.py
```

## 🔍 Implementation Details

The implementation follows the paper closely with configurable hyperparameters:

| Component | Default Value | Description |
|-----------|---------------|-------------|
| Model Dimension (d_model) | 512 | Size of embeddings throughout the model |
| Number of Attention Heads | 8 | Attention heads in multi-head attention |
| Feed-Forward Dimension | 2048 | Inner layer dimension of position-wise FFN |
| Number of Layers | 6 | Number of encoder/decoder blocks |
| Dropout Rate | 0.1 | Probability of dropping units for regularization |

## 🧠 Understanding the Transformer

<p align="center">
  <img src="https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png" width="400" alt="Multi-head Attention">
</p>

Key concepts from the "Attention Is All You Need" paper:

1. **Self-Attention** - Allows the model to focus on different parts of the input sequence when encoding each position.

2. **Multi-Head Attention** - Splits the attention mechanism into multiple heads, allowing the model to jointly attend to information from different representation subspaces.

3. **Positional Encoding** - Since the Transformer doesn't use recurrence, positional encodings are added to embed the position information of tokens in the sequence.

4. **Scaled Dot-Product Attention** - Attention function used in the paper, defined as:
   ```
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
   ```

5. **Residual Connections and Normalization** - Used to aid in training deeper networks.

## 🌐 Applications

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/LibreTranslate_logo.svg/1200px-LibreTranslate_logo.svg.png" height="100" alt="Translation">
  <img src="https://cdn-icons-png.flaticon.com/512/1183/1183672.png" height="100" alt="Summarization">
  <img src="https://cdn-icons-png.flaticon.com/512/4406/4406119.png" height="100" alt="Question Answering">
</p>

The Transformer architecture can be used for various natural language processing tasks:

1. **Machine Translation** - Translating text from one language to another
2. **Text Summarization** - Generating concise summaries of longer texts
3. **Question Answering** - Providing answers to questions based on a given context
4. **Text Generation** - Creating coherent text based on a prompt or context

## 📚 Citation

```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper authors
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) by Harvard NLP 