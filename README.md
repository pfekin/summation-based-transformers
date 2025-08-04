# Representational Superposition: A Sub-Quadratic Alternative to Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository implements **Representational Superposition**, a linear-time alternative to self-attention that achieves competitive performance while dramatically reducing computational complexity from O(n²) to O(n).

Instead of computing pairwise token interactions, representational superposition uses direct summation of embeddings that have been modulated by learnable positional encodings and projected through a linear transformation. This constraint-driven approach forces representations to self-organize without explicit attention mechanisms.

## Key Features

- **Linear complexity**: O(n) scaling vs. O(n²) for attention
- **Competitive performance** across classification, language modeling, and multimodal regression
- **Unified architecture** works across different modalities and tasks
- **Simple implementation** with minimal dependencies

## Quick Start

```bash
git clone https://github.com/pfekin/representational-superposition
cd representational-superposition
pip install -r requirements.txt

# Run classification benchmark
python classifier_benchmark.py

# Run language modeling benchmark  
python causal_benchmark.py

# Run multimodal regression benchmark
python multimodal_benchmark.py
```

### Google Colab
```python
!pip install --upgrade datasets fsspec huggingface_hub
!pip install git+https://github.com/pfekin/representational-superposition
```

## Algorithm 

### Representational Superposition

```python
function superposition(tokens, d_model, pos_bias=1):
    n = length(tokens)
    
    # Embed tokens
    X = embed(tokens)  # [n, d_model]
    
    # Learned positional embeddings
    pos_enc = PositionalEmbedding(n, d_model) + pos_bias
    X_pos = X * pos_enc  # Element-wise multiplication
    
    # Bias-free projection
    X_proj = relu(X_pos @ W1)  # No bias term
    
    # Direct summation O(n)
    pooled = sum(X_proj, axis=0)  # [n, d_model]
    
    return pooled
```
**Time Complexity: O(n·d)**

## Experimental Results

### Classification Tasks

| Dataset | Attention Val Acc | Superposition Val Acc | Speedup |
|---------|-------------------|------------------------|---------|
| IMDB | 0.87 | **0.88** | 6 - 18× |
| 20 Newsgroups | 0.58 | **0.65** | 6 - 18× |
| AG News | 0.91 | 0.91 | 6 - 18× |
| Reuters-21578 | 0.77 | **0.81** | 6 - 18× |

### Language Modeling Tasks

| Dataset | Metric | Attention Val Acc (PPL) | Superposition Val Acc (PPL) | Hybrid Val Acc (PPL) |
|---------| ------------ |------------------------|----------------------------|---------------------|
| IMDB | Perplexity | 251 | 259 | **220** |
|      | Accuracy | 0.1769 | 0.1664 | **0.1884** |
| AG News | Perplexity | 514 | 500 | **466** |
|         | Accuracy | 0.2122 | 0.2162 | **0.2251** |
| WikiText-2 | Perplexity | 603 | 582 | **554** |
|            | Accuracy | 0.1874 | 0.1858 | **0.1935** |
| CMU Book Summaries | Perplexity | 391 | 401 | **351** |
|                    | Accuracy | 0.1620 | 0.1513 | **0.1680** |

### Multimodal Regression

| Model | MAE | R² |
|-------|-----|-----|
| Ridge Regression (Tabular Only) | 0.0301 | 0.2526 |
| Attention (Concatenation) | 0.0314 | 0.2599 |
| Superposition | **0.0289** | **0.3576** |

## Requirements

- Python 3.8+
- PyTorch 1.9+ and/or tensorflow[and-cuda]
- transformers
- scikit-learn
- numpy
- matplotlib
- datasets
- fsspec
- huggingface_hub

## Theoretical Foundation

Representational superposition is grounded in several key principles:

1. **Constraint-driven emergence**: Structure arises from optimization pressure within constrained spaces
2. **Signal-to-noise preservation**: High-dimensional summation amplifies signal while suppressing uncorrelated noise  
3. **Deferred resolution**: Ambiguity is preserved until final output layers
4. **Modality decoupling**: Forces abstraction by preventing reliance on modality-specific cues

For further details, see the paper: **"Representational Superposition: A Sub-Quadratic Alternative to Attention"**

## Citation

If you use this code, algorithms, or ideas from this project in your research, please cite the work:

```bibtex
@article{representational_superposition_2025,
  title={Representational Superposition: A Sub-Quadratic Alternative to Attention},
  author={Pascal Ekin},
  journal={*},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Under the MIT License, you retain copyright ownership while allowing others to use, modify, and distribute your code with proper attribution.

## Contact

- **Author**: Pascal Ekin
- **Email**: pfekin@gmail.com 
- **Paper**: [Link to paper when published]
- **Issues**: Please use the GitHub issue tracker for bug reports and feature requests
