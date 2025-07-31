# Representational Superposition: A Sub-Quadratic Alternative to Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository implements **Representational Superposition**, a linear-time alternative to self-attention that achieves competitive performance while dramatically reducing computational complexity from O(n²) to O(n).

Instead of computing pairwise token interactions, representational superposition uses direct summation of positionally-modulated embeddings in a shared high-dimensional space. This constraint-driven approach forces representations to self-organize without explicit attention mechanisms.

## Key Features

- **Linear complexity**: O(n) scaling vs. O(n²) for attention
- **6-14× faster training** on classification tasks
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
function superposition(tokens, d_model):
    n = length(tokens)
    
    # Embed tokens
    X = embed(tokens)  # [n, d_model]
    
    # Positional modulation
    pos_enc = sinusoidal_encoding(n, d_model)
    X_pos = X * pos_enc  # Element-wise multiplication
    
    # Bias-free projection
    X_proj = relu(X_pos @ W1)  # No bias term
    
    # Direct summation O(n)
    pooled = sum(X_proj, axis=0)  # [d_model]
    
    return pooled
```
**Time Complexity: O(n·d)**

## Experimental Results

### Classification Tasks

| Dataset | Attention Accuracy | Superposition Accuracy | Speedup |
|---------|-------------------|------------------------|---------|
| IMDB | 0.87 | **0.88** | 6 - 14× |
| 20 Newsgroups | 0.58 | **0.65** | 6 - 14× |
| AG News | 0.91 | 0.91 | 6 - 14× |
| Reuters-21578 | 0.77 | **0.81** | 6 - 14× |

### Language Modeling Tasks

| Dataset | Attention Val Acc (PPL) | Superposition Val Acc (PPL) | Hybrid Val Acc (PPL) |
|---------|------------------------|----------------------------|---------------------|
| IMDB | 0.1769   (251) | 0.1664   (259) | **0.1884   (220)** |
| AG News | 0.2122   (514) | 0.2162   (500) | **0.2251   (466)** |
| WikiText-2 | 0.1874  (603) | 0.1858   (582) | **0.1935   (554)** |
| CMU Book Summaries | 0.1620    (391) | 0.1513   (401) | **0.1680   (351)** |

### Multimodal Regression

| Model | MAE | R² |
|-------|-----|-----|
| Ridge Regression (Tabular Only) | 0.03 | 0.25 |
| Attention (Concatenation) | 0.03 | 0.26 |
| Superposition | **0.03** | **0.36** |

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

For detailed theoretical analysis, see the paper: **"Representational Superposition: A Sub-Quadratic Alternative to Attention"**

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
