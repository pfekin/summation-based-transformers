# Linear-Complexity Sequence Modelling with Constraint-Driven Emergence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository implements **direct summation**, a linear-complexity alternative to self-attention that achieves competitive performance while dramatically reducing computational complexity from O(n²) to O(n).

Instead of computing pairwise token interactions, direct summation aggregates embeddings that have been modulated by learnable positional encodings and projected through ReLU transformations. This constraint-driven approach forces representations to self-organize without explicit attention mechanisms.

## Key Features

- **Linear complexity**: O(n) scaling vs. O(n²) for attention
- **Competitive performance** language modeling, across classification, and multimodal regression
- **Unified architecture** works across different modalities and tasks
- **Drop-in** replacement for self-attention, requiring no changes to the overall transformer architecture

## Quick Start

```bash
git clone https://github.com/pfekin/representational-superposition
cd representational-superposition
pip install -r requirements.txt

# Run language modeling benchmark  
python causal.py

# Run classification benchmark
python classifier.py

# Run multimodal regression benchmark
python multimodal.py
```

### Google Colab
```python
!pip install --upgrade datasets fsspec huggingface_hub
!pip install git+https://github.com/pfekin/representational-superposition
```

## Algorithm 

### Direct Summation

```python
function summation(tokens, d_model, pos_bias=1):
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
**Complexity: O(n·d)**

## Experimental Results

### Classification Tasks

| Dataset | Attention Val Acc | Superposition Val Acc | Speedup |
|---------|-------------------|------------------------|---------|
| IMDB | 0.87 | **0.88** | 6 - 18× |
| 20 Newsgroups | 0.58 | **0.65** | 6 - 18× |
| AG News | **0.91** | **0.91** | 6 - 18× |
| Reuters-21578 | 0.77 | **0.81** | 6 - 18× |

### Language Modeling Tasks

| Dataset | Validation Metric | Attention | Superposition | Hybrid |
|---------| ---------------------- |----------------------------|----------------------------|----------------------------|
| IMDB | Perplexity | 150 | 198 | **145** |
|      | Accuracy | 0.22 | 0.18 | **0.22** |
| AG News | Perplexity | **64** | 79 | 66 |
|         | Accuracy | **0.34** | 0.32 | **0.34** |
| WikiText-2 | Perplexity | 300 | 331 | **274** |
|            | Accuracy | 0.22 | 0.21 | **0.23** |
| CMU Book Summaries | Perplexity | 286 | 335 | **269** |
|                    | Accuracy | 0.18 | 0.17 | **0.19** |

### Multimodal Regression

| Model | MAE | R² | Speedup |
|-------|-----|-----|---------|
| Attention (Concatenation) | 0.0314 | 0.2599 | 1x |
| Superposition | **0.0289** | **0.3576** | 2 - 15x |

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

Constraint-driven emergence is grounded in principles of functional self-organization, where structure
and function co-emerge from system dynamics rather than predetermined design.

This summation-based approach rests on three necessary conditions:

**Constraint without Predetermined Pathways**: The summation operation eliminates dedicated channels for specific features or token relationships.

**Sufficient Representational Capacity**: The summation operation creates a representational bottleneck that forces the optimization process to discover encodings where
task-relevant information survives aggregation.

**Optimization Pressure as Feedback**: Gradient descent provides the necessary
feedback that gives representations meaning. 

For further details, see the paper: **"Linear-Time Sequence Modelling with Constraint-Driven Emergence"**

## Citation

If you use this code, algorithms, or ideas from this project in your research, please cite the work:

```bibtex
@article{Constraint_Driven_Emergence_2025,
  title={Linear-Time Sequence Modelling with Constraint-Driven Emergence},
  author={Pascal Ekin},
  journal={TechRxiv},  
  year={2025},
  doi={10.36227/techrxiv.12345678},  
  url={https://doi.org/10.36227/techrxiv.12345678},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Under the MIT License, you retain copyright ownership while allowing others to use, modify, and distribute your code with proper attribution.

## Contact

- **Author**: Pascal Ekin
- **Email**: pfekin@gmail.com 
- **Paper:** [Download from TechRxiv](https://doi.org/10.36227/techrxiv.12345678)  
- **Issues**: Please use the GitHub issue tracker for bug reports and feature requests
