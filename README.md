
# Summation-Based Transformers
**A Path Toward Linear Complexity Sequence Modeling**

[![Paper](https://img.shields.io/badge/Paper-TechRxiv-blue)](https://doi.org/10.36227/techrxiv.175790522.25734653/v2)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Overview

This repository implements summation-based aggregation as an alternative to self-attention in transformers. Instead of computing pairwise similarities (O(n²d) cost), tokens are modulated by learned positional encodings, passed through projection layers, and aggregated by direct summation (O(nd) cost).

**Key finding:** Hybrid architectures - quadratic attention driving linear mechanisms - match or exceed full-attention performance while keeping computational complexity near-linear. In this implementation, summation occupies most layers with a single attention layer at the output.

**Drop-in replacement:** Summation can replace attention inside transformer blocks without altering residuals, norms, or optimizers - making it straightforward to integrate into existing architectures.

## Quick Start

```bash
git clone https://github.com/pfekin/summation-based-transformers
cd summation-based-transformers
pip install -r requirements.txt

# Run forecatsing benchmark
python patch_tst_all.py

# Run language modeling benchmark
python causal.py

```

## Core Mechanism: Modulated Summation

The implementation uses **modulated summation** as the primary variant:

- **Positional encoding:** Multiplied (x * pos_enc)
- **Projection:** Linear without bias
- **Activation:** GELU or ReLU
- **Works for:** Forecasting, language modeling

**Note:** While modulated summation works well across most tested domains, other configurations may be better suited for specific tasks. Variants using additive positional encodings, linear projections with bias, or no nonlinearity could provide better fits for noisy or sparse-signal data. The design space remains open for exploration.

## Results

### 1. Time Series Forecasting (Main Benchmark)

PatchTST-Hybrid architecture tested against [PatchTST (ICLR 2023, SOTA or near-SOTA)](https://github.com/yuqinie98/PatchTST).  
3-layer model (hidden=128, context=512, pred_len=96, patch_len=16, stride=8) comparing full attention with hybrid (2 summation + 1 attention). 
| Dataset | PatchTST MSE | PatchTST Hybrid MSE | Improvement | Speedup |
|---------|--------------|------------|-------------|---------|
| Weather   | 0.1607        | **0.1548**      | +3.67%       | x1.45    |
| Traffic | 0.3263        | **0.3206**      | +1.75%       | x1.38    |
| ETTh1   | 0.4450      | **0.4387**      | +1.42%       | x1.36    |
| ETTh2   | 0.2438        | **0.1941**      | +20.39%       | x1.37    |
| ETTm1   | 0.3704        | **0.3295**      | +11.04%       | x1.34   |
| ETTm2   | 0.1850        | **0.1751**      | +5.35.%       | x1.44    |


**Key result:** Hybrid summation-attention architecture achieves competitive or superior performance compared to current SOTA while being significantly faster.

### 2. Autoregressive Language Modeling

4-layer decoder models (hidden=512, context=512) comparing full attention, pure summation, and hybrid (3 summation + 1 attention):

| Dataset | Full Attention (PPL/Acc) | Pure Summation (PPL/Acc) | Hybrid (PPL/Acc) |
|---------|--------------------------|--------------------------|------------------|
| CMU Books | 286 / 0.19 | 335 / 0.17 | **269 / 0.19** |
| WikiText-2 | 300 / 0.22 | 331 / 0.21 | **274 / 0.23** |
| AG News | **64 / 0.35** | 79 / 0.32 | 66 / 0.35 |
| IMDB | 150 / 0.22 | 198 / 0.19 | **145 / 0.22** |

**Tested at 512 context length.** Scaling behavior at 32K+ tokens remains an open question.

### 3. Document Classification

Single summation layer before classifier achieves 4-18× speedup over attention while matching or exceeding accuracy on IMDB, 20 Newsgroups, AG News, and Reuters-21578 datasets.

### 4. Multimodal Regression

Civil Comments dataset with text + metadata fusion: summation through shared bottleneck (256-dim) matches or exceeds concatenation baseline (1280-dim), suggesting bottleneck constraints may encourage modality-agnostic representations.

## Representational Dynamics

Analysis reveals distinct embedding evolution patterns across layers:

![PCA Trajectories](media/pca.png)
*PCA trajectories of embeddings across layers. Summation restructures embeddings through contraction and expansion before final convergence, while attention shows gradual refinement.*

**Key observations:**
- **Attention:** Gradual refinement, stable manifold convergence
- **Summation:** Sharp restructuring, alternating dimensionality contraction-expansion
- **Hybrid:** Summation builds differentiated structure, final attention disambiguates

See [paper](https://doi.org/10.36227/techrxiv.175790522.25734653/v2) Section 6 for detailed analysis.

## Architecture & Integration

**Integration patterns:** Beyond the simple "summation-first, attention-last" pattern tested here, other configurations are possible:
- **Interleaving:** Alternate summation and attention layers
- **Hierarchical gating:** Apply attention at selected depths
- **Task-specific hybrids:** Use attention only in output heads or selected submodules

Further empirical work is needed to identify optimal patterns for different use cases.

## Why Hybrids Work: A Kernel Perspective

Attention can be viewed as a kernel method that computes similarity through dot products in feature space. Summation, by contrast, eliminates pairwise comparison entirely - it's not approximating a kernel but operating through a fundamentally different mechanism.

The hybrid architecture suggests that using simpler restructuring mechanisms, driven by attention, provides more representational flexibility. Rather than stacking multiple kernel-based operations that each compute weighted aggregations based on similarity, summation builds structure through constraint while attention provides precise disambiguation at the output.

This framing positions hybrid architectures as exploring a broader principle: quadratic attention driving linear mechanisms. Summation becomes an exercise in computational minimalism - aggregating through the simplest possible operation while maintaining competitive performance.

## Computational Complexity

| Method | Time | Memory |
|--------|------|--------|
| Full attention (L layers) | L·O(n²d) | L·O(n²) |
| Hybrid (L-1 sum + 1 att) | (L-1)·O(nd) + O(n²d) | (L-1)·O(nd) + O(n²) |
| Pure summation (L layers) | L·O(nd) | L·O(nd) |

## Future Work & Collaboration

**Open questions:**
- **Scaling to production contexts:** How does the approach perform at 32K+ token contexts and billion-parameter scales?
- **Optimal integration patterns:** Which combinations of summation and attention work best for specific domains?
- **Design space exploration:** What other non-kernel mechanisms could effectively replace intermediate attention layers?

**We welcome:**
- Collaboration with teams that have access to significant compute resources
- Feedback from anyone implementing and evaluating the approach
- Discussion of integration patterns and architectural variations

The current results demonstrate that hybrid architectures can achieve **state-of-the-art or near-SOTA performance on forecasting benchmarks** with significant efficiency gains. Extending this to other domains and scales is an open research direction.

## Theoretical Context

The mechanism demonstrates constraint-driven representation learning: forcing aggregation through a shared bottleneck requires the model to organize information so task-relevant features survive summation. This connects to information bottleneck theory and on self-organization under constraint.

The paper explores broader interpretive frameworks in Section 7.3 for readers interested in connections between constraint, representation, and meaning in learned systems.

## Citation

```bibtex
@article{Summation_Based_Transformers_2025,
  title={Summation-Based Transformers: A Path Toward Linear Complexity Sequence Modeling},
  author={Pascal Ekin},
  journal={TechRxiv},
  year={2025},
  doi={10.36227/techrxiv.175790522.25734653/v2}
}
```

## Repository Structure

```
summation-based-transformers/
├── patch_tst_all.py       # Time series forecasting benchmarks
├── causal.py              # Language modeling experiments
├── classification.py      # Document classification
└── multimodal.py          # Multimodal regression
```

## Contact

- Email: pfekin@gmail.com
- Issues: Use the GitHub issue tracker for bugs and technical questions
- Collaboration inquiries welcome

