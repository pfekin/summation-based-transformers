# High-Dimensional Aggregation for Efficient Sequence Modeling  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  

> **Sub-quadratic alternative to transformer attention** using high-dimensional embeddings and geometric properties like near-orthogonality. Achieves competitive performance with **7–20× faster computation**.

---

## 🚀 Key Features  
- **Linear complexity** (O(n·d)) vs. attention’s quadratic cost (O(n²·d)).  
- **Emergent orthogonality**: Token embeddings self-organize to minimize interference during aggregation.  
- **Philosophically grounded**: Inspired by Deleuze’s *multiplicity*, Derrida’s *différance*, and Atlan’s *crystal-smoke* cognitive balance.  
- **State-of-the-art efficiency**: Matches attention on IMDB, WikiText, and AG News benchmarks.  

---

## 📊 Performance Highlights  

### Classification Accuracy vs. Attention Models  
![Classification Accuracy](classification.png)  

### Auto-regression Perplexity vs. Attention Models  
![Classification Accuracy]([image-url]/autoregression.png)  

### Cosine Similarity Distribution (Aggregation vs. Attention)  
![Cosine Similarity Histogram]([image-url]/histogram.png)  

---

## ⚙️ How It Works  
1. **Token Aggregation**: Sum embeddings in high-dimensional space (e.g., 768D).  
2. **Geometric Preservation**: Near-orthogonality preserves semantic relationships.  
3. **Positional Fusion**: Combines token + positional info via lightweight functions (e.g., multiplication, FiLM).  
