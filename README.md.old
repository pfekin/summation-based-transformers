# High-Dimensional Aggregation for Efficient Sequence Modeling  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  

> **Sub-quadratic alternative to transformer attention** using high-dimensional embeddings and geometric properties like near-orthogonality. Achieves competitive performance with **7–20× faster computation**.

---

## 🚀 Key Features  
- **Linear complexity** (O(n·d)) vs. attention’s quadratic cost (O(n²·d)).  
- **Emergent orthogonality**: Token embeddings self-organize to minimize interference during aggregation.  
- **Philosophically grounded**: Inspired by Deleuze’s *multiplicity* and Derrida’s *différance*. 
- **State-of-the-art efficiency**: Matches attention on IMDB, WikiText, and AG News benchmarks.  

---

## 📊 Performance Highlights  

### Classification Accuracy vs. Attention Models  
 ![classification](https://github.com/user-attachments/assets/a2f88cfb-0163-4df3-b16a-99392c7894db)


### Auto-regression Perplexity vs. Attention Models  
![autoregression](https://github.com/user-attachments/assets/2249f3a3-2737-4cc9-8920-0226d28aca5b)

### Cosine Similarity Distribution (Aggregation vs. Attention)  
![histogram](https://github.com/user-attachments/assets/31b1c523-e1ba-44d0-9b5e-25228dc68626)


---

## ⚙️ How It Works  
1. **Token Aggregation**: Sum embeddings in high-dimensional space (e.g., 768D).  
2. **Geometric Preservation**: Near-orthogonality preserves semantic relationships.  
3. **Positional Fusion**: Combines token + positional info via lightweight functions (e.g., multiplication, FiLM).  
