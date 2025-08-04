import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from load_datasets import load_wikitext2, load_imdb, load_ag_news, load_cmu_book_summaries
from sklearn.decomposition import PCA

# Custom collate function to handle variable-length sequences
def collate_fn(batch, pad_token_id=50256):
    # Get the maximum length in the batch
    max_length = max(len(item['input_ids']) for item in batch)
    
    # Pad all sequences to max_length
    input_ids = []
    attention_masks = []
    
    for item in batch:
        # Pad input_ids
        padded_input = torch.cat([
            item['input_ids'], 
            torch.full((max_length - len(item["input_ids"]),), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_input)
        
        # Pad attention_mask
        padded_mask = torch.cat([
            item['attention_mask'], 
            torch.zeros(max_length - len(item["attention_mask"]), dtype=torch.long)
        ])
        attention_masks.append(padded_mask)
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks)
    }

# Simple transformer block with configurable attention fusion
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, use_superposition=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_superposition = use_superposition
        
        # For Representational Superposition, we don't need multiple heads, just use embed_dim
        if use_superposition:
            self.head_dim = embed_dim
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                nn.ReLU(),# GELU
                nn.Linear(embed_dim, embed_dim, bias=False),
                nn.ReLU() # GELU
            ) 
        else:
            self.head_dim = embed_dim // num_heads
            self.q_proj = nn.Linear(embed_dim, self.head_dim * num_heads)
            self.k_proj = nn.Linear(embed_dim, self.head_dim * num_heads)
            self.v_proj = nn.Linear(embed_dim, self.head_dim * num_heads)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Simple feedforward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
    def attention(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        if self.use_superposition:
            # Superposition - project then sum token embeddings
            projected_embeddings = self.proj(q)
            
            summed_output = torch.cumsum(projected_embeddings, dim=1)
            
            return self.out_proj(summed_output)
        else:
            # Standard multi-head attention
            # Project to Q, K, V
            Q = self.q_proj(q)
            K = self.k_proj(k)
            V = self.v_proj(v)
            
            # Reshape for multi-head attention
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention computation
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            
            # Apply causal mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention_weights, V)
            
            # Reshape back
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            
            return self.out_proj(attention_output)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=4, max_seq_len=512, use_superposition=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_superposition = use_superposition

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        #
        # Puere superposition approach, uncomment to benchmark
        #
        """        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, use_superposition) 
            for _ in range(num_layers)
        ])
        """
        #
        # Hybrid approach (last layer is an attention layer) 
        #
        layers = []
        for i in range(num_layers):
            attention = False if i == (num_layers - 1) else use_superposition
            layers.append(TransformerBlock(embed_dim, num_heads, attention))
        self.layers = nn.ModuleList(layers)

        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, return_all_layers=False):
        batch_size, seq_len = input_ids.size()
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(input_ids.device)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        if self.use_superposition:
            x = self.token_embedding(input_ids) * (self.pos_embedding(positions) + 2.0) # positional encoding bias
        else:
            x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        all_layers = [x.detach()] if return_all_layers else None
        for layer in self.layers:
            x = layer(x, mask)
            if return_all_layers:
                all_layers.append(x.detach())

        x = self.final_norm(x)
        logits = self.output_projection(x)

        return (logits, x, all_layers) if return_all_layers else (logits, x)

# Training function
def train_epoch(model, dataloader, optimizer, device, pad_token_id):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        
        # Create targets (next token prediction)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        
        # Create mask to ignore padding tokens
        mask = (targets != pad_token_id)
        
        optimizer.zero_grad()
        
        logits, embeddings = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy excluding padding tokens
        predicted = logits.argmax(dim=-1)
        correct += ((predicted == targets) & mask).sum().item()
        total += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, accuracy, perplexity.item()

def validate(model, dataloader, device, pad_token_id):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_embeddings = []
    all_layerwise = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            mask = (targets != pad_token_id)

            logits, embeddings, layerwise = model(inputs, return_all_layers=True)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)

            total_loss += loss.item()
            predicted = logits.argmax(dim=-1)
            correct += ((predicted == targets) & mask).sum().item()
            total += mask.sum().item()

            valid_embeddings = embeddings[mask].cpu()
            all_embeddings.append(valid_embeddings)

            # Store per-layer outputs, masked
            for i, layer_out in enumerate(layerwise):
                if len(all_layerwise) <= i:
                    all_layerwise.append([])
                all_layerwise[i].append(layer_out[mask].cpu())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss))
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_layerwise = [torch.cat(layer_outputs, dim=0) for layer_outputs in all_layerwise]

    return avg_loss, accuracy, perplexity.item(), all_embeddings, all_layerwise

# Orthogonality analysis
def analyze_orthogonality(embeddings, title="Embedding Orthogonality"):
    # Flatten embeddings and sample subset for efficiency
    flat_embeddings = embeddings.view(-1, embeddings.size(-1))
    
    # Sample random pairs to compute cosine similarities
    n_samples = min(10000, flat_embeddings.size(0))
    indices = torch.randperm(flat_embeddings.size(0))[:n_samples]
    sample_embeddings = flat_embeddings[indices]
    
    # Compute pairwise cosine similarities
    normalized = F.normalize(sample_embeddings, dim=1)
    similarities = torch.mm(normalized, normalized.t())
    
    # Extract upper triangular part (excluding diagonal)
    mask = torch.triu(torch.ones_like(similarities, dtype=bool), diagonal=1)
    similarities = similarities[mask].numpy()
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, alpha=0.7, density=True)
    plt.title(f"{title} - Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Perfect Orthogonality")
    plt.legend()
    plt.show()
    
    return similarities

# Visualization for layer redundancy
def plot_layerwise_cosine(layer_outputs, title="Layer-wise Cosine Similarity"):
    similarities = []
    for i in range(len(layer_outputs) - 1):
        a = F.normalize(layer_outputs[i], dim=1)
        b = F.normalize(layer_outputs[i+1], dim=1)
        sim = (a * b).sum(dim=1).mean().item()
        similarities.append(sim)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(similarities)), similarities, marker="o")
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Cosine Similarity")
    plt.grid(True)
    plt.show()


# Visualization for embedding-to-mean similarity
def plot_embedding_alignment(layer_outputs, title="Embedding-to-Mean Alignment"):
    similarities = []
    for i, layer in enumerate(layer_outputs):
        mean_vec = layer.mean(dim=0, keepdim=True)
        layer_norm = F.normalize(layer, dim=1)
        mean_norm = F.normalize(mean_vec, dim=1)
        sim = (layer_norm * mean_norm).sum(dim=1)
        similarities.append(sim.numpy())

    plt.figure(figsize=(10, 5))
    for i, sim in enumerate(similarities):
        plt.hist(sim, bins=50, alpha=0.4, label=f"Layer {i}", density=True)
    plt.title(title)
    plt.xlabel("Cosine Similarity to Mean Vector")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_pca_comparison(layer_outputs_superposition, layer_outputs_attention, samples_per_layer=500):
    """
    Fits a single PCA on the combined embeddings from both models,
    then plots them on a shared scale for direct comparison.
    """
    # Helper function to sample embeddings from a model's layer outputs
    def sample_embeddings(layer_outputs):
        all_samples, all_labels = [], []
        for i, layer_tensor in enumerate(layer_outputs):
            if layer_tensor.size(0) == 0:
                continue
            sample_indices = torch.randperm(layer_tensor.size(0))[:samples_per_layer]
            all_samples.append(layer_tensor[sample_indices].cpu().numpy())
            all_labels.append(np.full(len(sample_indices), i))
        return np.vstack(all_samples), np.concatenate(all_labels)

    superposition_samples, superposition_labels = sample_embeddings(layer_outputs_superposition)
    attention_samples, attention_labels = sample_embeddings(layer_outputs_attention)

    combined_samples = np.vstack([superposition_samples, attention_samples])
    print(f"Fitting a single PCA on a combined dataset of {combined_samples.shape[0]} token embeddings...")
    pca = PCA(n_components=2)
    pca.fit(combined_samples)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=True, sharey=True)
    fig.suptitle('PCA of Layer-wise Embeddings (Shared Scale)', fontsize=20)

    # Plot superposition model
    ax1 = axes[0]
    principal_components_sup = pca.transform(superposition_samples)
    scatter1 = ax1.scatter(
        principal_components_sup[:, 0],
        principal_components_sup[:, 1],
        c=superposition_labels,
        cmap="viridis",
        alpha=0.6
    )
    ax1.set_title("Superposition Model", fontsize=16)
    ax1.set_xlabel("Principal Component 1", fontsize=12)
    ax1.set_ylabel("Principal Component 2", fontsize=12)
    ax1.grid(True)

    # Plot attention model
    ax2 = axes[1]
    principal_components_att = pca.transform(attention_samples)
    ax2.scatter(
        principal_components_att[:, 0],
        principal_components_att[:, 1],
        c=attention_labels,
        cmap="viridis",
        alpha=0.6
    )
    ax2.set_title("Standard Attention Model", fontsize=16)
    ax2.set_xlabel("Principal Component 1", fontsize=12)
    ax2.grid(True)
    
    unique_labels = sorted(list(set(superposition_labels)))
    legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                 label=f"Layer {i}",
                                 markerfacecolor=scatter1.cmap(scatter1.norm(i)),
                                 markersize=10) for i in unique_labels]
    fig.legend(handles=legend_handles, title="Layers", bbox_to_anchor=(0.98, 0.9), loc="upper left")

    plt.show()

    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by shared PCA: PC1: {explained_variance[0]:.2%}, PC2: {explained_variance[1]:.2%}")

# Main experiment
def main():
    MAX_SEQ_LENGTH = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    
    #print("Loading WikiText-2...")
    #train_dataset, val_dataset, _ = load_wikitext2(tokenizer, max_length=MAX_SEQ_LENGTH)
    #print(f"WikiText-2 - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    #print("\nLoading IMDB...")
    #train_dataset, val_dataset = load_imdb(tokenizer, max_length=MAX_SEQ_LENGTH)
    #print(f"IMDB - Train: {len(train_dataset)}, Test: {len(val_dataset)}")
    
    print("\nLoading AG News...")
    train_dataset, val_dataset = load_ag_news(tokenizer, max_length=MAX_SEQ_LENGTH)
    print(f"AG News - Train: {len(train_dataset)}, Test: {len(val_dataset)}")
    
    #print("\nLoading CMU Book Summaries (with splits)...")
    #train_dataset, val_dataset, _ = load_cmu_book_summaries(tokenizer, max_length=MAX_SEQ_LENGTH, split_data=True)
    #print(f"CMU Book Summaries - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Use smaller subset for faster experimentation (optional)
    train_subset_size = min(5000, len(train_dataset))  # Use 5000 samples or full dataset if smaller
    val_subset_size = min(1000, len(val_dataset))
    
    train_subset = torch.utils.data.Subset(train_dataset, range(train_subset_size))
    val_subset = torch.utils.data.Subset(val_dataset, range(val_subset_size))
    
    # Use custom collate function with proper pad token
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, 
                             collate_fn=lambda batch: collate_fn(batch, pad_token_id))
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, 
                           collate_fn=lambda batch: collate_fn(batch, pad_token_id))
    
    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
    
    # Train both models
    
    # Store results and final layer embeddings for both models
    results = {}
    all_layerwise_data = {} 
    
    
    for use_superposition in [True, False]:
        model_name = "Representational Superposition" if use_superposition else "Standard"
        print(f"\nTraining {model_name} Model")
        
        model = SimpleTransformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=512,
            num_heads=8,
            num_layers=4,
            max_seq_len = MAX_SEQ_LENGTH,
            use_superposition=use_superposition
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        model_results = []
        
        # Train for a few epochs
        for epoch in range(10):
            train_loss, train_acc, train_perplexity = train_epoch(model, train_loader, optimizer, device, pad_token_id)
            val_loss, val_acc, val_perplexity, val_embeddings, all_layerwise = validate(model, val_loader, device, pad_token_id)
            print(f"Epoch {epoch+1}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Perplexity: {train_perplexity:.2f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Perplexity: {val_perplexity:.2f}")
            
            model_results.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_perplexity": train_perplexity,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_perplexity": val_perplexity
            })
        
        results[model_name] = model_results
        
        
        
        # Store the final layerwise embeddings for the comparison plot later
        all_layerwise_data[model_name] = all_layerwise 
        
        # We will plot these after both models have run.
        # You can keep these individual plots if you still want them.
        print(f"\nAnalyzing embeddings for {model_name} model from the final epoch...")
        plot_layerwise_cosine(all_layerwise, title=f"{model_name} - Layer Redundancy")
        plot_embedding_alignment(all_layerwise, title=f"{model_name} - Embedding Alignment to Layer Mean")
        
        
        # Example usage after validation
        plot_layerwise_cosine(all_layerwise, title=f"{model_name} - Layer Redundancy")
        plot_embedding_alignment(all_layerwise, title=f"{model_name} - Embedding Alignment to Layer Mean")
        
        """
        # Analyze orthogonality
        print(f"\nAnalyzing orthogonality for {model_name} model...")
        similarities = analyze_orthogonality(val_embeddings, f"{model_name} Attention")
        
        print(f"{model_name} - Mean cosine similarity: {similarities.mean():.4f}")
        print(f"{model_name} - Std cosine similarity: {similarities.std():.4f}")
        """
    
    # After the main loop over both models is complete, create the comparison plot
    if "Superposition" in all_layerwise_data and "Standard" in all_layerwise_data:
        print("\nGenerating Combined PCA Comparison Plot")
        plot_pca_comparison(
            all_layerwise_data["Superposition"],
            all_layerwise_data["Standard"]
        ) 
    
    # Compare final results
    print("\nFinal Comparison")
    for model_name, model_results in results.items():
        final_result = model_results[-1]
        print(f"{model_name} Model (Final Epoch):")
        print(f"  Validation Perplexity: {final_result['val_perplexity']:.2f}")
        print(f"  Validation Accuracy: {final_result['val_acc']:.4f}")
        print(f"  Validation Loss: {final_result['val_loss']:.4f}")
        print()

if __name__ == "__main__":
    main()