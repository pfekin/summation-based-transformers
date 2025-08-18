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
import matplotlib.cm as cm

"""
Implementation Note:
The language modeling implementation uses cumsum() for clarity. 
This PyTorch function is broken and its performance does not reflect the theoretical O(n) vs O(n²) 
complexity difference.
"""

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
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
                nn.ReLU()
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
        # Pure superposition approach, uncomment to benchmark
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

def plot_pca_with_arrows(acts, model_name, ax):
    """
    acts: list of layer tensors [layer0, layer1, ...], shape [n_tokens, dim]
    """
    # Concatenate all layers for PCA
    X = torch.cat([x.reshape(-1, x.shape[-1]) for x in acts], dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X)

    # Compute start/end indices per layer
    idx_start = 0
    colors = cm.viridis(np.linspace(0,1,len(acts)))
    layer_centers = []

    for i, layer in enumerate(acts):
        n_tokens = layer.shape[0] if len(layer.shape)==2 else np.prod(layer.shape[:2])
        layer_proj = proj[idx_start:idx_start+n_tokens]
        idx_start += n_tokens

        # scatter points for tokens (small size)
        ax.scatter(layer_proj[:,0], layer_proj[:,1], s=2, color=colors[i], alpha=0.4)

        # compute mean per layer
        center = layer_proj.mean(axis=0)
        layer_centers.append(center)

    # draw arrows connecting layer centers
    layer_centers = np.array(layer_centers)
    for i in range(len(layer_centers)-1):
        ax.annotate("", xy=layer_centers[i+1], xytext=layer_centers[i],
                    arrowprops=dict(arrowstyle="->", color="black", alpha=0.8))

    ax.set_title(f"PCA - {model_name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # show first and last layer in legend
    ax.scatter([], [], color=colors[0], label="Layer 0")
    ax.scatter([], [], color=colors[-1], label=f"Layer {len(acts)-1}")
    ax.legend(markerscale=2)

def compute_layerwise_cosine(layer_outputs):
    sims = []
    for i in range(len(layer_outputs)-1):
        a = F.normalize(layer_outputs[i], dim=1)
        b = F.normalize(layer_outputs[i+1], dim=1)
        sims.append((a*b).sum(dim=1).mean().item())
    return sims


def compute_participation_ratio(acts):
    pr_values = []
    for x in acts:
        X = x.reshape(-1, x.shape[-1]).cpu().numpy()
        cov = np.cov(X, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        pr = (eigvals.sum()**2) / (np.square(eigvals).sum() + 1e-8)
        pr_values.append(pr)
    return pr_values


def eval_models(acts_vanilla, acts_mod, model_names=("Vanilla","Modified")):
    fig, axes = plt.subplots(2, 2, figsize=(12,10))

    # PCA top row
    plot_pca_with_arrows(acts_vanilla, model_names[0], axes[0,0])
    plot_pca_with_arrows(acts_mod, model_names[1], axes[0,1])

    # Layer-wise cosine similarity bottom-left
    sim_v = compute_layerwise_cosine(acts_vanilla)
    sim_m = compute_layerwise_cosine(acts_mod)
    layers_cos = np.arange(len(sim_v))
    ymin = min(sim_v + sim_m)
    ymax = max(sim_v + sim_m)
    pad = 0.1*(ymax-ymin)

    axes[1,0].plot(layers_cos, sim_v, marker="o", label=model_names[0])
    axes[1,0].plot(layers_cos, sim_m, marker="o", label=model_names[1])
    axes[1,0].set_ylim(ymin-pad, ymax+pad)
    axes[1,0].set_xlabel("Layer index")
    axes[1,0].set_ylabel("Mean cosine similarity")
    axes[1,0].set_title("Layer-wise cosine similarity")
    axes[1,0].grid(True)
    axes[1,0].legend()

    # Participation ratio bottom-right
    pr_v = compute_participation_ratio(acts_vanilla)
    pr_m = compute_participation_ratio(acts_mod)
    layers_pr = np.arange(len(pr_v))
    ymin = min(pr_v + pr_m)
    ymax = max(pr_v + pr_m)
    pad = 0.1*(ymax-ymin)

    axes[1,1].plot(layers_pr, pr_v, marker="o", label=model_names[0])
    axes[1,1].plot(layers_pr, pr_m, marker="o", label=model_names[1])
    axes[1,1].set_ylim(ymin-pad, ymax+pad)
    axes[1,1].set_xlabel("Layer index")
    axes[1,1].set_ylabel("Participation ratio")
    axes[1,1].set_title("Participation ratio")
    axes[1,1].grid(True)
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()


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
        model_name = "Superposition" if use_superposition else "Standard"
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
        
         
    eval_models(all_layerwise_data["Standard"], all_layerwise_data["Superposition"], model_names=("Attention", "Summation"))
    
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