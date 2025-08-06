import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from load_datasets import load_wikitext2, load_imdb, load_ag_news, load_cmu_book_summaries

# Custom collate function to handle variable-length sequences
def collate_fn(batch, pad_token_id=50256):
    # Get the maximum length in the batch
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Pad all sequences to max_length
    input_ids = []
    attention_masks = []
    
    for item in batch:
        # Pad input_ids
        padded_input = torch.cat([
            item["input_ids"], 
            torch.full((max_length - len(item["input_ids"]),), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_input)
        
        # Pad attention_mask
        padded_mask = torch.cat([
            item["attention_mask"], 
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
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def attention(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        if self.use_superposition:
            # Pure Addition/Superposition - project then sum token embeddings
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

# Simple transformer model
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
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        mask = mask.to(input_ids.device)
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        if self.use_superposition:
            x = self.token_embedding(input_ids) * (self.pos_embedding(positions) + 2.0) # positional encoding bias 
        else:
            x = self.token_embedding(input_ids) + self.pos_embedding(positions)
                
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits, x  # return both logits and final embeddings (for analysis)

# Training function
def train_epoch(model, dataloader, optimizer, device, pad_token_id):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        
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

# Validation function
def validate(model, dataloader, device, pad_token_id):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            # Create mask to ignore padding tokens
            mask = (targets != pad_token_id)
            
            logits, embeddings = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)
            
            total_loss += loss.item()
            
            # Calculate accuracy excluding padding tokens
            predicted = logits.argmax(dim=-1)
            correct += ((predicted == targets) & mask).sum().item()
            total += mask.sum().item()
            
            # Collect embeddings for orthogonality analysis - flatten to handle variable lengths
            # Only collect non-padding embeddings
            input_mask = (inputs != pad_token_id)
            valid_embeddings = embeddings[input_mask]  # Shape: [num_valid_tokens, embed_dim]
            all_embeddings.append(valid_embeddings.cpu())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    # Concatenate all valid embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    return avg_loss, accuracy, perplexity.item(), all_embeddings

# Main experiment
def main():
    MAX_SEQ_LENGTH = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    
    #
    # Uncomment the dataset for benchmarking vs baseline attention
    #
    
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
    results = {}
    
    for use_superposition in [True, False]:
        model_name = "Representational Superosition" if use_superposition else "Standard"
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
            val_loss, val_acc, val_perplexity, val_embeddings = validate(model, val_loader, device, pad_token_id)
            
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