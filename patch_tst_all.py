"""
Minimal PatchTST Implementation with Summation-Based Attention

Compares:
1. Baseline PatchTST with standard attention
2. Hybrid PatchTST with summation attention (trained from scratch)

Benchmark on ETTh1, ETTh2, ETTm1, ETTm2, Weather,  Traffic datasets
"""

import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Darts datasets for Traffic / Weather
from darts.datasets import TrafficDataset, WeatherDataset

# Data loading

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def timeseries_to_dataframe(ts):
    """Convert Darts TimeSeries to pandas DataFrame, handling 3D arrays"""
    # Newer Darts versions (>=0.30.0)
    if hasattr(ts, "pd_dataframe"):
        df = ts.pd_dataframe()
        # Handle potential 3D structure
        if isinstance(df, pd.DataFrame):
            return df
        else:
            arr = ts.all_values(copy=False)
            if arr.ndim == 3:
                arr = arr.squeeze(-1)  # Remove trailing dimension (T, n_vars, 1) -> (T, n_vars)
            cols = [f"var_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)
    # Older versions
    elif hasattr(ts, "all_values"):
        arr = ts.all_values(copy=False)
        if arr.ndim == 3:
            arr = arr.squeeze(-1)  # Convert (T, n_vars, 1) -> (T, n_vars)
        cols = [f"var_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
    else:
        raise AttributeError("Unsupported Darts TimeSeries object. Please update Darts.")


def load_dataset_general(name, seq_len, pred_len, batch_size=32,
                         val_ratio=0.2, test_ratio=0.2, max_vars=None):
    """
    Generalized loader:
      - ETTh1, ETTh2, ETTm1, ETTm2: raw CSV from Zhou repo
      - Traffic / Weather: from Darts loaders
    """
    if name == "ETTh1":
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        df = pd.read_csv(url)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
    elif name == "ETTh2":
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
        df = pd.read_csv(url)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
    elif name == "ETTm1":
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
        df = pd.read_csv(url)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
    elif name == "ETTm2":
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
        df = pd.read_csv(url)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
    elif name == "Traffic":
        ts = TrafficDataset().load()
        df = timeseries_to_dataframe(ts)
    elif name == "Weather":
        ts = WeatherDataset().load()
        df = timeseries_to_dataframe(ts)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if max_vars is not None and df.shape[1] > max_vars:
        df = df.iloc[:, :max_vars]

    values = df.values
    T, n_vars = values.shape

    # Train/val/test split
    n_test = int(T * test_ratio)
    n_val = int(T * val_ratio)
    n_train = T - n_val - n_test

    train_vals = values[:n_train]
    val_vals = values[n_train:n_train+n_val]
    test_vals = values[n_train+n_val:]

    # Standardize
    scaler = StandardScaler()
    scaler.fit(train_vals)
    train_scaled = scaler.transform(train_vals)
    val_scaled = scaler.transform(val_vals)
    test_scaled = scaler.transform(test_vals)

    # Handle NaN/Inf values after standardization
    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    val_scaled = np.nan_to_num(val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Wrap in PyTorch datasets
    train_ds = TimeSeriesDataset(train_scaled, seq_len, pred_len)
    val_ds = TimeSeriesDataset(val_scaled, seq_len, pred_len)
    test_ds = TimeSeriesDataset(test_scaled, seq_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_vars, scaler


# Patching layer

class Patching(nn.Module):
    """Convert time series into patches"""
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        batch_size, seq_len, n_vars = x.shape

        # Calculate number of patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        # Extract patches
        patches = torch.zeros(batch_size, n_vars, num_patches, self.patch_len, device=x.device)
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches[:, :, i, :] = x[:, start:end, :].transpose(1, 2)

        # Reshape: (batch, n_vars, num_patches, patch_len) -> (batch*n_vars, num_patches, patch_len)
        patches = patches.reshape(batch_size * n_vars, num_patches, self.patch_len)

        return patches, n_vars, num_patches


# Attention blocks

class StandardAttentionBlock(nn.Module):
    """Standard transformer block with multi-head self-attention"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SummationAttentionBlock(nn.Module):
    """
    Summation-based attention block
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            #nn.Linear(d_model, d_model, bias=False),
            #nn.ReLU(),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        summed = self.proj(x)
        x = self.norm1(x + summed)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


# PatchTST Models

class BaselinePatchTST(nn.Module):
    """PatchTST with standard attention"""
    def __init__(self, n_vars, seq_len, pred_len, patch_len=16, stride=8,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()

        self.patching = Patching(patch_len, stride)

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding
        num_patches = (seq_len - patch_len) // stride + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            StandardAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Prediction head
        self.head = nn.Linear(d_model * num_patches, pred_len)

        self.n_vars = n_vars
        self.pred_len = pred_len

    def forward(self, x):
        # x shape: (batch, seq_len, n_vars)
        batch_size = x.shape[0]

        # Patching: (batch*n_vars, num_patches, patch_len)
        patches, n_vars, num_patches = self.patching(x)

        # Embed patches
        x = self.patch_embedding(patches)  # (batch*n_vars, num_patches, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Flatten and predict
        x = x.reshape(batch_size * n_vars, -1)  # (batch*n_vars, num_patches * d_model)
        x = self.head(x)  # (batch*n_vars, pred_len)

        # Reshape to (batch, pred_len, n_vars)
        x = x.reshape(batch_size, n_vars, self.pred_len).transpose(1, 2)

        return x


class HybridPatchTST(nn.Module):
    """PatchTST with summation attention + final standard attention"""
    def __init__(self, n_vars, seq_len, pred_len, patch_len=16, stride=8,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()

        self.patching = Patching(patch_len, stride)

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding
        num_patches = (seq_len - patch_len) // stride + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Hybrid blocks: summation + final attention
        self.summation_blocks = nn.ModuleList([
            SummationAttentionBlock(d_model, d_ff, dropout)
            for _ in range(n_layers - 1)
        ])
        self.final_attention = StandardAttentionBlock(d_model, n_heads, d_ff, dropout)

        # Prediction head
        self.head = nn.Linear(d_model * num_patches, pred_len)

        self.n_vars = n_vars
        self.pred_len = pred_len

    def forward(self, x):
        # x shape: (batch, seq_len, n_vars)
        batch_size = x.shape[0]

        # Patching
        patches, n_vars, num_patches = self.patching(x)

        # Embed patches
        x = self.patch_embedding(patches)

        # Positional encoding using multiplication
        x = x * self.pos_encoding
        #x = x + self.pos_encoding

        # Summation blocks
        for block in self.summation_blocks:
            x = block(x)

        # Final attention
        x = self.final_attention(x)

        # Flatten and predict
        x = x.reshape(batch_size * n_vars, -1)
        x = self.head(x)

        # Reshape to (batch, pred_len, n_vars)
        x = x.reshape(batch_size, n_vars, self.pred_len).transpose(1, 2)

        return x


# Training & evaluation

def train_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs=30):
    """Train model and return best validation loss"""
    best_val_loss = float("inf")

    # Add gradient clipping to prevent explosions
    max_grad_norm = 1.0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for past_values, future_values in train_loader:
            past_values = past_values.to(device)
            future_values = future_values.to(device)

            optimizer.zero_grad()
            outputs = model(past_values)
            loss = criterion(outputs, future_values)

            # Check for NaN
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss detected at epoch {epoch+1}, skipping batch")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for past_values, future_values in val_loader:
                past_values = past_values.to(device)
                future_values = future_values.to(device)

                outputs = model(past_values)
                loss = criterion(outputs, future_values)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)

        # Early stopping if training diverges
        if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss)):
            print(f"  Training diverged at epoch {epoch+1}. Stopping early.")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Best={best_val_loss:.4f}")

    return best_val_loss


def evaluate_model(model, loader, device):
    """Compute MSE and MAE on a given DataLoader."""
    model.eval()
    mse_loss = 0.0
    mae_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            mse_loss += torch.nn.functional.mse_loss(y_pred, y, reduction="sum").item()
            mae_loss += torch.nn.functional.l1_loss(y_pred, y, reduction="sum").item()
            count += y.numel()
    return mse_loss / count, mae_loss / count


def measure_inference_speed(model, loader, device, warmup=2, reps=5):
    """Measure inference speed in samples/sec."""
    model.eval()
    # Warmup passes
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= warmup:
                break
            _ = model(x.to(device))

    # Timed passes
    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
        for r in range(reps):
            for x, _ in loader:
                x = x.to(device)
                batch_size = x.size(0)
                start = time.time()
                _ = model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_time += time.time() - start
                total_samples += batch_size
    return total_samples / total_time


# Main experiment

def run_experiment(dataset_name, train_loader, val_loader, test_loader, n_vars, config, device):
    print("\n" + "="*70)
    print(f"Dataset: {dataset_name}")
    print("="*70)

    # Initialize models
    baseline_model = BaselinePatchTST(
        n_vars=n_vars, seq_len=config['seq_len'], pred_len=config['pred_len'],
        patch_len=config['patch_len'], stride=config['stride'],
        d_model=config['d_model'], n_heads=config['n_heads'],
        n_layers=config['n_layers'], d_ff=config['d_ff'], dropout=config['dropout']
    ).to(device)

    hybrid_model = HybridPatchTST(
        n_vars=n_vars, seq_len=config['seq_len'], pred_len=config['pred_len'],
        patch_len=config['patch_len'], stride=config['stride'],
        d_model=config['d_model'], n_heads=config['n_heads'],
        n_layers=config['n_layers'], d_ff=config['d_ff'], dropout=config['dropout']
    ).to(device)

    print(f"Baseline params: {sum(p.numel() for p in baseline_model.parameters()):,}")
    print(f"Hybrid params: {sum(p.numel() for p in hybrid_model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer_baseline = torch.optim.Adam(baseline_model.parameters(), lr=config['lr'])
    optimizer_hybrid = torch.optim.Adam(hybrid_model.parameters(), lr=config['lr'])

    # Train baseline
    print("\nTraining baseline PatchTST...")
    start_time = time.time()
    baseline_val = train_model(baseline_model, train_loader, val_loader,
                               criterion, optimizer_baseline, device, config['n_epochs'])
    baseline_time = time.time() - start_time

    # Train hybrid
    print("\nTraining hybrid PatchTST (summation)...")
    start_time = time.time()
    hybrid_val = train_model(hybrid_model, train_loader, val_loader,
                             criterion, optimizer_hybrid, device, config['n_epochs'])
    hybrid_time = time.time() - start_time

    print("\nValidation Results:")
    print(f"Baseline best val MSE: {baseline_val:.4f} (time {baseline_time:.1f}s)")
    print(f"Hybrid   best val MSE: {hybrid_val:.4f} (time {hybrid_time:.1f}s)")

    # Final test evaluation
    baseline_test_mse, baseline_test_mae = evaluate_model(baseline_model, test_loader, device)
    hybrid_test_mse, hybrid_test_mae   = evaluate_model(hybrid_model, test_loader, device)

    baseline_speed = measure_inference_speed(baseline_model, test_loader, device)
    hybrid_speed   = measure_inference_speed(hybrid_model, test_loader, device)

    print("\nTest Results:")
    print(f"Baseline -> MSE: {baseline_test_mse:.4f}, MAE: {baseline_test_mae:.4f}, Inference: {baseline_speed:.1f} samples/s")
    print(f"Hybrid   -> MSE: {hybrid_test_mse:.4f}, MAE: {hybrid_test_mae:.4f}, Inference: {hybrid_speed:.1f} samples/s")

    return {
        'dataset': dataset_name,
        'baseline_mse': baseline_test_mse,
        'baseline_mae': baseline_test_mae,
        'hybrid_mse': hybrid_test_mse,
        'hybrid_mae': hybrid_test_mae,
        'baseline_speed': baseline_speed,
        'hybrid_speed': hybrid_speed
    }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    CONFIG = {
        'seq_len': 512,
        'pred_len': 96,
        'patch_len': 16,
        'stride': 8,
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 3,
        'd_ff': 256,
        'batch_size': 32,
        'n_epochs': 10,
        'lr': 1e-4,  
        'dropout': 0.10
    }

    datasets_to_run = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Traffic"]

    results = []
    for dset in datasets_to_run:
        max_vars = 100 if dset == "Traffic" else None # WARNING, it uses > 8Gb of VRAM
        train_loader, val_loader, test_loader, n_vars, _ = load_dataset_general(
            dset, CONFIG['seq_len'], CONFIG['pred_len'],
            batch_size=CONFIG['batch_size'], max_vars=max_vars
        )
        result = run_experiment(dset, train_loader, val_loader, test_loader, n_vars, CONFIG, device)
        results.append(result)

    # Print summary table
    print("\n" + "="*70)
    print("Final benchmark summary")
    print("="*70)
    for r in results:
        print(f"\n{r['dataset']}:")
        print(f"  Baseline: MSE={r['baseline_mse']:.4f}, MAE={r['baseline_mae']:.4f}, Speed={r['baseline_speed']:.1f} samples/s")
        print(f"  Hybrid:   MSE={r['hybrid_mse']:.4f}, MAE={r['hybrid_mae']:.4f}, Speed={r['hybrid_speed']:.1f} samples/s")
        improvement = ((r['baseline_mse'] - r['hybrid_mse']) / r['baseline_mse']) * 100
        speedup = r['hybrid_speed'] / r['baseline_speed']
        print(f"  -> MSE improvement: {improvement:+.1f}%, Speed ratio: {speedup:.2f}x")
