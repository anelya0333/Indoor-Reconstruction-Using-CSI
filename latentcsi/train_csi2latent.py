#!/usr/bin/env python3
"""
train_csi2latent.py
===============================================================================
Purpose
===============================================================================
Train a simple MLP (Multi-Layer Perceptron) model to predict Stable Diffusion
VAE latent representations directly from normalized CSI (Channel State
Information) feature vectors.

This effectively learns a mapping:
    CSI features → (VAE latent space)

Each CSI vector corresponds to a specific video frame whose image has been
encoded into a latent tensor using the Stable Diffusion autoencoder (4×64×64).

===============================================================================
Pipeline Context
===============================================================================
Stage: 08 — CSI→Latent Regression Model Training

Previous stages:
  00–07  → data preprocessing, alignment, and VAE latent encoding.
  07_encode_latents.py → produced latents_{split}.npy (4×64×64 tensors)
  03_normalize_and_split.py → produced normalized CSI features + splits

This script:
  - Loads CSI features and corresponding latents for each split.
  - Trains an MLP model to minimize MSE between predicted and ground-truth latents.
  - Saves checkpoints (best and last) during training.

===============================================================================
Usage
===============================================================================
    python train_csi2latent.py \
        --splits_dir dataset_out/splits \
        --latents_dir dataset_out/latents \
        --out runs/latentcsi \
        --batch 32 \
        --epochs 40 \
        --lr 2e-4 \
        --hidden 2048

===============================================================================
Inputs
===============================================================================
- --splits_dir :
    Directory containing dataset splits (features_norm.npy, idx_train.npy, idx_val.npy).

- --latents_dir :
    Directory containing precomputed VAE latent .npy files (latents_train.npy, latents_val.npy).

- --out :
    Output directory for checkpoints and logs.

- --batch :
    Batch size for training (default = 32).

- --epochs :
    Number of training epochs (default = 40).

- --lr :
    Learning rate for AdamW optimizer (default = 2e-4).

- --hidden :
    Size of hidden layers in the MLP (default = 2048).

===============================================================================
Outputs
===============================================================================
Saved to `--out` directory:
| File | Description |
|------|--------------|
| last.pt | Latest checkpoint after each epoch |
| best.pt | Checkpoint with lowest validation MSE |

Console output:
    JSON logs with per-epoch training and validation MSE

===============================================================================
Requirements
===============================================================================
- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- NumPy  
- tqdm  
===============================================================================
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
# Dataset Definition
# =============================================================================

class PairSet(Dataset):
    """
    Dataset class linking CSI features to their corresponding image latents.

    Each sample is a tuple:
        (CSI_features, latent_tensor)

    Attributes:
        X (np.ndarray): CSI feature matrix (N×D)
        Z (np.ndarray): Latent tensor array (N×4×64×64)
        idx (np.ndarray): Row indices corresponding to split subset
    """
    def __init__(self, feats_npy: str, latents_npy: str, idx_npy: str):
        """
        Args:
            feats_npy (str): Path to normalized CSI feature matrix (.npy).
            latents_npy (str): Path to latent tensor file (.npy) for this split.
            idx_npy (str): Path to index array (.npy) indicating which rows of
                           features_norm.npy correspond to this split.
        """
        self.X = np.load(feats_npy).astype("float32")   # (N_total, D)
        self.Z = np.load(latents_npy).astype("float32") # (N_split, 4, 64, 64)
        idx = np.load(idx_npy)                          # e.g., idx_train.npy

        # Subselect feature rows corresponding to this split
        self.X = self.X[idx]
        self.Z = self.Z  # Latents already in correct order (1:1 with split)
        assert len(self.X) == len(self.Z), "Mismatch between CSI features and latent samples."

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, i: int):
        """
        Retrieve one (CSI, latent) pair.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - CSI features tensor: (D,)
                - Latent tensor: (4, 64, 64)
        """
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Z[i])


# =============================================================================
# Model Definition
# =============================================================================

class CSI2LatentMLP(nn.Module):
    """
    Multi-Layer Perceptron mapping CSI feature vectors to latent tensors.

    Architecture:
        Input:  D-dimensional CSI feature vector
        Hidden: 2 fully-connected layers with SiLU activation
        Output: Flattened latent tensor (4×64×64 = 16384 dims)
    """
    def __init__(self, d_in: int, z_shape=(4, 64, 64), hidden: int = 2048):
        """
        Args:
            d_in (int): Input feature dimension (number of CSI features).
            z_shape (tuple): Shape of the output latent tensor.
            hidden (int): Hidden layer width.
        """
        super().__init__()
        z_flat = int(np.prod(z_shape))  # 4×64×64 = 16384
        self.z_shape = z_shape
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, z_flat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input CSI feature batch (B×D)

        Returns:
            torch.Tensor: Predicted latent tensors (B×4×64×64)
        """
        y = self.net(x)
        b = x.shape[0]
        return y.view(b, *self.z_shape)


# =============================================================================
# Training Routine
# =============================================================================

def main():
    """
    Train an MLP to regress VAE latents from CSI feature vectors.

    Workflow:
      1. Load training and validation datasets.
      2. Initialize model, optimizer, and MSE loss.
      3. Iterate through epochs:
         - Train on train split.
         - Evaluate on validation split.
         - Log losses and save checkpoints.
    """
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Train MLP to map CSI → VAE latent space.")
    ap.add_argument("--splits_dir", default="dataset_out/splits",
                    help="Directory containing feature and split index files.")
    ap.add_argument("--latents_dir", default="dataset_out/latents",
                    help="Directory containing precomputed VAE latents.")
    ap.add_argument("--out", default="runs/latentcsi",
                    help="Output directory for model checkpoints and logs.")
    ap.add_argument("--batch", type=int, default=32, help="Batch size (default: 32).")
    ap.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate for AdamW.")
    ap.add_argument("--hidden", type=int, default=2048, help="Hidden layer width.")
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)

    feats = os.path.join(args.splits_dir, "features_norm.npy")
    idx_tr = os.path.join(args.splits_dir, "idx_train.npy")
    idx_va = os.path.join(args.splits_dir, "idx_val.npy")

    Ztr = os.path.join(args.latents_dir, "latents_train.npy")
    Zva = os.path.join(args.latents_dir, "latents_val.npy")

    D = np.load(feats).shape[1]  # Input feature dimension

    # Datasets and DataLoaders
    ds_tr = PairSet(feats, Ztr, idx_tr)
    ds_va = PairSet(feats, Zva, idx_va)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2)

    # -------------------------------------------------------------------------
    # Model, optimizer, and loss
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSI2LatentMLP(D, (4, 64, 64), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---------------------- TRAIN ----------------------
        model.train()
        total_train_loss = 0.0
        num_train_samples = 0

        for xb, zb in tqdm(dl_tr, desc=f"Epoch {epoch} [train]", ncols=80):
            xb, zb = xb.to(device), zb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, zb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss.item() * xb.size(0)
            num_train_samples += xb.size(0)

        train_mse = total_train_loss / num_train_samples

        # ---------------------- VALIDATION ----------------------
        model.eval()
        total_val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad():
            for xb, zb in dl_va:
                xb, zb = xb.to(device), zb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, zb)
                total_val_loss += loss.item() * xb.size(0)
                num_val_samples += xb.size(0)

        val_mse = total_val_loss / num_val_samples

        # ---------------------- LOGGING ----------------------
        log = {"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse}
        print(json.dumps(log))

        # ---------------------- CHECKPOINTS ----------------------
        ckpt = {
            "model": model.state_dict(),
            "d_in": D,
            "hidden": args.hidden,
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if val_mse < best_val:
            best_val = val_mse
            torch.save(ckpt, os.path.join(args.out, "best.pt"))

    print(f"Training complete. Best validation MSE: {best_val:.6f}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
