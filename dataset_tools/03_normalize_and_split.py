#!/usr/bin/env python3
"""
03_normalize_and_split.py
===============================================================================
Purpose
===============================================================================
Normalize CSI (Channel State Information) feature data and create reproducible
train/validation/test splits for downstream machine learning tasks.

The script expects pre-aligned CSI-frame data produced by
`02_align_and_build_pairs.py` and performs the following operations:

1. Load CSI feature matrix (`features.npy`) and corresponding manifest (`manifest.jsonl`).
2. Shuffle and split the dataset into train, validation, and test subsets
   according to user-specified proportions.
3. Compute normalization statistics (mean and standard deviation) **using only
   the training set** to prevent data leakage.
4. Apply z-score normalization to all features.
5. Save:
     - Normalized feature matrix (`features_norm.npy`)
     - Normalization parameters (`norm_mu.npy`, `norm_sigma.npy`)
     - Split indices (`idx_train.npy`, etc.)
     - Image path lists (`train.txt`, `val.txt`, `test.txt`)

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------
    python 03_normalize_and_split.py \
        --aligned_dir dataset_out/aligned \
        --out_dir dataset_out/splits \
        --train 0.8 --val 0.1 --test 0.1

-------------------------------------------------------------------------------
Inputs
-------------------------------------------------------------------------------
- aligned_dir:
    Directory containing aligned data:
        features.npy          : NxD matrix of raw CSI features
        manifest.jsonl        : JSON lines with image metadata

- train, val, test:
    Fractions of total dataset for each split.
    Must sum (approximately) to 1.0.

-------------------------------------------------------------------------------
Outputs
-------------------------------------------------------------------------------
Saved to `out_dir`:

| File | Description |
|------|--------------|
| features_norm.npy | Normalized feature matrix (same order as manifest) |
| norm_mu.npy | Per-feature mean (from training set) |
| norm_sigma.npy | Per-feature std (from training set) |
| train.txt / val.txt / test.txt | Image lists for each split |
| idx_train.npy / idx_val.npy / idx_test.npy | Row indices for splits |

-------------------------------------------------------------------------------
Assumptions
-------------------------------------------------------------------------------
- The number of rows in `features.npy` equals the number of lines in `manifest.jsonl`.
- Random shuffling uses a **fixed seed (42)** for reproducibility.
- Mean and std are computed **only** from the training subset.
===============================================================================
"""

import argparse
import json
import os
import numpy as np


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def main():
    """Main entry point for dataset normalization and splitting."""
    # -----------------------------------------------------------------------
    # Argument parsing
    # -----------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Normalize CSI features and split dataset.")
    ap.add_argument("--aligned_dir", required=True,
                    help="Path to directory containing aligned features and manifest.jsonl.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for normalized data and split lists.")
    ap.add_argument("--train", type=float, default=0.8,
                    help="Proportion of samples assigned to training split (default=0.8).")
    ap.add_argument("--val", type=float, default=0.1,
                    help="Proportion of samples assigned to validation split (default=0.1).")
    ap.add_argument("--test", type=float, default=0.1,
                    help="Proportion of samples assigned to test split (default=0.1).")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Load feature matrix and corresponding image paths
    # -----------------------------------------------------------------------
    X = np.load(os.path.join(args.aligned_dir, "features.npy"))  # shape: (N, D)

    with open(os.path.join(args.aligned_dir, "manifest.jsonl")) as f:
        imgs = [json.loads(line)["image"] for line in f]

    N = len(imgs)
    assert X.shape[0] == N, (
        f"Mismatch between features ({X.shape[0]}) and manifest entries ({N})"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Shuffle indices reproducibly
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)  # Fixed seed for deterministic splits
    idx = np.arange(N)
    rng.shuffle(idx)

    # -----------------------------------------------------------------------
    # Compute split sizes
    # -----------------------------------------------------------------------
    n_train = int(args.train * N)
    n_val = int(args.val * N)
    n_test = N - n_train - n_val  # Remainder goes to test set

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    # -----------------------------------------------------------------------
    # Compute normalization parameters from the training subset
    # -----------------------------------------------------------------------
    mu = X[idx_train].mean(axis=0, keepdims=True)          # (1, D)
    sigma = X[idx_train].std(axis=0, keepdims=True) + 1e-8 # (1, D) avoid divide-by-zero

    # Apply z-score normalization to the entire dataset
    Xn = (X - mu) / sigma

    # -----------------------------------------------------------------------
    # Save normalized data and parameters
    # -----------------------------------------------------------------------
    np.save(os.path.join(args.out_dir, "features_norm.npy"), Xn)
    np.save(os.path.join(args.out_dir, "norm_mu.npy"), mu)
    np.save(os.path.join(args.out_dir, "norm_sigma.npy"), sigma)

    # -----------------------------------------------------------------------
    # Helper function to save image lists
    # -----------------------------------------------------------------------
    def write_list(path: str, indices: np.ndarray):
        """
        Write a list of image paths corresponding to given indices.

        Args:
            path (str): Destination text file path.
            indices (np.ndarray): 1D array of row indices to write.
        """
        with open(path, "w") as f:
            for i in indices:
                f.write(imgs[i] + "\n")

    # Save lists of image paths for each split
    write_list(os.path.join(args.out_dir, "train.txt"), idx_train)
    write_list(os.path.join(args.out_dir, "val.txt"), idx_val)
    write_list(os.path.join(args.out_dir, "test.txt"), idx_test)

    # Also save the numeric indices for programmatic use
    np.save(os.path.join(args.out_dir, "idx_train.npy"), idx_train)
    np.save(os.path.join(args.out_dir, "idx_val.npy"), idx_val)
    np.save(os.path.join(args.out_dir, "idx_test.npy"), idx_test)

    # -----------------------------------------------------------------------
    # Summary and user guidance
    # -----------------------------------------------------------------------
    print("------------------------------------------------------------")
    print(f"Total samples: {N}")
    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    print(f"Normalized features saved → {args.out_dir}/features_norm.npy")
    print(f"Normalization parameters → {args.out_dir}/norm_mu.npy / norm_sigma.npy")
    print("Split lists generated → train.txt / val.txt / test.txt")
    print("Next step: run 04_resize_images.sh or your training pipeline.")
    print("------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
