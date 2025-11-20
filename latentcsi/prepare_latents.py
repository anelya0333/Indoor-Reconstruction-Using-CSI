#!/usr/bin/env python3
"""
encode_latents.py
===============================================================================
Purpose
===============================================================================
Convert video frame images into compressed latent representations using the
VAE (Variational Autoencoder) component from a pretrained Stable Diffusion model.

This script prepares **latent features** for downstream multimodal learning
(e.g., CSI→latent mapping), dramatically reducing image dimensionality
while retaining perceptual information.

===============================================================================
Pipeline Context
===============================================================================
Stage: 07 — Frame Latent Encoding

Previous stages:
  00–04 : Data extraction, alignment, and preprocessing
  train_baseline.py : CSI→Image model training
  evaluate_csi2image.py : Model evaluation

This script:
  • Loads each frame image.
  • Applies the VAE encoder from Stable Diffusion.
  • Saves latent tensors (4×64×64 per 512×512 image) for each data split.

===============================================================================
Usage
===============================================================================
    python encode_latents.py \
        --frames_root dataset_out/frames \
        --splits_dir dataset_out/splits \
        --model runwayml/stable-diffusion-v1-5 \
        --out_dir dataset_out/latents

===============================================================================
Inputs
===============================================================================
- --frames_root :
    Directory containing frame images to be encoded (e.g., dataset_out/frames).

- --splits_dir :
    Directory containing text files that list the images belonging to each split:
      train.txt, val.txt, test.txt

- --model :
    Hugging Face model identifier or local path for a Stable Diffusion checkpoint.
    Default: "runwayml/stable-diffusion-v1-5"

- --out_dir :
    Destination directory for output latent .npy files.

===============================================================================
Outputs
===============================================================================
Saves one file per split in `out_dir`:
    latents_train.npy   → (N_train, 4, 64, 64)
    latents_val.npy     → (N_val, 4, 64, 64)
    latents_test.npy    → (N_test, 4, 64, 64)

Each latent corresponds to a 512×512 image encoded through the Stable Diffusion VAE.

===============================================================================
Requirements
===============================================================================
- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- diffusers ≥ 0.21.0  
- torchvision, Pillow, numpy, tqdm  
- GPU recommended for speed (CPU fallback supported)

===============================================================================
Notes
===============================================================================
- Stable Diffusion VAE scaling factor = **0.18215**
  (required to match latent magnitudes expected by the diffusion model).
- Output latent tensors are stored as `float32` NumPy arrays.
- Uses tqdm progress bars for user feedback during encoding.
===============================================================================
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
import torchvision.transforms as T


# =============================================================================
# Helper Functions
# =============================================================================

def load_image(p: str) -> Image.Image:
    """
    Load an image from disk and convert to RGB.

    Args:
        p (str): Path to the image file.

    Returns:
        PIL.Image.Image: Loaded RGB image.
    """
    return Image.open(p).convert("RGB")


# =============================================================================
# Main Routine
# =============================================================================

def main():
    """
    Encode video frame images into Stable Diffusion VAE latent space.

    Workflow:
      1. Load pretrained VAE model from Hugging Face or local cache.
      2. Read image lists for train/val/test splits.
      3. For each image, apply resizing, normalization, and encoding.
      4. Save resulting latent arrays (.npy) for each split.

    Each encoded latent is shape (4, 64, 64) per 512×512 input image.
    """
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Encode frames into Stable Diffusion VAE latents.")
    ap.add_argument("--frames_root", default="dataset_out/frames",
                    help="Directory containing raw frame images.")
    ap.add_argument("--splits_dir", default="dataset_out/splits",
                    help="Directory containing split text files (train.txt, val.txt, test.txt).")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5",
                    help="Stable Diffusion model identifier or path (default: runwayml/stable-diffusion-v1-5).")
    ap.add_argument("--out_dir", default="dataset_out/latents",
                    help="Output directory for .npy latent files.")
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Environment setup
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained VAE from Stable Diffusion
    # The subfolder 'vae' contains only the autoencoder component
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device)
    vae.eval()  # Disable dropout and batch norm updates

    # -------------------------------------------------------------------------
    # Image preprocessing pipeline
    # -------------------------------------------------------------------------
    # Converts to 512×512, scales to [0,1], then normalizes to [-1,1]
    tfm = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Process each dataset split
    # -------------------------------------------------------------------------
    for split in ["train", "val", "test"]:
        list_txt = os.path.join(args.splits_dir, f"{split}.txt")
        if not os.path.exists(list_txt):
            print(f"⚠️  Warning: Missing split file: {list_txt}. Skipping {split}.")
            continue

        # Read image list (relative or absolute paths)
        with open(list_txt) as f:
            rels = [line.strip() for line in f if line.strip()]

        Z = []  # Latent collection for this split

        for rel in tqdm(rels, desc=f"VAE encode {split}", ncols=80):
            # Resolve relative → absolute paths
            p = Path(rel)
            if not p.is_absolute():
                p = Path(args.frames_root) / rel

            # Load and preprocess image
            x = tfm(load_image(str(p))).unsqueeze(0).to(device)  # (1,3,512,512)

            # Encode into latent space
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.mean  # (1,4,64,64)
                latent = 0.18215 * latent                # Stable Diffusion latent scaling

            Z.append(latent.cpu().numpy())

        # Concatenate all latents for this split and save
        if Z:
            Z = np.concatenate(Z, axis=0)  # (N,4,64,64)
            np.save(os.path.join(args.out_dir, f"latents_{split}.npy"), Z)
            print(f"✅ Saved {split} latents: {Z.shape}")
        else:
            print(f"⚠️  No images found for split: {split}")

    print(f"All splits processed. Latents saved to: {args.out_dir}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
