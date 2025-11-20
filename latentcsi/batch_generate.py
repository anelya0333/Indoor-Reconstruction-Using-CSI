#!/usr/bin/env python3
"""
batch_demo_latent_diffusion.py
===============================================================================
Purpose
===============================================================================
Generate a batch of reconstructed images from CSI features using:
  1. A trained CSI→Latent MLP mapper.
  2. A Stable Diffusion Img2Img pipeline for refinement.

This script demonstrates how CSI signal embeddings can be converted back into
realistic RGB images by leveraging the learned latent-space mapping and
the generative prior from Stable Diffusion.

===============================================================================
Pipeline Context
===============================================================================
Stage: 10 — Batch Demonstration / Visualization

Upstream dependencies:
  - train_csi2latent.py        → trained CSI→latent MLP checkpoint (.pt)
  - 03_normalize_and_split.py  → normalized CSI feature splits
  - encode_latents.py          → generated latent encodings for reference

This script:
  • Loads normalized CSI features and corresponding image list.
  • Uses the trained mapper to predict latents.
  • Decodes the latents with Stable Diffusion’s VAE.
  • Optionally refines them with the full Stable Diffusion Img2Img pipeline.
  • Saves a batch of ground truth, mapper reconstructions, and refined images.

===============================================================================
Usage
===============================================================================
    python batch_demo_latent_diffusion.py \
        --ckpt runs/latentcsi/best.pt \
        --features_norm dataset_out/splits/features_norm.npy \
        --list_txt dataset_out/splits/test.txt \
        --frames_root dataset_out/frames \
        --model runwayml/stable-diffusion-v1-5 \
        --prompt "a photorealistic indoor room, natural lighting" \
        --neg "blurry, text, watermark, low quality" \
        --steps 20 \
        --cfg 5.0 \
        --strength 0.2 \
        --count 24 \
        --outdir runs/batch_demo

===============================================================================
Inputs
===============================================================================
- --ckpt :
    Trained CSI→latent model checkpoint (.pt) with:
      { "model": state_dict, "d_in": int, "hidden": int }

- --features_norm :
    Path to normalized CSI feature matrix (features_norm.npy).

- --list_txt :
    Path to text file listing image paths for the evaluation split.

- --frames_root :
    Root directory for image frames referenced in list_txt.

- --model :
    Hugging Face model ID or local path for Stable Diffusion (default: runwayml/stable-diffusion-v1-5).

- --prompt / --neg :
    Text prompts guiding diffusion refinement (positive/negative).

- --steps / --cfg / --strength :
    Diffusion parameters:
      steps     → number of denoising steps.
      cfg       → classifier-free guidance scale.
      strength  → noise strength for Img2Img (0=noise-free).

- --count :
    Number of examples to generate (default: 24).

- --outdir :
    Directory for saving results (default: runs/batch_demo).

===============================================================================
Outputs
===============================================================================
Saved to `--outdir`:
    ├── 000_gt.jpg           → Ground truth frame
    ├── 000_mapper.jpg       → Mapper-only VAE decode
    ├── 000_latentcsi.jpg    → Final diffusion-refined reconstruction
    ├── 001_gt.jpg, ... etc.

===============================================================================
Requirements
===============================================================================
- Python ≥ 3.8
- PyTorch ≥ 2.0
- diffusers ≥ 0.21.0
- torchvision
- Pillow
- tqdm
===============================================================================
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, DPMSolverMultistepScheduler
import torchvision.transforms as T


# =============================================================================
# Model Definition: CSI → Latent Mapper
# =============================================================================
class CSI2LatentMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) mapping CSI feature vectors to VAE latent tensors.

    Architecture:
      Input:  D-dimensional CSI vector
      Hidden: 2 fully connected layers with SiLU activation
      Output: Flattened latent vector reshaped to (4×64×64)

    Args:
        d_in (int): Input feature dimension.
        z_shape (tuple): Shape of the output latent tensor. Default = (4, 64, 64).
        hidden (int): Hidden layer size. Default = 2048.
    """
    def __init__(self, d_in, z_shape=(4, 64, 64), hidden=2048):
        super().__init__()
        self.z_shape = z_shape
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, int(np.prod(z_shape))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: CSI → latent.

        Args:
            x (torch.Tensor): Input CSI features of shape (B, D).

        Returns:
            torch.Tensor: Predicted latent tensor of shape (B, 4, 64, 64).
        """
        b = x.size(0)
        y = self.net(x)
        return y.view(b, *self.z_shape)


# =============================================================================
# Utility Functions
# =============================================================================
def to_tensor_01(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a torch tensor in [0,1] range.

    Returns:
        torch.Tensor: Image tensor with shape (1, 3, H, W).
    """
    return T.ToTensor()(img).unsqueeze(0)


# =============================================================================
# Main Routine
# =============================================================================
def main():
    """
    Batch reconstruction and visualization:
      - Predict latent from CSI features.
      - Decode via Stable Diffusion VAE.
      - Refine via Img2Img diffusion.
      - Save all outputs for comparison.
    """
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Batch demo of CSI→Image reconstruction via Stable Diffusion.")
    ap.add_argument("--ckpt", default="runs/latentcsi/best.pt", help="Trained CSI→latent checkpoint.")
    ap.add_argument("--features_norm", default="dataset_out/splits/features_norm.npy", help="Normalized CSI features.")
    ap.add_argument("--list_txt", default="dataset_out/splits/test.txt", help="Image list for evaluation split.")
    ap.add_argument("--frames_root", default="dataset_out/frames", help="Root directory for frame images.")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model identifier.")
    ap.add_argument("--prompt", default="a photorealistic indoor room, natural lighting", help="Positive text prompt.")
    ap.add_argument("--neg", default="blurry, text, watermark, low quality", help="Negative text prompt.")
    ap.add_argument("--steps", type=int, default=20, help="Number of diffusion denoising steps.")
    ap.add_argument("--cfg", type=float, default=5.0, help="Classifier-free guidance scale.")
    ap.add_argument("--strength", type=float, default=0.2, help="Img2Img noise strength.")
    ap.add_argument("--count", type=int, default=24, help="Number of samples to process.")
    ap.add_argument("--outdir", default="runs/batch_demo", help="Output directory for generated images.")
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Setup environment
    # -------------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    with open(args.list_txt) as f:
        imgs = [l.strip() for l in f if l.strip()]

    # Load normalized features and corresponding split indices
    X = np.load(args.features_norm).astype("float32")
    idx_path = os.path.join(
        os.path.dirname(args.list_txt),
        "idx_test.npy" if "test" in args.list_txt else "idx_val.npy"
    )
    idx = np.load(idx_path)
    X = X[idx]  # select subset

    # -------------------------------------------------------------------------
    # Load trained mapper and Stable Diffusion components
    # -------------------------------------------------------------------------
    ck = torch.load(args.ckpt, map_location="cpu")
    mapper = CSI2LatentMLP(ck["d_in"], (4, 64, 64), hidden=ck["hidden"]).to(device).eval()
    mapper.load_state_dict(ck["model"])

    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device).eval()
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if device.type == "cuda":
        pipe.to(dtype=dtype)

    # -------------------------------------------------------------------------
    # Batch processing loop
    # -------------------------------------------------------------------------
    for i in tqdm(range(min(args.count, len(imgs))), desc="Batch demo"):
        # Load ground truth image
        p = imgs[i] if os.path.isabs(imgs[i]) else os.path.join(args.frames_root, imgs[i])
        gt = Image.open(p).convert("RGB")

        # Predict latent from CSI
        x = torch.from_numpy(X[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            z = mapper(x)
            rec = vae.decode((z / 0.18215).to(dtype)).sample  # Decode latent → image [-1,1]
            rec01 = (rec.clamp(-1, 1) + 1) / 2                # Normalize to [0,1]

        # Prepare initial PIL image for diffusion
        init_pil = Image.fromarray(
            (rec01.squeeze(0).permute(1, 2, 0).mul(255).byte().cpu().numpy())
        )

        # Refine reconstruction via Stable Diffusion Img2Img
        out = pipe(
            prompt=args.prompt,
            negative_prompt=args.neg,
            image=init_pil,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
        )
        out_img = out.images[0]

        # Save outputs for comparison
        gt.save(os.path.join(args.outdir, f"{i:03d}_gt.jpg"))
        init_pil.save(os.path.join(args.outdir, f"{i:03d}_mapper.jpg"))
        out_img.save(os.path.join(args.outdir, f"{i:03d}_latentcsi.jpg"))

    print(f"✅ Batch demo complete. Results saved to: {args.outdir}")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    main()
