#!/usr/bin/env python3
"""
evaluate_latent_diffusion.py
===============================================================================
Purpose
===============================================================================
Evaluate and visualize the performance of a CSI→Latent→Image reconstruction
pipeline using both:
  1. Direct VAE decoding from predicted latents ("mapper-only")
  2. Full image refinement via Stable Diffusion’s Img2Img pipeline ("diffusion")

The script compares reconstructions to ground truth frames and computes
quantitative metrics (PSNR, SSIM, LPIPS). It also saves qualitative
side-by-side comparison grids.

===============================================================================
Pipeline Context
===============================================================================
Stage: 09 — Full Reconstruction & Evaluation

Dependencies from previous stages:
  - train_csi2latent.py         → trained latent mapper (best.pt)
  - encode_latents.py           → precomputed VAE latents
  - 03_normalize_and_split.py   → normalized CSI features + indices

This script:
  - Loads CSI features for a split (val/test)
  - Loads trained CSI→latent MLP
  - Decodes predicted latents through the Stable Diffusion VAE
  - Optionally refines reconstructions via Stable Diffusion Img2Img
  - Computes metrics vs. original images
  - Saves images and metrics report

===============================================================================
Usage
===============================================================================
    python evaluate_latent_diffusion.py \
        --frames_root dataset_out/frames \
        --splits_dir dataset_out/splits \
        --latents_dir dataset_out/latents \
        --ckpt runs/latentcsi/best.pt \
        --model runwayml/stable-diffusion-v1-5 \
        --split val \
        --steps 20 \
        --cfg 5.0 \
        --strength 0.2 \
        --prompt "a photorealistic indoor room, natural lighting" \
        --neg "blurry, low quality, watermark, text" \
        --out runs/eval_latentcsi

===============================================================================
Inputs
===============================================================================
- --frames_root :
    Directory containing ground-truth frame images.

- --splits_dir :
    Directory containing dataset splits, normalized features, and index arrays.

- --latents_dir :
    Directory with VAE latent encodings (used to match split indices).

- --ckpt :
    Trained CSI→Latent model checkpoint (.pt) containing:
        { "model", "d_in", "hidden" }

- --model :
    Hugging Face Stable Diffusion model identifier (default: runwayml/stable-diffusion-v1-5).

- --split :
    Split to evaluate ("val" or "test").

- --steps / --cfg / --strength :
    Parameters controlling Stable Diffusion refinement:
        steps      → number of denoising steps
        cfg        → classifier-free guidance scale
        strength   → noise injection amount (0 = no change)

- --prompt / --neg :
    Positive and negative text prompts for diffusion guidance.

===============================================================================
Outputs
===============================================================================
- Directory: `runs/eval_latentcsi/`
    ├── grids_val/ or grids_test/   → image comparison grids [GT | mapper | diffusion]
    ├── metrics_val.json            → averaged metrics (JSON)
    └── printed JSON summary

===============================================================================
Metrics
===============================================================================
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (tiny approximate implementation)
- LPIPS (learned perceptual similarity using VGG backbone)

===============================================================================
Requirements
===============================================================================
- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision
- diffusers
- lpips
- tqdm
- Pillow
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
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)


# =============================================================================
# Utility Functions
# =============================================================================

def ssim_tiny(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Lightweight (windowless) SSIM approximation.

    Args:
        x, y (torch.Tensor): Images in [0,1] range, same shape.

    Returns:
        torch.Tensor: Scalar SSIM approximation (higher = better).
    """
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )


# -----------------------------------------------------------------------------
# Inline CSI→Latent MLP (must match train_csi2latent.py)
# -----------------------------------------------------------------------------
class CSI2LatentMLP(nn.Module):
    """
    Small fully connected mapper:
        Input  : CSI feature vector (D,)
        Output : Predicted latent tensor (4×64×64)
    """
    def __init__(self, d_in, z_shape=(4, 64, 64), hidden=2048):
        super().__init__()
        self.z_shape = z_shape
        z_flat = int(np.prod(z_shape))
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, z_flat),
        )

    def forward(self, x):
        y = self.net(x)
        b = x.shape[0]
        return y.view(b, *self.z_shape)


def to_tensor_01(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized torch tensor in [0,1] with shape (1,3,H,W)."""
    return T.ToTensor()(img).unsqueeze(0)


def pil_from_tensor_01(x: torch.Tensor) -> Image.Image:
    """Convert torch tensor in [0,1] back to a PIL RGB image."""
    x = (x.clamp(0, 1) * 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


# =============================================================================
# Main Evaluation Routine
# =============================================================================

def main():
    """
    Evaluate CSI→Latent reconstructions (mapper + diffusion) vs. ground truth.
    """
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Evaluate CSI→Latent→Image reconstruction pipeline.")
    ap.add_argument("--frames_root", default="dataset_out/frames", help="Path to original frames.")
    ap.add_argument("--splits_dir", default="dataset_out/splits", help="Path to splits directory.")
    ap.add_argument("--latents_dir", default="dataset_out/latents", help="Directory of saved latents.")
    ap.add_argument("--ckpt", default="runs/latentcsi/best.pt", help="Trained CSI→Latent checkpoint.")
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model ID or path.")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Split to evaluate (default: val).")
    ap.add_argument("--steps", type=int, default=20, help="Diffusion denoising steps.")
    ap.add_argument("--cfg", type=float, default=5.0, help="Classifier-free guidance scale.")
    ap.add_argument("--strength", type=float, default=0.2, help="Noise strength for Img2Img (0=no change).")
    ap.add_argument("--prompt", default="a photorealistic indoor room, natural lighting", help="Positive prompt.")
    ap.add_argument("--neg", default="blurry, low quality, watermark, text", help="Negative prompt.")
    ap.add_argument("--out", default="runs/eval_latentcsi", help="Output directory for results.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # -------------------------------------------------------------------------
    # Load data split and corresponding CSI features
    # -------------------------------------------------------------------------
    with open(os.path.join(args.splits_dir, f"{args.split}.txt")) as f:
        paths = [l.strip() for l in f if l.strip()]

    idx = np.load(os.path.join(args.splits_dir, f"idx_{args.split}.npy"))
    X = np.load(os.path.join(args.splits_dir, "features_norm.npy")).astype("float32")[idx]

    # -------------------------------------------------------------------------
    # Load trained CSI→Latent mapper
    # -------------------------------------------------------------------------
    ck = torch.load(args.ckpt, map_location="cpu")
    mapper = CSI2LatentMLP(ck["d_in"], (4, 64, 64), hidden=ck["hidden"]).to(device).eval()
    mapper.load_state_dict(ck["model"])

    # -------------------------------------------------------------------------
    # Load Stable Diffusion components
    # -------------------------------------------------------------------------
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device).eval()

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device.type == "cuda":
        pipe.to(dtype=dtype)

    # -------------------------------------------------------------------------
    # Initialize LPIPS perceptual metric
    # -------------------------------------------------------------------------
    import lpips
    lp = lpips.LPIPS(net="vgg").to(device).eval()

    # Metric accumulators
    psnr_m = ssim_m = lpips_m = 0.0  # mapper-only
    psnr_d = ssim_d = lpips_d = 0.0  # diffusion
    N = len(paths)

    # Output directory for visual comparisons
    grid_dir = os.path.join(args.out, f"grids_{args.split}")
    os.makedirs(grid_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------------
    for i, (p, xrow) in enumerate(tqdm(list(zip(paths, X)), total=N, desc=f"eval {args.split}")):
        # --- Load ground truth image ---
        p_abs = p if os.path.isabs(p) else os.path.join(args.frames_root, p)
        gt = Image.open(p_abs).convert("RGB")
        gt_t = to_tensor_01(gt).to(device)

        # --- Mapper-only reconstruction ---
        with torch.no_grad():
            z = mapper(torch.from_numpy(xrow).unsqueeze(0).to(device))
            rec = vae.decode((z / 0.18215).to(dtype)).sample  # Decode latent to image [-1,1]
            rec01 = (rec.clamp(-1, 1) + 1) / 2                # Normalize to [0,1]

        # --- Diffusion refinement (Img2Img) ---
        init_pil = pil_from_tensor_01(rec01)
        out = pipe(
            prompt=args.prompt,
            negative_prompt=args.neg,
            image=init_pil,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
        )
        dif_pil = out.images[0]
        dif_t = to_tensor_01(dif_pil).to(device)

        # --- Compute metrics ---
        def _psnr(a, b, eps=1e-8): 
            return (-10 * torch.log10(((a - b) ** 2).mean() + eps)).item()

        psnr_m += _psnr(rec01, gt_t)
        ssim_m += ssim_tiny(rec01, gt_t).item()
        psnr_d += _psnr(dif_t, gt_t)
        ssim_d += ssim_tiny(dif_t, gt_t).item()

        with torch.no_grad():
            lpips_m += lp(rec01 * 2 - 1, gt_t * 2 - 1).item()
            lpips_d += lp(dif_t * 2 - 1, gt_t * 2 - 1).item()

        # --- Save comparison grid: [GT | Mapper | Diffusion] ---
        grid = torch.cat([gt_t, rec01, dif_t], dim=0)
        vutils.save_image(grid, os.path.join(grid_dir, f"cmp_{i:05d}.jpg"), nrow=3)

    # -------------------------------------------------------------------------
    # Aggregate metrics and save report
    # -------------------------------------------------------------------------
    stats = {
        "split": args.split,
        "N": N,
        "mapper": {"PSNR": psnr_m / N, "SSIM_approx": ssim_m / N, "LPIPS": lpips_m / N},
        "diffuse": {"PSNR": psnr_d / N, "SSIM_approx": ssim_d / N, "LPIPS": lpips_d / N},
        "cfg": args.cfg,
        "strength": args.strength,
        "steps": args.steps,
        "prompt": args.prompt,
    }

    print(json.dumps(stats, indent=2))
    with open(os.path.join(args.out, f"metrics_{args.split}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Evaluation complete. Results saved to {args.out}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
