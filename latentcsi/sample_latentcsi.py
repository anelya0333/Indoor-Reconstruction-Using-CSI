#!/usr/bin/env python3
import argparse, numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, DPMSolverMultistepScheduler

# --- inline mapper (avoid imports) ---
import torch.nn as nn
class CSI2LatentMLP(nn.Module):
    def __init__(self, d_in, z_shape=(4,64,64), hidden=2048):
        super().__init__()
        z_flat = int(np.prod(z_shape))
        self.z_shape = z_shape
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, z_flat),
        )
    def forward(self, x):
        y = self.net(x)
        b = x.shape[0]
        return y.view(b, *self.z_shape)

def tensor_to_pil(x):
    # x is in [-1,1], shape (B,3,H,W)
    x = (x.clamp(-1,1) + 1) / 2
    x = (x * 255).round().byte().cpu().permute(0,2,3,1).numpy()
    return [Image.fromarray(a) for a in x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--ckpt", default="runs/latentcsi/best.pt")
    ap.add_argument("--features_norm", default="dataset_out/splits/features_norm.npy")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--prompt", default="a photorealistic indoor room, high detail, natural lighting")
    ap.add_argument("--neg", default="blurry, text, watermark, low quality")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--strength", type=float, default=0.2)  # 0..1
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="latentcsi_out.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    # 1) Load mapper and predict latent z0 (scaled by 0.18215 during training)
    ck = torch.load(args.ckpt, map_location="cpu")
    D, hidden = ck["d_in"], ck["hidden"]
    mapper = CSI2LatentMLP(D, (4,64,64), hidden=hidden).to(device).eval()
    mapper.load_state_dict(ck["model"])

    X = np.load(args.features_norm).astype("float32")
    x = torch.from_numpy(X[args.row]).unsqueeze(0).to(device)

    with torch.no_grad():
        z0 = mapper(x)  # (1,4,64,64), already scaled (0.18215*)

    # 2) Decode latent → image using the SD VAE
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device)
    vae.eval()
    with torch.no_grad():
        # VAE expects unscaled latent; undo the SD scaling factor
        lat = z0 / 0.18215
        img_t = vae.decode(lat.to(dtype)).sample  # (1,3,512,512) in [-1,1]
    init_pil = tensor_to_pil(img_t)[0]  # PIL image

    # 3) Run SD img2img using the decoded image as init
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model)
    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if device.type == "cuda":
        pipe.to(dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.neg,
        image=init_pil,                 # start from our decoded CSI image
        strength=args.strength,         # how much to deviate (0.15–0.35 good range)
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator
    )
    out_img = result.images[0]
    out_img.save(args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
