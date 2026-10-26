"""
diffusion.py
============
Enterprise AI Meeting Assistant - Denoising Diffusion Model (DDPM + U-Net)

Matches block "4. Deep Learning Layer -> Diffusion Model" in the system
diagram: a U-Net trained as a denoising process, with iterative refinement
at generation time.

How it works:
    Forward process : gradually add Gaussian noise to a spectrogram over T steps
                      x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps
    Reverse process : a U-Net eps_theta(x_t, t) predicts the noise; sampling
                      starts from pure noise and iteratively refines it back to
                      a clean spectrogram (DDIM sampler -> ~50 steps instead of 1000)

Operates on the tensors produced by preprocessing.py
(N x 1 x N_MELS x N_FRAMES, values in [0, 1]). Internally the data is rescaled
to [-1, 1] (standard for diffusion) and mapped back to [0, 1] on output.

Same training recipe as autoencoder.py / vae.py:
    - Dropout in the U-Net residual blocks, Adam weight_decay
    - Gradient clipping
    - ReduceLROnPlateau keyed on validation loss
    - Early stopping that restores the BEST validation checkpoint
    - EMA (exponential moving average) of the weights -- strongly improves
      diffusion sample quality, so EMA weights are what we validate and save

Evaluation metrics (from the diagram's "Evaluation Engine"):
    - MSE (+ PSNR, MAE) : noise a held-out spectrogram part-way, denoise it back
                          with the iterative refinement loop, compare to the original
    - Perceptual metric : the diagram lists "LPC"; this is read as LPIPS-style
                          perceptual distance. Default = distance in the trained
                          Autoencoder's latent space (domain-specific, no downloads).
                          Pass --lpips to also compute true LPIPS (pip install lpips).
    - FID               : unconditional samples vs. real data (same AE features as gan.py)

Usage:
    python diffusion.py --epochs 100 --batch-size 16 --timesteps 1000 \
        --ddim-steps 50 --lr 2e-4 --patience 10
"""

import os
import math
import copy
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing import N_MELS, N_FRAMES, load_splits
from autoencoder import MelSpectrogramDataset, psnr
from gan import load_feature_extractor, extract_features, frechet_distance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"


# --------------------------------------------------------------------------
# Noise schedule + forward/reverse process
# --------------------------------------------------------------------------
class Diffusion:
    """Holds the cosine noise schedule, forward noising, and the DDIM sampler."""

    def __init__(self, timesteps=1000, device=DEVICE):
        self.T = timesteps
        self.device = device

        # cosine schedule (Nichol & Dhariwal, 2021) -- gentler than linear on small images
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        f = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        f = f / f[0]
        betas = (1 - (f[1:] / f[:-1])).clamp(0, 0.999)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        self.betas = betas.float().to(device)
        self.alphas_cumprod = alphas_cumprod.float().to(device)

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(a_bar) * x0 + sqrt(1 - a_bar) * noise."""
        a = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(self, model, x, t_start=None, steps=50):
        """
        Iterative refinement (deterministic DDIM, eta = 0).
        Starts from x at noise level t_start and denoises down to t = 0.
          - unconditional generation : x ~ N(0, I), t_start = T - 1
          - refinement / reconstruct : x = q_sample(x0, t_start)
        """
        if t_start is None:
            t_start = self.T - 1
        steps = max(2, min(steps, t_start + 1))
        ts = torch.linspace(t_start, 0, steps, device=x.device).round().long()

        for i in range(len(ts)):
            t = ts[i]
            t_prev = ts[i + 1] if i + 1 < len(ts) else None
            t_batch = torch.full((x.size(0),), int(t), device=x.device, dtype=torch.long)

            eps = model(x, t_batch)
            ab = self.alphas_cumprod[t]
            ab_prev = self.alphas_cumprod[t_prev] if t_prev is not None else torch.tensor(1.0, device=x.device)

            x0_pred = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
        return x


# --------------------------------------------------------------------------
# Model: U-Net noise predictor
# --------------------------------------------------------------------------
def timestep_embedding(t, dim):
    """Sinusoidal embedding of the diffusion step t (like Transformer positions)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]   # inject the noise level
        h = self.conv2(self.drop(self.act(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = self.norm(x).view(b, c, h * w).transpose(1, 2)   # (B, HW, C)
        out, _ = self.attn(seq, seq, seq, need_weights=False)
        return x + out.transpose(1, 2).view(b, c, h, w)


class UNet(nn.Module):
    """
    Input  (B, 1, N_MELS, N_FRAMES) noisy spectrogram + step t
    Output (B, 1, N_MELS, N_FRAMES) predicted noise
    Encoder: 32 -> 64 -> 128 channels (2x downsample each), self-attention at the
    bottleneck, decoder mirrors it with skip connections.
    """

    def __init__(self, base=32, dropout=0.1):
        super().__init__()
        assert N_MELS % 4 == 0 and N_FRAMES % 4 == 0, "N_MELS and N_FRAMES must be divisible by 4"
        self.base = base
        t_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        # encoder
        self.down1 = ResBlock(base, base, t_dim, dropout)                 # H x W
        self.pool1 = nn.Conv2d(base, base, 3, stride=2, padding=1)         # H/2
        self.down2 = ResBlock(base, base * 2, t_dim, dropout)
        self.pool2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1) # H/4

        # bottleneck
        self.mid1 = ResBlock(base * 2, base * 4, t_dim, dropout)
        self.mid_attn = SelfAttention(base * 4)
        self.mid2 = ResBlock(base * 4, base * 4, t_dim, dropout)

        # decoder
        self.up_conv1 = nn.Conv2d(base * 4, base * 4, 3, padding=1)
        self.up1 = ResBlock(base * 4 + base * 2, base * 2, t_dim, dropout)
        self.up_conv2 = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.up2 = ResBlock(base * 2 + base, base, t_dim, dropout)

        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    @staticmethod
    def _upsample(x, conv):
        return conv(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.base))

        h = self.in_conv(x)
        s1 = self.down1(h, t_emb)                       # (base,   H,   W)
        h = self.pool1(s1)
        s2 = self.down2(h, t_emb)                       # (2base,  H/2, W/2)
        h = self.pool2(s2)

        h = self.mid1(h, t_emb)                         # (4base,  H/4, W/4)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        h = self.up1(torch.cat([self._upsample(h, self.up_conv1), s2], dim=1), t_emb)
        h = self.up2(torch.cat([self._upsample(h, self.up_conv2), s1], dim=1), t_emb)
        return self.out(h)


class EMA:
    """Exponential moving average of model weights (used for validation + sampling)."""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))  # warm-up
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.mul_(d).add_(p.detach(), alpha=1 - d)


# --------------------------------------------------------------------------
# Helpers: scaling, generation, refinement
# --------------------------------------------------------------------------
def to_model_range(x):   # [0, 1] -> [-1, 1]
    return x * 2.0 - 1.0


def to_data_range(x):    # [-1, 1] -> [0, 1]
    return ((x + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def generate(model, diffusion, n, steps=50, batch_size=64):
    """Unconditional generation: pure noise -> spectrogram via iterative refinement."""
    model.eval()
    out = []
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        x = torch.randn(b, 1, N_MELS, N_FRAMES, device=DEVICE)
        out.append(to_data_range(diffusion.ddim_sample(model, x, diffusion.T - 1, steps)).cpu())
    return torch.cat(out)


@torch.no_grad()
def refine(model, diffusion, x0, t_start, steps=50, seed=0):
    """Noise real spectrograms to level t_start, then denoise them back (iterative refinement)."""
    model.eval()
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    x0 = to_model_range(x0.to(DEVICE))
    noise = torch.randn(x0.shape, generator=g, device=DEVICE)
    t = torch.full((x0.size(0),), t_start, device=DEVICE, dtype=torch.long)
    x_t = diffusion.q_sample(x0, t, noise)
    return to_data_range(diffusion.ddim_sample(model, x_t, t_start, steps))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model, diffusion, loader):
    """
    Validation noise-prediction loss. Uses a FIXED seed for timesteps/noise so
    the value is deterministic and comparable between epochs (otherwise the
    random t makes the metric too noisy for early stopping).
    """
    model.eval()
    g = torch.Generator(device=DEVICE)
    g.manual_seed(0)
    total, n = 0.0, 0
    for x in loader:
        x = to_model_range(x.to(DEVICE))
        t = torch.randint(0, diffusion.T, (x.size(0),), generator=g, device=DEVICE)
        noise = torch.randn(x.shape, generator=g, device=DEVICE)
        pred = model(diffusion.q_sample(x, t, noise), t)
        total += nn.functional.mse_loss(pred, noise, reduction="sum").item()
        n += x.numel()
    return total / n


def compute_lpips(real, recon):
    """Optional true LPIPS (pip install lpips). Trained on natural images -> use as a secondary signal."""
    try:
        import lpips
        fn = lpips.LPIPS(net="alex", verbose=False).to(DEVICE).eval()
        scores = []
        with torch.no_grad():
            for i in range(0, real.size(0), 32):
                a = (real[i:i + 32].to(DEVICE) * 2 - 1).repeat(1, 3, 1, 1)
                b = (recon[i:i + 32].to(DEVICE) * 2 - 1).repeat(1, 3, 1, 1)
                scores.append(fn(a, b).flatten().cpu())
        return torch.cat(scores).mean().item()
    except Exception as e:
        print(f"[WARN] LPIPS unavailable ({e}); skipping.")
        return None


@torch.no_grad()
def evaluate_refinement(model, diffusion, data, encoder, t_start, steps, use_lpips=False, batch_size=64):
    """
    Reconstruction quality of the denoising loop: MSE / MAE / PSNR plus a
    perceptual distance in the trained AE's latent space ("Feature_Distance",
    RMS per latent dimension, lower = better).
    """
    recons = torch.cat([
        refine(model, diffusion, data[i:i + batch_size], t_start, steps).cpu()
        for i in range(0, data.size(0), batch_size)
    ])
    mse = nn.functional.mse_loss(recons, data).item()
    mae = nn.functional.l1_loss(recons, data).item()

    f_real = torch.from_numpy(extract_features(encoder, data))
    f_rec = torch.from_numpy(extract_features(encoder, recons))
    feat_dist = ((f_real - f_rec) ** 2).mean().sqrt().item()

    metrics = {"MSE": mse, "MAE": mae, "PSNR": psnr(mse), "Feature_Distance": feat_dist}
    if use_lpips:
        metrics["LPIPS"] = compute_lpips(data, recons)
    return metrics, recons


def save_sample_grid(real, recon, fake, path, n=6):
    """Rows: real | refined reconstruction | unconditional sample (needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping sample grid.")
        return
    rows = [("Real", real), ("Refined", recon), ("Generated", fake)]
    fig, axes = plt.subplots(len(rows), n, figsize=(2.2 * n, 1.9 * len(rows)))
    for r, (name, tensor) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(tensor[i, 0].numpy(), origin="lower", aspect="auto", cmap="magma")
            axes[r, i].axis("off")
        axes[r, 0].set_title(name, loc="left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved sample grid to {path}")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_diffusion(
    epochs=100,
    batch_size=16,
    base_channels=32,
    timesteps=1000,
    ddim_steps=50,
    lr=2e-4,
    weight_decay=1e-5,
    dropout=0.1,
    grad_clip=1.0,
    ema_decay=0.995,
    patience=10,
    min_delta=1e-5,
    refine_t=250,
    fid_samples=500,
    use_lpips=False,
):
    """
    Early stopping / best-checkpoint selection is based on the validation
    noise-prediction loss of the EMA model (fixed-seed, so it is deterministic).

    refine_t    : noise level (out of `timesteps`) used for the denoise-and-compare
                  reconstruction metrics. Larger = harder task.
    fid_samples : number of unconditional samples for the final FID (0 to skip;
                  sampling is slow on CPU).
    """
    X_train, X_val, X_test = load_splits()

    train_loader = DataLoader(MelSpectrogramDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MelSpectrogramDataset(X_val), batch_size=batch_size)
    test_loader = DataLoader(MelSpectrogramDataset(X_test), batch_size=batch_size)

    diffusion = Diffusion(timesteps)
    model = UNet(base=base_channels, dropout=dropout).to(DEVICE)
    ema = EMA(model, decay=ema_decay)
    print(f"[Diffusion] U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 3)
    )

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x in train_loader:
            x = to_model_range(x.to(DEVICE))
            t = torch.randint(0, timesteps, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)

            optimizer.zero_grad()
            pred = model(diffusion.q_sample(x, t, noise), t)
            loss = nn.functional.mse_loss(pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_loss(ema.model, diffusion, val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(ema.model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        flag = " *" if improved else ""
        print(
            f"[Diffusion] Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.5f} | "
            f"val_loss(EMA)={val_loss:.5f}{flag} | lr={current_lr:.2e}"
        )

        if epochs_without_improvement >= patience:
            print(f"[Diffusion] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    if best_state is not None:
        ema.model.load_state_dict(best_state)
        print(f"\nRestored best model (val_loss={best_val_loss:.5f}).")
    best_model = ema.model

    # ---- Final test evaluation ----
    print("\nFinal Test Evaluation (best checkpoint, EMA weights):")
    encoder = load_feature_extractor()
    test_metrics = {"Noise_Pred_Loss": evaluate_loss(best_model, diffusion, test_loader)}

    refine_metrics, recons = evaluate_refinement(
        best_model, diffusion, X_test, encoder, refine_t, ddim_steps, use_lpips
    )
    test_metrics.update(refine_metrics)

    fake = None
    if fid_samples > 0:
        n = min(fid_samples, X_test.size(0))
        print(f"Generating {n} unconditional samples for FID ({ddim_steps} DDIM steps)...")
        fake = generate(best_model, diffusion, n, ddim_steps)
        test_metrics["FID"] = frechet_distance(
            extract_features(encoder, X_test[:n]), extract_features(encoder, fake)
        )

    for k, v in test_metrics.items():
        if v is not None:
            print(f"  {k}: {v:.5f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "diffusion.pt")
    torch.save(
        {
            "model_state": best_model.state_dict(),   # EMA weights
            "base_channels": base_channels,
            "timesteps": timesteps,
            "dropout": dropout,
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    print(f"Saved Diffusion checkpoint to {ckpt_path}")

    if fake is None:
        fake = generate(best_model, diffusion, 6, ddim_steps)
    save_sample_grid(X_test[:6], recons[:6], fake[:6], os.path.join(OUTPUT_DIR, "diffusion_samples.png"))

    return best_model, history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Diffusion model (DDPM + U-Net)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=50, help="sampling steps (iterative refinement)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--refine-t", type=int, default=250, help="noise level for the denoise-and-compare MSE test")
    parser.add_argument("--fid-samples", type=int, default=500, help="0 to skip FID")
    parser.add_argument("--lpips", action="store_true", help="also compute true LPIPS (needs `pip install lpips`)")
    args = parser.parse_args()

    train_diffusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        timesteps=args.timesteps,
        ddim_steps=args.ddim_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        patience=args.patience,
        refine_t=args.refine_t,
        fid_samples=args.fid_samples,
        use_lpips=args.lpips,
    )
