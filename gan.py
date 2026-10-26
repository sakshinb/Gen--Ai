"""
gan.py
======
Enterprise AI Meeting Assistant - Generative Adversarial Network (GAN)

Learns to generate realistic Mel-spectrogram chunks from random noise z ~ N(0, I)
by playing a two-player game:
    Generator     G(z)  -> fake spectrogram (1 x N_MELS x N_FRAMES)
    Discriminator D(x)  -> logit: "is this spectrogram real?"

Operates on the same tensors produced by preprocessing.py
(shape: N x 1 x N_MELS x N_FRAMES, values min-max normalized to [0, 1]),
and reuses MelSpectrogramDataset from autoencoder.py.

Design choices (DCGAN-style, with stabilizers because GANs are fragile on
small datasets such as our ~3000-chunk prototype subset):
    - Generator mirrors the AE/VAE decoder (FC -> 3x ConvTranspose -> Sigmoid)
    - Discriminator uses Spectral Normalization + LeakyReLU + Dropout2d
      (no BatchNorm) to keep D from overpowering G
    - Non-saturating generator loss, BCE-with-logits for numerical stability
    - One-sided label smoothing (real label = 0.9)
    - Adam(lr=2e-4, betas=(0.5, 0.999)) -- standard GAN setting
    - Gradient clipping on both networks
    - Best checkpoint chosen by validation FID (lower = better), NOT by loss,
      because GAN losses are not a reliable measure of sample quality

Evaluation metrics:
    - FID (Frechet Inception Distance, computed on features from the trained
      Autoencoder's encoder -- the standard Inception network is built for
      natural RGB images and is not meaningful for Mel spectrograms)
    - D_Real_Acc / D_Fake_Acc : discriminator balance (healthy ~ 0.5-0.8)
    - Diversity : mean pairwise L1 distance between generated samples,
      compared to the real data (a much lower value => mode collapse)

Usage:
    python gan.py --epochs 100 --batch-size 32 --latent-dim 128 \
        --lr 2e-4 --dropout 0.1 --patience 10
"""

import os
import copy
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader

from preprocessing import N_MELS, N_FRAMES, load_splits
from autoencoder import MelSpectrogramDataset, Autoencoder  # reuse dataset + AE (for FID features)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Generator(nn.Module):
    """z (latent_dim) -> (1, N_MELS, N_FRAMES) spectrogram in [0, 1]."""

    def __init__(self, latent_dim=128, conv_shape=(128, math.ceil(N_MELS / 8), math.ceil(N_FRAMES / 8))):
        super().__init__()
        c, h, w = conv_shape
        self.conv_shape = conv_shape
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, c * h * w),
            nn.BatchNorm1d(c * h * w),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c, 64, kernel_size=4, stride=2, padding=1),   # H/4 x W/4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # H/2 x W/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # H x W
            nn.Sigmoid(),  # spectrograms are min-max normalized to [0, 1]
        )

    def forward(self, z):
        c, h, w = self.conv_shape
        x = self.fc(z).view(-1, c, h, w)
        x = self.deconv(x)
        # guard against rounding mismatches
        if x.shape[-2:] != (N_MELS, N_FRAMES):
            x = nn.functional.interpolate(x, size=(N_MELS, N_FRAMES), mode="bilinear", align_corners=False)
        return x


class Discriminator(nn.Module):
    """(1, N_MELS, N_FRAMES) -> single real/fake LOGIT (no Sigmoid; BCEWithLogits is used)."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, 1, N_MELS, N_FRAMES)).view(1, -1).shape[1]
        self.fc = spectral_norm(nn.Linear(flat_size, 1))

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h).squeeze(1)  # (B,) logits


class GAN(nn.Module):
    def __init__(self, latent_dim=128, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator(dropout)

    @torch.no_grad()
    def sample(self, n=1, device=DEVICE):
        """Generate new spectrograms from random latent noise."""
        was_training = self.generator.training
        self.generator.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        out = self.generator(z)
        self.generator.train(was_training)
        return out


def init_weights(m):
    """DCGAN-style init (spectral-norm layers are wrapped, so check the base attr)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        weight = m.weight_orig if hasattr(m, "weight_orig") else m.weight
        nn.init.normal_(weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------
# FID (Frechet Distance) on Autoencoder-encoder features
# --------------------------------------------------------------------------
def load_feature_extractor(latent_dim_ae=256, dropout_ae=0.1):
    """
    Use the trained Autoencoder's encoder as a domain-specific feature extractor
    (checkpoints/autoencoder.pt). Falls back to a FIXED random-init encoder if
    the checkpoint is missing -- still usable for tracking relative progress
    within a single run, but NOT comparable with runs using a trained encoder.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, "autoencoder.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        ae = Autoencoder(latent_dim=ckpt["latent_dim"], dropout=ckpt.get("dropout", dropout_ae))
        ae.load_state_dict(ckpt["model_state"])
        print(f"[GAN] FID feature extractor: trained AE encoder ({ckpt_path})")
    else:
        print("[WARN] checkpoints/autoencoder.pt not found -- using a fixed random encoder for FID. "
              "Train autoencoder.py first for meaningful, comparable FID values.")
        torch.manual_seed(0)
        ae = Autoencoder(latent_dim=latent_dim_ae, dropout=dropout_ae)
    encoder = ae.encoder.to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def extract_features(encoder, data: torch.Tensor, batch_size=64) -> np.ndarray:
    feats = []
    for i in range(0, data.shape[0], batch_size):
        x = data[i:i + batch_size].to(DEVICE)
        z, _ = encoder(x)
        feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(feat_real: np.ndarray, feat_fake: np.ndarray, eps=1e-6) -> float:
    """d^2 = ||mu1-mu2||^2 + Tr(S1 + S2 - 2*sqrt(S1*S2))"""
    from scipy import linalg

    mu1, mu2 = feat_real.mean(axis=0), feat_fake.mean(axis=0)
    s1 = np.cov(feat_real, rowvar=False)
    s2 = np.cov(feat_fake, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def mean_pairwise_l1(x: torch.Tensor, max_n=200) -> float:
    """Average L1 distance between random pairs -- low value => mode collapse."""
    x = x[:max_n].flatten(1)
    perm = torch.randperm(x.size(0))
    return (x - x[perm]).abs().mean().item()


@torch.no_grad()
def evaluate_gan(model, real_data: torch.Tensor, encoder, batch_size=64):
    """
    real_data : tensor (N, 1, N_MELS, N_FRAMES) of held-out REAL spectrograms.
    Generates the same number of fakes and reports FID, discriminator
    accuracy on real/fake, and a diversity score.
    """
    model.eval()
    n = real_data.shape[0]

    fake_data = []
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        z = torch.randn(b, model.latent_dim, device=DEVICE)
        fake_data.append(model.generator(z).cpu())
    fake_data = torch.cat(fake_data)

    real_correct, fake_correct = 0, 0
    for i in range(0, n, batch_size):
        real_correct += (model.discriminator(real_data[i:i + batch_size].to(DEVICE)) > 0).sum().item()
        fake_correct += (model.discriminator(fake_data[i:i + batch_size].to(DEVICE)) <= 0).sum().item()

    fid = frechet_distance(extract_features(encoder, real_data), extract_features(encoder, fake_data))

    return {
        "FID": fid,
        "D_Real_Acc": real_correct / n,
        "D_Fake_Acc": fake_correct / n,
        "Diversity_Fake": mean_pairwise_l1(fake_data),
        "Diversity_Real": mean_pairwise_l1(real_data),
        "Fake_Mean": fake_data.mean().item(),
        "Real_Mean": real_data.mean().item(),
    }


def save_sample_grid(model, real_data, path, n=8):
    """Save a real-vs-generated spectrogram comparison image (needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping sample grid.")
        return

    fake = model.sample(n).cpu()
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(real_data[i, 0].numpy(), origin="lower", aspect="auto", cmap="magma")
        axes[1, i].imshow(fake[i, 0].numpy(), origin="lower", aspect="auto", cmap="magma")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Real", loc="left")
    axes[1, 0].set_title("Generated", loc="left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved sample grid to {path}")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_gan(
    epochs=100,
    batch_size=32,
    latent_dim=128,
    lr=2e-4,
    beta1=0.5,
    dropout=0.1,
    real_label=0.9,        # one-sided label smoothing
    grad_clip=1.0,
    n_critic=1,            # D updates per G update
    eval_every=2,
    patience=10,           # counted in EVALUATIONS (eval_every epochs each)
    min_delta=1e-3,
):
    """
    Early stopping / best-checkpoint selection uses validation FID (lower is
    better). GAN training losses oscillate by design and do not track sample
    quality, so loss-based checkpointing would pick arbitrary epochs.
    """
    X_train, X_val, X_test = load_splits()

    train_loader = DataLoader(MelSpectrogramDataset(X_train), batch_size=batch_size,
                              shuffle=True, drop_last=True)  # drop_last: BatchNorm needs B > 1

    model = GAN(latent_dim=latent_dim, dropout=dropout).to(DEVICE)
    model.generator.apply(init_weights)
    model.discriminator.apply(init_weights)

    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()

    encoder = load_feature_extractor()

    history = {"d_loss": [], "g_loss": [], "val_fid": [], "val_fid_epoch": [],
               "d_real_acc": [], "d_fake_acc": []}
    best_fid = float("inf")
    best_state = None
    evals_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        d_run, g_run, n_seen = 0.0, 0.0, 0

        for real in train_loader:
            real = real.to(DEVICE)
            b = real.size(0)

            # ---- Discriminator step(s) ----
            for _ in range(n_critic):
                z = torch.randn(b, latent_dim, device=DEVICE)
                fake = model.generator(z).detach()
                opt_d.zero_grad()
                d_real = model.discriminator(real)
                d_fake = model.discriminator(fake)
                d_loss = (bce(d_real, torch.full_like(d_real, real_label)) +
                          bce(d_fake, torch.zeros_like(d_fake))) * 0.5
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), grad_clip)
                opt_d.step()

            # ---- Generator step (non-saturating loss: want D(G(z)) -> "real") ----
            z = torch.randn(b, latent_dim, device=DEVICE)
            opt_g.zero_grad()
            g_out = model.discriminator(model.generator(z))
            g_loss = bce(g_out, torch.ones_like(g_out))
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), grad_clip)
            opt_g.step()

            d_run += d_loss.item() * b
            g_run += g_loss.item() * b
            n_seen += b

        d_loss_epoch, g_loss_epoch = d_run / n_seen, g_run / n_seen
        history["d_loss"].append(d_loss_epoch)
        history["g_loss"].append(g_loss_epoch)

        msg = f"[GAN] Epoch {epoch:03d}/{epochs} | D_loss={d_loss_epoch:.4f} | G_loss={g_loss_epoch:.4f}"

        if epoch % eval_every == 0 or epoch == epochs:
            m = evaluate_gan(model, X_val, encoder)
            history["val_fid"].append(m["FID"])
            history["val_fid_epoch"].append(epoch)
            history["d_real_acc"].append(m["D_Real_Acc"])
            history["d_fake_acc"].append(m["D_Fake_Acc"])

            improved = m["FID"] < (best_fid - min_delta)
            if improved:
                best_fid = m["FID"]
                best_state = copy.deepcopy(model.state_dict())
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            flag = " *" if improved else ""
            msg += (f" | val_FID={m['FID']:.3f}{flag} | D_acc(real/fake)="
                    f"{m['D_Real_Acc']:.2f}/{m['D_Fake_Acc']:.2f} | "
                    f"div(fake/real)={m['Diversity_Fake']:.3f}/{m['Diversity_Real']:.3f}")
            print(msg)

            if evals_without_improvement >= patience:
                print(f"[GAN] Early stopping at epoch {epoch} "
                      f"(no FID improvement for {patience} evaluations).")
                break
        else:
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (val_FID={best_fid:.3f}).")

    print("\nFinal Test Evaluation (best checkpoint):")
    test_metrics = evaluate_gan(model, X_test, encoder)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.5f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "gan.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "generator_state": model.generator.state_dict(),
            "latent_dim": latent_dim,
            "dropout": dropout,
            "best_val_fid": best_fid,
        },
        ckpt_path,
    )
    print(f"Saved GAN checkpoint to {ckpt_path}")

    save_sample_grid(model, X_test, os.path.join(OUTPUT_DIR, "gan_samples.png"))

    return model, history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GAN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--real-label", type=float, default=0.9, help="One-sided label smoothing value")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-critic", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10, help="in evaluations, not epochs")
    args = parser.parse_args()

    train_gan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        beta1=args.beta1,
        dropout=args.dropout,
        real_label=args.real_label,
        grad_clip=args.grad_clip,
        n_critic=args.n_critic,
        eval_every=args.eval_every,
        patience=args.patience,
    )
