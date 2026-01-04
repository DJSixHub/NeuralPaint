"""
Clean training script for U-Net segmentation with anti-aliasing.
Uses binary .pt files for fast training.
Optimized with best hyperparameters from HPO.

Pure perceptual optimization: -MAE + β*SSIM
No binary thresholding - continuous quality metrics only.
"""
from pathlib import Path
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Performance
torch.backends.cudnn.benchmark = True

# Best hyperparameters from HPO (perceptual optimization)
BEST_BASE = 32
BEST_LR = 0.0004974340400907659
BEST_WD = 8.412043612446531e-07
BEST_BATCH = 32
BEST_BCE_WEIGHT = 0.4
BEST_EDGE_WEIGHT = 0.15
BEST_SSIM_WEIGHT = 0.45
# Edge scale sensitivity
BEST_EDGE_DOWNSAMPLE = 2
BEST_EDGE_COARSE_WEIGHT = 0.3
# Perceptual scale
BEST_SSIM_WINDOW_SIZE = 11
# Output activation
BEST_TEMPERATURE = 1.0
# Optimizer
BEST_OPTIMIZER = 'AdamW'

# Training defaults
DEFAULT_EPOCHS = 50
DEFAULT_VAL_FRAC = 0.2
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 0.0005
LR_PLATEAU_PATIENCE = 3
LR_PLATEAU_FACTOR = 0.5


# ============================================================================
# LOSS & METRIC FUNCTIONS
# ============================================================================

def bce_soft_loss_from_logits(logits: torch.Tensor, targets_soft: torch.Tensor) -> torch.Tensor:
    """BCE loss on soft/anti-aliased targets."""
    return F.binary_cross_entropy_with_logits(logits, targets_soft)


def soft_mae_from_logits(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """MAE on soft predictions. CRITICAL for visual quality.
    
    Measures anti-aliasing and halo blur fidelity on edges/boundaries.
    Lower = better edge smoothness and visual quality.
    
    Args:
        temperature: Output sharpness control. Affects edge softness.
    """
    probs = torch.sigmoid(logits / temperature)
    return (probs - targets_soft).abs().mean()


def ssim_loss(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    window_size: int = 11,
    temperature: float = 1.0,
) -> torch.Tensor:
    """SSIM loss for perceptual quality.
    
    Captures structural similarity better than pixel-wise losses.
    Critical for anti-aliasing and visual fidelity.
    
    Args:
        window_size: Gaussian window size (7, 11, or 15). Larger = coarser perception.
        temperature: Output sharpness control. >1 = softer edges, <1 = sharper.
    """
    probs = torch.sigmoid(logits / temperature)
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window (sigma scales with window size)
    sigma = window_size / 7.0
    gauss = torch.exp(-torch.arange(window_size, dtype=probs.dtype, device=probs.device).sub(window_size // 2).pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)
    
    # Compute local means
    mu1 = F.conv2d(probs, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(targets_soft, window, padding=window_size // 2, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(probs * probs, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(targets_soft * targets_soft, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(probs * targets_soft, window, padding=window_size // 2, groups=1) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1.0 - ssim_map.mean()  # Convert to loss (lower is better)


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype)
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def edge_loss_from_logits(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    weight: float = 1.0,
    downsample_factor: int = 2,
    coarse_weight: float = 0.3,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Multi-scale edge loss with magnitude + direction.
    
    Combines:
    - Gradient magnitude (edge strength)
    - Gradient direction (edge orientation)
    - Multi-scale (coarse + fine edges)
    
    Args:
        downsample_factor: Scale reduction for coarse edges (1, 2, or 4)
        coarse_weight: Relative importance of coarse vs fine edges
        temperature: Output sharpness control
    """
    if weight <= 0:
        return logits.new_tensor(0.0)

    probs = torch.sigmoid(logits / temperature)
    kx, ky = _sobel_kernels(logits.device, logits.dtype)

    # Original scale
    gx_p = F.conv2d(probs, kx, padding=1)
    gy_p = F.conv2d(probs, ky, padding=1)
    g_mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-8)

    gx_t = F.conv2d(targets_soft, kx, padding=1)
    gy_t = F.conv2d(targets_soft, ky, padding=1)
    g_mag_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-8)

    # Magnitude loss
    mag_loss = (g_mag_p - g_mag_t).abs().mean()
    
    # Direction loss (cosine similarity of gradient vectors)
    # Normalize gradients to unit vectors
    gx_p_norm = gx_p / (g_mag_p + 1e-8)
    gy_p_norm = gy_p / (g_mag_p + 1e-8)
    gx_t_norm = gx_t / (g_mag_t + 1e-8)
    gy_t_norm = gy_t / (g_mag_t + 1e-8)
    
    # Cosine similarity (dot product of normalized gradients)
    cos_sim = (gx_p_norm * gx_t_norm + gy_p_norm * gy_t_norm).clamp(-1, 1)
    dir_loss = (1.0 - cos_sim).mean()  # 0 = same direction, 2 = opposite
    
    # Multi-scale: add coarser scale via pooling
    if downsample_factor > 1:
        probs_down = F.avg_pool2d(probs, downsample_factor, downsample_factor)
        targets_down = F.avg_pool2d(targets_soft, downsample_factor, downsample_factor)
        
        gx_p_down = F.conv2d(probs_down, kx, padding=1)
        gy_p_down = F.conv2d(probs_down, ky, padding=1)
        g_mag_p_down = torch.sqrt(gx_p_down * gx_p_down + gy_p_down * gy_p_down + 1e-8)
        
        gx_t_down = F.conv2d(targets_down, kx, padding=1)
        gy_t_down = F.conv2d(targets_down, ky, padding=1)
        g_mag_t_down = torch.sqrt(gx_t_down * gx_t_down + gy_t_down * gy_t_down + 1e-8)
        
        mag_loss_down = (g_mag_p_down - g_mag_t_down).abs().mean()
    else:
        mag_loss_down = 0.0
    
    # Combine components (coarse_weight is now tunable)
    total_loss = mag_loss + 0.5 * dir_loss + coarse_weight * mag_loss_down
    
    return weight * total_loss


@dataclass
class LossConfig:
    bce_weight: float = 0.4
    edge_weight: float = 0.15
    ssim_weight: float = 0.45
    # Edge scale sensitivity
    edge_downsample: int = 2
    edge_coarse_weight: float = 0.3
    # Perceptual scale
    ssim_window_size: int = 11
    # Output activation shaping
    temperature: float = 1.0


def sota_loss(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    cfg: LossConfig,
) -> torch.Tensor:
    """Pure perceptual loss: BCE + Edge + SSIM.
    
    No binary operations - continuous optimization only.
    Balances:
    - Pixel-wise accuracy (BCE soft)
    - Edge fidelity (multi-scale magnitude + direction + tunable scales)
    - Perceptual quality (SSIM structural similarity with tunable window)
    - Output softness (temperature-controlled activation)
    """
    bce = bce_soft_loss_from_logits(logits, targets_soft)
    edge = edge_loss_from_logits(
        logits, targets_soft,
        weight=cfg.edge_weight,
        downsample_factor=cfg.edge_downsample,
        coarse_weight=cfg.edge_coarse_weight,
        temperature=cfg.temperature
    )
    ssim = ssim_loss(logits, targets_soft, window_size=cfg.ssim_window_size, temperature=cfg.temperature)
    
    return cfg.bce_weight * bce + edge + cfg.ssim_weight * ssim


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, use_se: bool = True):
        super().__init__()
        g1 = min(groups, out_ch)
        g1 = max(1, g1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(g1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(g1, out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        y = self.se(y)
        return F.silu(y + self.skip(x))


class ResUNet(nn.Module):
    """Modernized U-Net: residual blocks + GroupNorm + SiLU + optional SE."""

    def __init__(self, in_ch: int = 3, base: int = 32, use_se: bool = True):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base, use_se=use_se)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(base, base * 2, use_se=use_se)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock(base * 2, base * 4, use_se=use_se)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock(base * 4, base * 2, use_se=use_se)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ResBlock(base * 2, base, use_se=use_se)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        u2 = self.up2(e3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.outc(d1)


# ============================================================================
# DATASET & TRAINING
# ============================================================================

LOSS_CFG = LossConfig(
    bce_weight=BEST_BCE_WEIGHT,
    edge_weight=BEST_EDGE_WEIGHT,
    ssim_weight=BEST_SSIM_WEIGHT,
    edge_downsample=BEST_EDGE_DOWNSAMPLE,
    edge_coarse_weight=BEST_EDGE_COARSE_WEIGHT,
    ssim_window_size=BEST_SSIM_WINDOW_SIZE,
    temperature=BEST_TEMPERATURE
)


class FastBinaryDataset(Dataset):
    """Fast dataset using preprocessed .pt binary files.
    
    Images are pre-normalized to 256×256 during preprocessing.
    """
    
    def __init__(self, root: Path, files: list[str]):
        self.root = Path(root)
        self.binary_dir = self.root / 'binary'
        self.files = files
        
        if not self.binary_dir.exists():
            raise FileNotFoundError(f"Binary directory not found: {self.binary_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int):
        fname = self.files[idx]
        p_bin = self.binary_dir / Path(fname).with_suffix('.pt')
        
        if not p_bin.exists():
            raise FileNotFoundError(f"Binary file not found: {p_bin}")
        
        # Fast load from preprocessed binary
        data = torch.load(p_bin, map_location='cpu', weights_only=True)
        
        # Normalize to [0,1]
        img = data['img'].float() / 255.0
        mask = data['mask'].float() / 255.0
        
        return {
            'image': img,
            'mask': mask
        }


def build_file_list(root: Path):
    """Build file list from binary cache directory."""
    binary_dir = root / 'binary'
    if not binary_dir.exists():
        return []
    
    files = []
    for p in sorted(binary_dir.rglob('*.pt')):
        rel = p.relative_to(binary_dir).as_posix()
        rel = rel[:-3] if rel.endswith('.pt') else rel  # Remove .pt extension
        files.append(rel)
    return files


def train_and_evaluate(
    dataset_root: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BEST_BATCH,
    val_frac: float = DEFAULT_VAL_FRAC,
    num_workers: int = 4,
    resume: str = None,
    eval_only: bool = False,
    base: int = BEST_BASE,
    lr: float = BEST_LR,
    weight_decay: float = BEST_WD,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(dataset_root)
    
    # Build file list from binary cache
    files = build_file_list(root)
    if not files:
        raise ValueError(f"No binary files found in {root / 'binary'}. Generate dataset first.")
    
    print(f"Found {len(files)} binary files")
    
    # Train/val split
    random.seed(42)
    random.shuffle(files)
    n_val = int(len(files) * val_frac)
    train_files = files[n_val:]
    val_files = files[:n_val]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Datasets
    train_ds = FastBinaryDataset(root, train_files)
    val_ds = FastBinaryDataset(root, val_files)
    
    # DataLoaders - optimized for fast training
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    # Model with channels-last memory format for faster convolutions
    model = ResUNet(in_ch=3, base=base, use_se=True).to(device)
    model = model.to(memory_format=torch.channels_last)

    # Optimizer (AdamW preferred for better weight decay handling)
    opt_class = torch.optim.AdamW if BEST_OPTIMIZER == 'AdamW' else torch.optim.Adam
    try:
        opt = opt_class(model.parameters(), lr=lr, weight_decay=weight_decay, fused=(device.type == 'cuda'))
    except TypeError:
        opt = opt_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = ReduceLROnPlateau(
        opt, mode='min', factor=LR_PLATEAU_FACTOR, 
        patience=LR_PLATEAU_PATIENCE, min_lr=1e-9
    )
    # Using Combined BCE+Dice loss (no separate loss_fn needed)
    
    # Mixed precision - handle multiple PyTorch versions
    if device.type == 'cuda':
        try:
            # PyTorch 2.0+ API
            scaler = torch.amp.GradScaler(device_type='cuda')
        except TypeError:
            scaler = torch.amp.GradScaler('cuda')
        autocast_fn = lambda: torch.amp.autocast('cuda')
    else:
        scaler = None
        from contextlib import nullcontext
        autocast_fn = lambda: nullcontext()
    
    # Output directory
    out_dir = Path(__file__).parent.parent.parent / 'models' / 'segmentation'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_iou = -1.0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck.get('model_state', ck))
        try:
            opt.load_state_dict(ck.get('opt_state', {}))
        except:
            pass
        start_epoch = ck.get('epoch', 0) + 1
        best_val_loss = ck.get('best_val_loss', float('inf'))
        best_val_iou = ck.get('best_val_iou', -1.0)
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    if eval_only:
        model.eval()
        sum_val_loss = 0.0
        sum_val_mae = 0.0
        sum_ssim = 0.0
        cnt_val = 0
        with torch.no_grad():
            for vb in tqdm(val_loader, desc='Eval', leave=False):
                imgs = vb['image'].to(device, non_blocking=True)
                imgs = imgs.to(memory_format=torch.channels_last)
                masks = vb['mask'].to(device, non_blocking=True)

                with autocast_fn():
                    out = model(imgs)

                loss_val = sota_loss(out, masks, LOSS_CFG)
                sum_val_loss += loss_val.item() * imgs.size(0)
                sum_val_mae += soft_mae_from_logits(out, masks, temperature=LOSS_CFG.temperature).item() * imgs.size(0)
                sum_ssim += (1.0 - ssim_loss(out, masks, window_size=LOSS_CFG.ssim_window_size, temperature=LOSS_CFG.temperature).item()) * imgs.size(0)
                cnt_val += imgs.size(0)

        epoch_val_loss = sum_val_loss / max(1, cnt_val)
        epoch_val_mae = sum_val_mae / max(1, cnt_val)
        epoch_ssim = sum_ssim / max(1, cnt_val)
        composite = -epoch_val_mae + 0.5 * epoch_ssim
        print(
            f"Eval: loss={epoch_val_loss:.6f}, mae={epoch_val_mae:.4f}, "
            f"ssim={epoch_ssim:.4f}, composite={composite:.4f}"
        )
        return
    
    # Training loop
    train_losses = []
    val_losses = []
    val_maes = []
    val_ssims = []
    epochs_no_improve = 0
    best_composite_score = -float('inf')  # -MAE + β*SSIM
    
    print(f"\nTraining with: base={base}, lr={lr:.2e}, wd={weight_decay:.2e}, batch={batch_size}")
    print(f"Device: {device}, Mixed precision: {scaler is not None}")
    print(f"Goal: Maximize -MAE + β*SSIM (pure perceptual quality)\n")
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in pbar:
            imgs = batch['image'].to(device, non_blocking=True)
            imgs = imgs.to(memory_format=torch.channels_last)
            masks = batch['mask'].to(device, non_blocking=True)
            
            opt.zero_grad()
            
            with autocast_fn():
                preds = model(imgs)
                loss = sota_loss(preds, masks, LOSS_CFG)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_train_loss = running_loss / len(train_ds)
        train_losses.append(epoch_train_loss)
        
        # Validation (every 2 epochs)
        if (epoch + 1) % 2 == 0:
            model.eval()
            sum_val_loss = 0.0
            sum_val_mae = 0.0
            sum_ssim = 0.0
            cnt_val = 0
            
            with torch.no_grad():
                for vb in tqdm(val_loader, desc='Validation', leave=False):
                    imgs = vb['image'].to(device, non_blocking=True)
                    imgs = imgs.to(memory_format=torch.channels_last)
                    masks = vb['mask'].to(device, non_blocking=True)
                    
                    with autocast_fn():
                        out = model(imgs)
                    
                    # Loss
                    loss_val = sota_loss(out, masks, LOSS_CFG)
                    sum_val_loss += loss_val.item() * imgs.size(0)
                    
                    # Perceptual metrics
                    sum_val_mae += soft_mae_from_logits(out, masks, temperature=LOSS_CFG.temperature).item() * imgs.size(0)
                    sum_ssim += (1.0 - ssim_loss(out, masks, window_size=LOSS_CFG.ssim_window_size, temperature=LOSS_CFG.temperature).item()) * imgs.size(0)
                    
                    cnt_val += imgs.size(0)
            
            epoch_val_loss = sum_val_loss / cnt_val
            epoch_val_mae = sum_val_mae / cnt_val
            epoch_ssim = sum_ssim / cnt_val
            val_losses.append(epoch_val_loss)
            val_maes.append(epoch_val_mae)
            val_ssims.append(epoch_ssim)
            
            scheduler.step(epoch_val_loss)
            
            print(
                f'Epoch {epoch+1}: '
                f'train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f} | '
                f'mae={epoch_val_mae:.4f}, ssim={epoch_ssim:.4f}'
            )
            
            # Pure perceptual composite: -MAE + β*SSIM
            beta = 0.5
            composite_score = -epoch_val_mae + beta * epoch_ssim
            
            if composite_score > best_composite_score:
                epochs_no_improve = 0
                best_composite_score = composite_score
                best_val_loss = min(best_val_loss, epoch_val_loss)
                torch.save(model.state_dict(), out_dir / 'best_model.pth')
                print(f'  → New best model saved (composite={composite_score:.4f}, mae={epoch_val_mae:.4f}, ssim={epoch_ssim:.4f})')
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f'\nEarly stopping: no improvement for {EARLY_STOPPING_PATIENCE} validations')
                break
            
            # Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'scaler_state': scaler.state_dict() if scaler else {},
                'best_val_loss': best_val_loss,
                'best_val_iou': best_val_iou,
            }, out_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), out_dir / 'final_model.pth')
    
    # Plot training curves with improved visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot - both train and val on same axes
    ax1 = axes[0]
    epochs_train = list(range(1, len(train_losses) + 1))
    epochs_val = [i * 2 for i in range(1, len(val_losses) + 1)]
    
    ax1.plot(epochs_train, train_losses, label='Train Loss', color='#2E86AB', linewidth=2)
    ax1.plot(epochs_val, val_losses, label='Val Loss', color='#E63946', linewidth=2, marker='o', markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, len(train_losses) + 1)
    
    # SSIM plot (perceptual quality)
    ax2 = axes[1]
    ax2.plot(epochs_val, val_ssims, label='Val SSIM', color='#06A77D', linewidth=2, marker='s', markersize=5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax2.set_title('Validation SSIM (Perceptual Quality)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, len(train_losses) + 1)
    ax2.set_ylim(0, 1.0)
    
    # Add best performance annotations
    if val_ssims:
        best_ssim = max(val_ssims)
        best_ssim_epoch = epochs_val[val_ssims.index(best_ssim)]
        ax2.axhline(y=best_ssim, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax2.text(1, best_ssim + 0.02, f'Best: {best_ssim:.4f}', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Create a detailed summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Combined loss view with dual y-axis
    ax_twin = ax.twinx()
    
    line1 = ax.plot(epochs_train, train_losses, label='Train Loss', color='#2E86AB', linewidth=2, alpha=0.8)
    line2 = ax.plot(epochs_val, val_losses, label='Val Loss', color='#E63946', linewidth=2, marker='o', markersize=4)
    line3 = ax_twin.plot(epochs_val, val_ssims, label='Val SSIM', color='#06A77D', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#2E86AB')
    ax_twin.set_ylabel('SSIM', fontsize=12, fontweight='bold', color='#06A77D')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax_twin.tick_params(axis='y', labelcolor='#06A77D')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=10)
    
    best_ssim = max(val_ssims) if val_ssims else 0
    best_mae = min(val_maes) if val_maes else 1
    best_composite = -best_mae + 0.5 * best_ssim
    ax.set_title(f'Training Summary - Best SSIM: {best_ssim:.4f}, MAE: {best_mae:.4f}, Composite: {best_composite:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'training_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f'\nTraining complete!')
    print(f'Best val loss: {best_val_loss:.6f}')
    if val_ssims:
        print(f'Best SSIM (perceptual quality): {max(val_ssims):.4f}')
    if val_maes:
        print(f'Best MAE (AA fidelity): {min(val_maes):.4f}')
        print(f'Best composite score: {best_composite_score:.4f}')
    print(f'Models saved in: {out_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train U-Net for character segmentation')
    parser.add_argument('--data-root', type=str, 
                        default=str(Path(__file__).parent.parent.parent / 'datasets' / 'synthetic'),
                        help='Dataset root directory')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BEST_BATCH,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on val set (requires --resume or uses current weights).')
    
    args = parser.parse_args()
    
    train_and_evaluate(
        dataset_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=args.resume,
        eval_only=args.eval_only,
    )
