"""HPO for character segmentation with anti-aliasing.

Optimizes perceptual quality: SSIM + MAE + Edge fidelity.
No binary thresholding - pure continuous optimization.
"""
from pathlib import Path
import random
from typing import List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from tqdm import tqdm
import argparse

# HPO defaults
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True  # Optimize kernels for fixed input size
torch.backends.cudnn.deterministic = False  # Disable for speed
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for Ampere+ GPUs
torch.backends.cudnn.allow_tf32 = True

DATASET_ROOT = Path(__file__).parent.parent.parent / 'datasets' / 'synthetic'
HPO_SUBSET_FRAC = 0.1  # 5% subset for fast trials
HPO_NUM_WORKERS = 12  # Increased for faster loading
HPO_EPOCHS = 8
GRADIENT_ACCUMULATION_STEPS = 2  # Reduce optimizer overhead


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


# Cache Sobel kernels globally for speed
_SOBEL_CACHE = {}

def _get_cached_sobel(device: torch.device, dtype: torch.dtype) -> tuple:
    key = (device, dtype)
    if key not in _SOBEL_CACHE:
        _SOBEL_CACHE[key] = _sobel_kernels(device, dtype)
    return _SOBEL_CACHE[key]

def edge_loss_from_logits(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    weight: float = 1.0,
    downsample_factor: int = 2,
    coarse_weight: float = 0.3,
    temperature: float = 1.0,
    use_direction: bool = False,
    use_multiscale: bool = False,
) -> torch.Tensor:
    """Multi-scale edge loss with magnitude + direction.
    
    Combines:
    - Gradient magnitude (edge strength)
    - Gradient direction (edge orientation) - OPTIONAL
    - Multi-scale (coarse + fine edges) - OPTIONAL
    
    Args:
        downsample_factor: Scale reduction for coarse edges (1, 2, or 4)
        coarse_weight: Relative importance of coarse vs fine edges
        temperature: Output sharpness control
        use_direction: Enable direction loss (disable for HPO stability)
        use_multiscale: Enable multi-scale (disable for HPO stability)
    """
    if weight <= 0:
        return logits.new_tensor(0.0)

    probs = torch.sigmoid(logits / temperature)
    kx, ky = _get_cached_sobel(logits.device, logits.dtype)

    # Original scale - compute both in one go for better memory locality
    gx_p = F.conv2d(probs, kx, padding=1)
    gy_p = F.conv2d(probs, ky, padding=1)
    gx_t = F.conv2d(targets_soft, kx, padding=1)
    gy_t = F.conv2d(targets_soft, ky, padding=1)
    
    # Magnitude - fused computation
    g_mag_p = (gx_p.square() + gy_p.square() + 1e-8).sqrt().clamp(min=1e-6)
    g_mag_t = (gx_t.square() + gy_t.square() + 1e-8).sqrt().clamp(min=1e-6)

    # Magnitude loss (always enabled)
    mag_loss = (g_mag_p - g_mag_t).abs().mean()
    
    total_loss = mag_loss
    
    # Direction loss (optional - disable for HPO)
    if use_direction:
        # Normalize gradients to unit vectors
        gx_p_norm = gx_p / (g_mag_p + 1e-8)
        gy_p_norm = gy_p / (g_mag_p + 1e-8)
        gx_t_norm = gx_t / (g_mag_t + 1e-8)
        gy_t_norm = gy_t / (g_mag_t + 1e-8)
        
        # Cosine similarity (dot product of normalized gradients)
        cos_sim = (gx_p_norm * gx_t_norm + gy_p_norm * gy_t_norm).clamp(-1, 1)
        dir_loss = (1.0 - cos_sim).mean()
        total_loss = total_loss + 0.5 * dir_loss
    
    # Multi-scale (optional - disable for HPO)
    if use_multiscale and downsample_factor > 1:
        probs_down = F.avg_pool2d(probs, downsample_factor, downsample_factor)
        targets_down = F.avg_pool2d(targets_soft, downsample_factor, downsample_factor)
        
        gx_p_down = F.conv2d(probs_down, kx, padding=1)
        gy_p_down = F.conv2d(probs_down, ky, padding=1)
        g_mag_p_down = torch.sqrt(gx_p_down * gx_p_down + gy_p_down * gy_p_down + 1e-8)
        g_mag_p_down = torch.clamp(g_mag_p_down, min=1e-6)
        
        gx_t_down = F.conv2d(targets_down, kx, padding=1)
        gy_t_down = F.conv2d(targets_down, ky, padding=1)
        g_mag_t_down = torch.sqrt(gx_t_down * gx_t_down + gy_t_down * gy_t_down + 1e-8)
        g_mag_t_down = torch.clamp(g_mag_t_down, min=1e-6)
        
        mag_loss_down = (g_mag_p_down - g_mag_t_down).abs().mean()
        total_loss = total_loss + coarse_weight * mag_loss_down
    
    return weight * total_loss


@dataclass
class LossConfig:
    bce_weight: float = 0.4
    edge_weight: float = 0.15
    ssim_weight: float = 0.45
    # Edge scale sensitivity
    edge_downsample: int = 2  # 1, 2, or 4
    edge_coarse_weight: float = 0.3  # Coarse vs fine balance
    # Perceptual scale
    ssim_window_size: int = 11  # 7, 11, or 15
    # Output activation shaping
    temperature: float = 1.0  # Edge softness control
    # HPO stability flags
    use_ssim: bool = False  # Disable SSIM during HPO (NaN-prone)
    use_direction_loss: bool = False  # Disable direction loss during HPO
    use_multiscale: bool = False  # Disable multi-scale during HPO


def sota_loss(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    cfg: LossConfig,
) -> torch.Tensor:
    """Pure perceptual loss: BCE + Edge + SSIM (optional).
    
    HPO mode: Simplified for stability (magnitude-only edge, no SSIM)
    Full mode: Complete perceptual optimization
    """
    bce = bce_soft_loss_from_logits(logits, targets_soft)
    edge = edge_loss_from_logits(
        logits, targets_soft,
        weight=cfg.edge_weight,
        downsample_factor=cfg.edge_downsample,
        coarse_weight=cfg.edge_coarse_weight,
        temperature=cfg.temperature,
        use_direction=cfg.use_direction_loss,
        use_multiscale=cfg.use_multiscale
    )
    
    if cfg.use_ssim:
        ssim = ssim_loss(logits, targets_soft, window_size=cfg.ssim_window_size, temperature=cfg.temperature)
    else:
        ssim = 0.0
    
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
# DATASET
# ============================================================================

# ============================================================================
# DATASET
# ============================================================================

class FastBinaryDataset(Dataset):
    """Optimized dataset using preprocessed .pt binary files.
    
    Each .pt file contains {'img': tensor(3,H,W), 'mask': tensor(1,H,W)}
    Images are pre-normalized to 256×256 during preprocessing.
    """
    def __init__(self, root: Path, files: List[str]):
        self.root = Path(root)
        self.binary_dir = self.root / 'binary'
        self.files = list(files)
        
        if not self.binary_dir.exists():
            raise FileNotFoundError(
                f"Binary cache directory not found: {self.binary_dir}. "
                f"Run preprocess_binary.py first to generate .pt files."
            )

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
    """Build file list from binary cache directory.
    Returns list of relative paths (without .pt extension) for dataset loading.
    """
    binary_dir = root / 'binary'
    if not binary_dir.exists():
        return []
    
    files = []
    for p in sorted(binary_dir.rglob('*.pt')):
        try:
            # Get relative path and remove .pt extension to match image naming
            rel = p.relative_to(binary_dir).as_posix()
            rel = rel[:-3] if rel.endswith('.pt') else rel  # Remove .pt
        except Exception:
            rel = p.stem  # Use stem (filename without extension)
        files.append(rel)
    return files


# Genera o carga listas fijas de archivos para HPO (reproducible).
HPO_SEED = 1234
def ensure_hpo_filelists(root: Path, train_fname: str = 'hpo_train_files.txt', val_fname: str = 'hpo_val_files.txt', val_frac: float = 0.2):
    train_path = root / train_fname
    val_path = root / val_fname
    # If files already exist, read them. If they're empty, fall back to rebuilding from available images.
    if train_path.exists() and val_path.exists():
        train_files = [l.strip() for l in train_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        val_files = [l.strip() for l in val_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        if train_files and val_files:
            print(f"Loaded existing HPO lists: train={len(train_files)} val={len(val_files)}")
            return train_files, val_files
        else:
            print("Existing HPO list files found but empty — regenerating from images.")

    files = build_file_list(root)
    if not files:
        return [], []
    rnd = random.Random(HPO_SEED)
    rnd.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    val_files = files[:n_val]
    train_files = files[n_val:]
    train_path.write_text('\n'.join(train_files), encoding='utf-8')
    val_path.write_text('\n'.join(val_files), encoding='utf-8')
    print(f"Generated HPO lists: train={len(train_files)} val={len(val_files)} (from {len(files)} images)")
    return train_files, val_files


def objective(trial: optuna.trial.Trial) -> float:
    """Objective function for HPO. Maximizes perceptual quality: -MAE + β*SSIM."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Suggested hyperparameters (simplified for HPO stability)
    base = trial.suggest_categorical('base', [24, 32])
    lr = trial.suggest_float('lr', 3e-4, 1e-3, log=True)
    wd = trial.suggest_float('wd', 1e-8, 1e-6, log=True)
    batch = trial.suggest_categorical('batch', [16, 32])
    
    # Loss component weights (simplified - no SSIM during HPO)
    bce_weight = trial.suggest_float('bce_weight', 0.5, 0.7)
    edge_weight = trial.suggest_float('edge_weight', 0.3, 0.5)
    
    # Output activation temperature (edge softness)
    temperature = trial.suggest_float('temperature', 0.9, 1.2)
    
    # Optimizer choice
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])

    # Load preprocessed binary file lists
    train_files_all, val_files_all = ensure_hpo_filelists(DATASET_ROOT)
    if not train_files_all or not val_files_all:
        raise RuntimeError(
            f"HPO file lists are empty. Check dataset path '{DATASET_ROOT.as_posix()}' and ensure binary/ directory exists."
        )
    
    # Use subset for faster HPO (10% of data)
    def _sample_fixed(lst: List[str], salt: int = 0) -> List[str]:
        k = max(1, int(len(lst) * HPO_SUBSET_FRAC))
        if len(lst) <= k:
            return lst
        rnd = random.Random(HPO_SEED + salt)
        return rnd.sample(lst, k)

    train_files = _sample_fixed(train_files_all, salt=1)
    val_files = _sample_fixed(val_files_all, salt=2)

    print(f"[Trial {trial.number}] base={base}, lr={lr:.2e}, wd={wd:.2e}, batch={batch}, bce_weight={bce_weight:.2f}")
    print(f"[Trial {trial.number}] Using {len(train_files)} train, {len(val_files)} val samples")

    # Fast binary datasets
    train_ds = FastBinaryDataset(DATASET_ROOT, train_files)
    val_ds = FastBinaryDataset(DATASET_ROOT, val_files)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, 
        num_workers=HPO_NUM_WORKERS, pin_memory=True, 
        persistent_workers=(HPO_NUM_WORKERS > 0),
        prefetch_factor=8  # Aggressive prefetch for speed
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch * 2, shuffle=False,  # Larger batch for validation
        num_workers=HPO_NUM_WORKERS, pin_memory=True, 
        persistent_workers=(HPO_NUM_WORKERS > 0),
        prefetch_factor=8
    )

    model = ResUNet(in_ch=3, base=base, use_se=True).to(device)
    model = model.to(memory_format=torch.channels_last)

    # Optimizer with fused option (faster on CUDA)
    opt_class = torch.optim.AdamW if optimizer_name == 'AdamW' else torch.optim.Adam
    opt = opt_class(model.parameters(), lr=lr, weight_decay=wd, fused=(device.type == 'cuda'))

    loss_cfg = LossConfig(
        bce_weight=float(bce_weight),
        edge_weight=float(edge_weight),
        ssim_weight=0.0,
        edge_downsample=2,
        edge_coarse_weight=0.0,
        ssim_window_size=11,
        temperature=float(temperature),
        use_ssim=False,
        use_direction_loss=False,
        use_multiscale=False
    )

    # AMP for 30-45% speedup on training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    n_epochs = HPO_EPOCHS
    for epoch in range(n_epochs):
        model.train()
        train_iter = tqdm(train_loader, desc=f"Trial {trial.number} Train Ep {epoch+1}/{n_epochs}", leave=False, unit='batch', mininterval=2.0)
        
        for batch_idx, batch_data in enumerate(train_iter):
            imgs = batch_data['image'].to(device, non_blocking=True, memory_format=torch.channels_last)
            masks = batch_data['mask'].to(device, non_blocking=True, memory_format=torch.channels_last)
            
            # AMP for training speed with gradient accumulation
            if scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    preds = model(imgs)
                    loss = sota_loss(preds, masks, loss_cfg)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale loss
                
                scaler.scale(loss).backward()
                
                # Only step optimizer every N batches
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
            else:
                preds = model(imgs)
                loss = sota_loss(preds, masks, loss_cfg)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    opt.step()
                    opt.zero_grad()
            
            # Update progress bar less frequently
            if batch_idx % 5 == 0:
                train_iter.set_postfix({'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}'})

        if (epoch + 1) % 2 == 0:
            model.eval()
            
            # Freeze GN/SE for validation speedup
            for m in model.modules():
                if isinstance(m, (nn.GroupNorm, SEBlock)):
                    m.eval()
            
            val_losses = []
            mae_vals = []
            
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                val_iter = tqdm(val_loader, desc=f"Trial {trial.number} Val Ep {epoch+1}/{n_epochs}", leave=False, unit='batch', mininterval=2.0)
                for vb in val_iter:
                    imgs = vb['image'].to(device, non_blocking=True, memory_format=torch.channels_last)
                    masks = vb['mask'].to(device, non_blocking=True, memory_format=torch.channels_last)
                    
                    # Use AMP in validation too for speed (fp16)
                    out = model(imgs)
                    loss_val = sota_loss(out, masks, loss_cfg)
                    val_losses.append(float(loss_val.item()))
                    mae_vals.append(float(soft_mae_from_logits(out, masks, temperature=loss_cfg.temperature).item()))

            mean_loss = float(np.mean(val_losses)) if val_losses else 0.0
            mean_mae = float(np.mean(mae_vals)) if mae_vals else 1.0

            step = (epoch // 2)
            composite_score = -mean_mae
            
            trial.report(composite_score, step)
            trial.set_user_attr(
                f"val_metrics_step_{step}",
                {
                    "loss": mean_loss,
                    "mae": mean_mae,
                    "composite_score": composite_score,
                    "bce_weight": float(bce_weight),
                    "edge_weight": float(edge_weight),
                    "temperature": float(temperature),
                    "optimizer": optimizer_name,
                },
            )
            print(
                f"Trial {trial.number} Val step {step}: "
                f"composite={composite_score:.4f} (mae={mean_mae:.4f})"
            )
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            epoch_val_score = composite_score

    return epoch_val_score if epoch_val_score is not None else 0.0


def main():
    global HPO_NUM_WORKERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--study-name', type=str, default='final_hpo_unet')
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--hpo-num-workers', type=int, default=HPO_NUM_WORKERS)
    args = parser.parse_args()
    if getattr(args, 'hpo_num_workers', None) is not None:
        HPO_NUM_WORKERS = args.hpo_num_workers

    if args.storage:
        storage = args.storage
    else:
        db_path = Path(__file__).parent / 'optuna_hpo.db'
        storage = f"sqlite:///{db_path.as_posix()}"

    print('Found files:', len(build_file_list(DATASET_ROOT)))
    if len(build_file_list(DATASET_ROOT)) == 0:
        print(f"No image files found in {DATASET_ROOT}. Please generate the dataset or point --dataset-root to the correct folder.")
        return
    # Maximize perceptual composite: -MAE + β*SSIM
    study = optuna.create_study(direction='maximize', pruner=SuccessiveHalvingPruner(), storage=storage, study_name=args.study_name, load_if_exists=True)

    # Run optimization (graceful Ctrl+C)
    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Keeping study results so far.")

    # Print best trial only if trials were completed
    if len(study.trials) > 0:
        try:
            print('Best trial:', study.best_trial.number, study.best_trial.value)
            print(study.best_trial.params)
        except ValueError:
            print('No completed trials found.')
    else:
        print('No trials were executed.')


if __name__ == '__main__':
    main()
