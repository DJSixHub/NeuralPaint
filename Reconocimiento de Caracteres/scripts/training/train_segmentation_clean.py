# Entrenamiento etapa 1: U-Net con BCE + pérdida de bordes, usando binarios .pt.
# Configuración basada en los mejores hiperparámetros del HPO (Trial 1: MAE=0.00218).
from pathlib import Path
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optimizaciones de rendimiento
torch.backends.cudnn.benchmark = True  # Ajusta kernels para tamaño fijo
torch.backends.cudnn.deterministic = False  # Desactiva determinismo para velocidad
torch.backends.cuda.matmul.allow_tf32 = True  # Habilita TF32 en Ampere+
torch.backends.cudnn.allow_tf32 = True

# Mejores hiperparámetros del HPO (Trial 1)
BEST_BASE = 24
BEST_LR = 0.0008219135273838939
BEST_WD = 5.930255053475689e-08
BEST_BATCH = 16
BEST_BCE_WEIGHT = 0.5955360489927681
BEST_EDGE_WEIGHT = 0.4825539311206654
BEST_TEMPERATURE = 0.940152548229547

# Valores por defecto de entrenamiento
DEFAULT_EPOCHS = 50
DEFAULT_VAL_FRAC = 0.2
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 7


# ============================================================================
# LOSS & METRIC FUNCTIONS
# ============================================================================

def bce_soft_loss_from_logits(logits: torch.Tensor, targets_soft: torch.Tensor) -> torch.Tensor:
    # BCE sobre objetivos suavizados
    return F.binary_cross_entropy_with_logits(logits, targets_soft)


def weighted_bce_loss_from_logits(
    logits: torch.Tensor, 
    targets_soft: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 3.0,
) -> torch.Tensor:
    # BCE ponderada para fine-tuning; penaliza falsos positivos
    # pos_weight y neg_weight controlan texto vs fondo
    # Calcula BCE por píxel
    bce_per_pixel = F.binary_cross_entropy_with_logits(logits, targets_soft, reduction='none')
    
    # Aplica pesos: pos_weight texto, neg_weight fondo
    weights = torch.where(targets_soft > 0.5, pos_weight, neg_weight)
    weighted_bce = (bce_per_pixel * weights).mean()
    
    return weighted_bce


def soft_mae_from_logits(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    # MAE sobre predicciones suaves; mide aliasing y halos
    probs = torch.sigmoid(logits / temperature)
    return (probs - targets_soft).abs().mean()


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype)
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


# Cache de Sobel global para velocidad
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
    temperature: float = 1.0,
) -> torch.Tensor:
    # Pérdida de bordes con magnitud de gradiente
    if weight <= 0:
        return logits.new_tensor(0.0)

    probs = torch.sigmoid(logits / temperature)
    kx, ky = _get_cached_sobel(logits.device, logits.dtype)

    # Gradientes con Sobel
    gx_p = F.conv2d(probs, kx, padding=1)
    gy_p = F.conv2d(probs, ky, padding=1)
    gx_t = F.conv2d(targets_soft, kx, padding=1)
    gy_t = F.conv2d(targets_soft, ky, padding=1)
    
    # Magnitud de gradiente
    g_mag_p = (gx_p.square() + gy_p.square() + 1e-8).sqrt()
    g_mag_t = (gx_t.square() + gy_t.square() + 1e-8).sqrt()

    mag_loss = (g_mag_p - g_mag_t).abs().mean()
    return weight * mag_loss


@dataclass
class LossConfig:
    bce_weight: float = 0.6
    edge_weight: float = 0.5
    temperature: float = 1.0
    # Parámetros para fine-tuning (etapa 2)
    use_weighted_bce: bool = False
    pos_weight: float = 1.0
    neg_weight: float = 3.0


def sota_loss(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    cfg: LossConfig,
) -> torch.Tensor:
    # Pérdida para etapa 1 y 2 (BCE + bordes; ponderada en fine-tuning)
    if cfg.use_weighted_bce:
        bce = weighted_bce_loss_from_logits(
            logits, targets_soft, 
            pos_weight=cfg.pos_weight, 
            neg_weight=cfg.neg_weight
        )
    else:
        bce = bce_soft_loss_from_logits(logits, targets_soft)
    
    edge = edge_loss_from_logits(logits, targets_soft, weight=cfg.edge_weight, temperature=cfg.temperature)
    return cfg.bce_weight * bce + edge


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
    # U-Net residual con GN/SiLU y SE opcional
    def __init__(self, in_ch: int = 3, base: int = 32, use_se: bool = True, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
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
        # Usa checkpointing de gradientes para ahorrar memoria si está activo
        if self.use_checkpoint and self.training:
            e1 = torch.utils.checkpoint.checkpoint(self.enc1, x, use_reentrant=False)
            p1 = self.pool1(e1)
            e2 = torch.utils.checkpoint.checkpoint(self.enc2, p1, use_reentrant=False)
            p2 = self.pool2(e2)
            e3 = torch.utils.checkpoint.checkpoint(self.enc3, p2, use_reentrant=False)
            u2 = self.up2(e3)
            d2 = torch.utils.checkpoint.checkpoint(self.dec2, torch.cat([u2, e2], dim=1), use_reentrant=False)
            u1 = self.up1(d2)
            d1 = torch.utils.checkpoint.checkpoint(self.dec1, torch.cat([u1, e1], dim=1), use_reentrant=False)
        else:
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

# Etapa 1: BCE + magnitud de bordes (config HPO)
LOSS_CFG = LossConfig(
    bce_weight=BEST_BCE_WEIGHT,
    edge_weight=BEST_EDGE_WEIGHT,
    temperature=BEST_TEMPERATURE
)


class FastBinaryDataset(Dataset):
    # Dataset rápido con binarios .pt normalizados a 256×256
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
        
        # Carga rápida desde binario preprocesado
        data = torch.load(p_bin, map_location='cpu', weights_only=True)
        
        # Normaliza a [0,1]
        img = data['img'].float() / 255.0
        mask = data['mask'].float() / 255.0
        
        return {
            'image': img,
            'mask': mask
        }


def build_file_list(root: Path):
    # Construye lista desde el directorio binario
    binary_dir = root / 'binary'
    if not binary_dir.exists():
        return []
    
    files = []
    for p in sorted(binary_dir.rglob('*.pt')):
        rel = p.relative_to(binary_dir).as_posix()
        rel = rel[:-3] if rel.endswith('.pt') else rel  # Quita .pt
        files.append(rel)
    return files


def train_and_evaluate(
    dataset_root: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BEST_BATCH,
    val_frac: float = DEFAULT_VAL_FRAC,
    num_workers: int = 8,
    fine_tune: bool = False,
    fine_tune_model: str = None,
    resume: str = None,
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
    
    # DataLoaders optimizados para velocidad
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device='cuda',  # PyTorch 2.1+: evita salto CPU→GPU extra
        persistent_workers=(num_workers > 0),
        prefetch_factor=8 if num_workers > 0 else None,  # Prefetch agresivo
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device='cuda',  # PyTorch 2.1+: evita salto CPU→GPU extra
        persistent_workers=(num_workers > 0),
        prefetch_factor=8 if num_workers > 0 else None,  # Prefetch agresivo
    )
    
    # Directorio de salida para checkpoints
    out_dir = Path(__file__).parent.parent.parent / 'models' / 'segmentation'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuración de fine-tuning (etapa 2)
    if fine_tune:
        learning_rate = 1e-5  # Very low LR for fine-tuning
        LOSS_CFG.use_weighted_bce = True
        LOSS_CFG.pos_weight = 1.0
        LOSS_CFG.neg_weight = 3.0
        stage_name = "Stage 2 (Fine-tuning)"
    else:
        learning_rate = BEST_LR
        stage_name = "Stage 1"
    
    # Modelo
    model = ResUNet(in_ch=3, base=BEST_BASE, use_se=True, use_checkpoint=False).to(device)
    model = model.to(memory_format=torch.channels_last)
    
    # Optimizador
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=BEST_WD)
    
    # Carga checkpoint para reanudar o afinar
    start_epoch = 0
    best_val_loss = float('inf')
    best_composite_score = -float('inf')
    
    if resume:
        # Reanuda desde checkpoint (modelo + optimizador + época)
        resume_path = Path(resume)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            
            # Carga estado de modelo
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            else:
                model.load_state_dict(ckpt)
            
            # Carga estado del optimizador
            if 'opt_state' in ckpt:
                try:
                    opt.load_state_dict(ckpt['opt_state'])
                except:
                    print("Warning: Could not load optimizer state (LR may have changed)")
            
            # Recupera estado de entrenamiento
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Best val loss: {best_val_loss:.6f}")
        else:
            print(f"Warning: Resume checkpoint not found at {resume_path}")
            print("Starting from scratch...")
    
    elif fine_tune:
        # Fine-tuning: solo pesos de modelo (optimizador nuevo)
        if fine_tune_model is None:
            fine_tune_model = out_dir / 'checkpoint_epoch_30.pth'
        else:
            fine_tune_model = Path(fine_tune_model)
        
        if fine_tune_model.exists():
            print(f"Loading pretrained model: {fine_tune_model}")
            ckpt = torch.load(fine_tune_model, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            else:
                model.load_state_dict(ckpt)
        else:
            print(f"Warning: Fine-tune model not found at {fine_tune_model}")
            print("Starting from scratch...")
    
    # Precisión mixta según versión de PyTorch
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
    
    # Bucle de entrenamiento
    train_losses = []
    val_losses = []
    val_maes = []
    epochs_no_improve = 0
    
    print(f"\n{stage_name}: BCE + Edge magnitude")
    if fine_tune:
        print(f"  Weighted BCE: pos_weight={LOSS_CFG.pos_weight}, neg_weight={LOSS_CFG.neg_weight}")
    if resume:
        print(f"  Resuming from epoch {start_epoch}")
    print(f"Params: base={BEST_BASE}, lr={learning_rate:.2e}, batch={batch_size}")
    print(f"Device: {device}, AMP: {scaler is not None}\n")
    
    for epoch in range(start_epoch, epochs):
        # Entrenamiento con acumulación de gradientes
        model.train()
        opt.zero_grad(set_to_none=True)  # Gradientes limpios al inicio
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', mininterval=2.0)
        
        for batch_idx, batch in enumerate(pbar):
            imgs = batch['image'].to(device, non_blocking=True, memory_format=torch.channels_last)
            masks = batch['mask'].to(device, non_blocking=True, memory_format=torch.channels_last)
            
            with autocast_fn():
                preds = model(imgs)
                loss = sota_loss(preds, masks, LOSS_CFG)
                loss = loss / GRADIENT_ACCUMULATION_STEPS  # Escala por acumulación
            
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Paso del optimizador cada N lotes
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                loss.backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            
            running_loss += loss.item() * imgs.size(0) * GRADIENT_ACCUMULATION_STEPS
            # Evita pbar.set_postfix para no sincronizar GPU
        
        # Maneja último lote incompleto
        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        
        epoch_train_loss = running_loss / len(train_ds)
        train_losses.append(epoch_train_loss)
        
        # Validación (cada 2 épocas)
        if (epoch + 1) % 2 == 0:
            model.eval()
            sum_val_loss = 0.0
            sum_val_mae = 0.0
            cnt_val = 0
            
            with torch.no_grad(), autocast_fn():
                for vb in tqdm(val_loader, desc='Validation', leave=False, mininterval=2.0):
                    imgs = vb['image'].to(device, non_blocking=True, memory_format=torch.channels_last)
                    masks = vb['mask'].to(device, non_blocking=True, memory_format=torch.channels_last)
                    
                    out = model(imgs)
                    loss_val = sota_loss(out, masks, LOSS_CFG)
                    sum_val_loss += loss_val.item() * imgs.size(0)
                    sum_val_mae += soft_mae_from_logits(out, masks, temperature=LOSS_CFG.temperature).item() * imgs.size(0)
                    cnt_val += imgs.size(0)
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            epoch_val_loss = sum_val_loss / cnt_val
            epoch_val_mae = sum_val_mae / cnt_val
            val_losses.append(epoch_val_loss)
            val_maes.append(epoch_val_mae)
            
            print(f'Epoch {epoch+1}: train={epoch_train_loss:.4f}, val={epoch_val_loss:.4f}, mae={epoch_val_mae:.4f}')
            
            composite_score = -epoch_val_mae
            
            if composite_score > best_composite_score:
                epochs_no_improve = 0
                best_composite_score = composite_score
                best_val_loss = min(best_val_loss, epoch_val_loss)
                best_model_name = 'fine_tuning_best_model.pth' if fine_tune else 'best_model.pth'
                torch.save(model.state_dict(), out_dir / best_model_name)
                print(f'  → Best: mae={epoch_val_mae:.4f}')
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping ({EARLY_STOPPING_PATIENCE} validations)')
                break
            
            checkpoint_name = f'fine_tuning_checkpoint_epoch_{epoch+1}.pth' if fine_tune else f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'scaler_state': scaler.state_dict() if scaler else {},
                'best_val_loss': best_val_loss,
            }, out_dir / checkpoint_name)
    
    # Guarda modelo final
    final_model_name = 'fine_tuning_final_model.pth' if fine_tune else 'final_model.pth'
    torch.save(model.state_dict(), out_dir / final_model_name)
    
    # Gráfica de curvas (Stage 1: Loss + MAE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de pérdida train/val
    ax1 = axes[0]
    epochs_train = list(range(1, len(train_losses) + 1))
    epochs_val = [i * 2 for i in range(1, len(val_losses) + 1)]
    
    ax1.plot(epochs_train, train_losses, label='Train Loss', color='#2E86AB', linewidth=2)
    ax1.plot(epochs_val, val_losses, label='Val Loss', color='#E63946', linewidth=2, marker='o', markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss (Stage 1)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, len(train_losses) + 1)
    
    # Gráfica de MAE
    ax2 = axes[1]
    ax2.plot(epochs_val, val_maes, label='Val MAE', color='#E63946', linewidth=2, marker='s', markersize=5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax2.set_title('Validation MAE (Stage 1)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, len(train_losses) + 1)
    
    # Anotación del mejor MAE
    if val_maes:
        best_mae = min(val_maes)
        best_mae_epoch = epochs_val[val_maes.index(best_mae)]
        ax2.axhline(y=best_mae, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax2.text(1, best_mae + 0.002, f'Best: {best_mae:.4f}', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Gráfica resumen detallada
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vista combinada con doble eje Y
    ax_twin = ax.twinx()
    
    line1 = ax.plot(epochs_train, train_losses, label='Train Loss', color='#2E86AB', linewidth=2, alpha=0.8)
    line2 = ax.plot(epochs_val, val_losses, label='Val Loss', color='#E63946', linewidth=2, marker='o', markersize=4)
    line3 = ax_twin.plot(epochs_val, val_maes, label='Val MAE', color='#FF6B35', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#2E86AB')
    ax_twin.set_ylabel('MAE', fontsize=12, fontweight='bold', color='#FF6B35')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax_twin.tick_params(axis='y', labelcolor='#FF6B35')
    
    # Combina leyendas
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=10)
    
    best_mae = min(val_maes) if val_maes else 1
    ax.set_title(f'Stage 1 Training Summary - Best MAE: {best_mae:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'training_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f'\n{stage_name} complete: Best MAE={min(val_maes):.4f}')
    print(f'Models: {out_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stage 1/2: U-Net training')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Dataset root (default: datasets/synthetic for Stage 1, datasets/synthetic_finetuning for Stage 2)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BEST_BATCH)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--fine-tune', action='store_true',
                        help='Stage 2: Fine-tune with weighted BCE (load from checkpoint_epoch_30.pth)')
    parser.add_argument('--fine-tune-model', type=str, default=None,
                        help='Path to pretrained model for fine-tuning (default: models/segmentation/checkpoint_epoch_30.pth)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint (loads model + optimizer + epoch)')
    args = parser.parse_args()
    
    # Selecciona data-root según la etapa
    if args.data_root is None:
        if args.fine_tune or args.resume:
            data_root = str(Path(__file__).parent.parent.parent / 'datasets' / 'synthetic_finetuning')
        else:
            data_root = str(Path(__file__).parent.parent.parent / 'datasets' / 'synthetic')
    else:
        data_root = args.data_root
    
    train_and_evaluate(
        dataset_root=data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fine_tune=args.fine_tune,
        fine_tune_model=args.fine_tune_model,
        resume=args.resume,
    )
