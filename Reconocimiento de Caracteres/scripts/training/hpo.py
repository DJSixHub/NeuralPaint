# HPO para segmentación con anti-aliasing: optimiza BCE + edge (loss), compara trials por MAE.
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

# Valores por defecto para HPO
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True  # Optimiza kernels para tamaño fijo
torch.backends.cudnn.deterministic = False  # Desactiva determinismo por velocidad
torch.backends.cuda.matmul.allow_tf32 = True  # Habilita TF32 en Ampere+
torch.backends.cudnn.allow_tf32 = True

DATASET_ROOT = Path(__file__).parent.parent.parent / 'datasets' / 'synthetic'
HPO_SUBSET_FRAC = 0.1  # 10% subset para trials rápidos
HPO_NUM_WORKERS = 12
HPO_EPOCHS = 8
GRADIENT_ACCUMULATION_STEPS = 2  # Reduce optimizer overhead


# ============================================================================
# LOSS & METRIC FUNCTIONS
# ============================================================================

def bce_soft_loss_from_logits(logits: torch.Tensor, targets_soft: torch.Tensor) -> torch.Tensor:
    # BCE sobre objetivos suavizados/anti-aliased
    return F.binary_cross_entropy_with_logits(logits, targets_soft)


def soft_mae_from_logits(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    # MAE sobre predicciones suaves; controla aliasing y halos
    probs = torch.sigmoid(logits / temperature)
    return (probs - targets_soft).abs().mean()


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
    temperature: float = 1.0,
) -> torch.Tensor:
    # Pérdida de bordes con magnitud de gradiente Sobel
    if weight <= 0:
        return logits.new_tensor(0.0)

    probs = torch.sigmoid(logits / temperature)
    kx, ky = _get_cached_sobel(logits.device, logits.dtype)

    # Calcula gradientes con Sobel
    gx_p = F.conv2d(probs, kx, padding=1)
    gy_p = F.conv2d(probs, ky, padding=1)
    gx_t = F.conv2d(targets_soft, kx, padding=1)
    gy_t = F.conv2d(targets_soft, ky, padding=1)
    
    # Magnitud de gradiente
    g_mag_p = (gx_p.square() + gy_p.square() + 1e-8).sqrt()
    g_mag_t = (gx_t.square() + gy_t.square() + 1e-8).sqrt()

    # Pérdida de magnitud
    mag_loss = (g_mag_p - g_mag_t).abs().mean()
    
    return weight * mag_loss


@dataclass
class LossConfig:
    bce_weight: float = 0.6
    edge_weight: float = 0.4
    temperature: float = 1.0


def sota_loss(
    logits: torch.Tensor,
    targets_soft: torch.Tensor,
    cfg: LossConfig,
) -> torch.Tensor:
    # Pérdida para HPO: BCE + bordes (magnitud Sobel)
    bce = bce_soft_loss_from_logits(logits, targets_soft)
    edge = edge_loss_from_logits(
        logits, targets_soft,
        weight=cfg.edge_weight,
        temperature=cfg.temperature
    )
    
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
    # U-Net residual con GroupNorm, SiLU y SE opcional
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
# DATASET
# ============================================================================

# ============================================================================
# DATASET
# ============================================================================

class FastBinaryDataset(Dataset):
    # Dataset optimizado con binarios .pt normalizados a 256×256
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
    # Construye lista desde caché binario (sin extensión .pt)
    binary_dir = root / 'binary'
    if not binary_dir.exists():
        return []
    
    files = []
    for p in sorted(binary_dir.rglob('*.pt')):
        try:
            # Ruta relativa sin .pt para casar nombres
            rel = p.relative_to(binary_dir).as_posix()
            rel = rel[:-3] if rel.endswith('.pt') else rel  # Quita .pt
        except Exception:
            rel = p.stem  # Usa solo el nombre
        files.append(rel)
    return files


# Genera o carga listas fijas de archivos para HPO (reproducible).
HPO_SEED = 1234
def ensure_hpo_filelists(root: Path, train_fname: str = 'hpo_train_files.txt', val_fname: str = 'hpo_val_files.txt', val_frac: float = 0.2):
    train_path = root / train_fname
    val_path = root / val_fname
    # Si existen las listas, se usan; si están vacías, se regeneran desde las imágenes
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
    # Loss de entrenamiento: BCE + edge | Métrica de comparación: -MAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Hiperparámetros sugeridos (simplificado para estabilidad)
        base = trial.suggest_categorical('base', [24, 32])
        lr = trial.suggest_float('lr', 3e-4, 1e-3, log=True)
        wd = trial.suggest_float('wd', 1e-8, 1e-6, log=True)
        batch = trial.suggest_categorical('batch', [16, 32])
    
        # Pesos de la pérdida de entrenamiento
        bce_weight = trial.suggest_float('bce_weight', 0.5, 0.7)
        edge_weight = trial.suggest_float('edge_weight', 0.3, 0.5)
    
        # Temperatura de activación (suaviza bordes)
        temperature = trial.suggest_float('temperature', 0.9, 1.2)
    
        # Selección de optimizador
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])

        # Carga listas binarias preprocesadas
        train_files_all, val_files_all = ensure_hpo_filelists(DATASET_ROOT)
        if not train_files_all or not val_files_all:
            raise RuntimeError(
                f"HPO file lists are empty. Check dataset path '{DATASET_ROOT.as_posix()}' and ensure binary/ directory exists."
            )
    
        # Subconjunto para HPO rápido (10 % de datos)
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

        # Datasets binarios rápidos
        train_ds = FastBinaryDataset(DATASET_ROOT, train_files)
        val_ds = FastBinaryDataset(DATASET_ROOT, val_files)
    
        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=True, 
            num_workers=HPO_NUM_WORKERS, pin_memory=True, 
            persistent_workers=(HPO_NUM_WORKERS > 0),
            prefetch_factor=8  # Aggressive prefetch for speed
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch, shuffle=False,  # Same batch size as training
            num_workers=HPO_NUM_WORKERS, pin_memory=True, 
            persistent_workers=(HPO_NUM_WORKERS > 0),
            prefetch_factor=8
        )

        # Activa checkpointing con batch=32 para ahorrar memoria
        use_checkpoint = (batch >= 32)
        model = ResUNet(in_ch=3, base=base, use_se=True, use_checkpoint=use_checkpoint).to(device)
        model = model.to(memory_format=torch.channels_last)
        
        if use_checkpoint:
            print(f"[Trial {trial.number}] Gradient checkpointing enabled for batch={batch}")

        # Optimizador con opción fused en CUDA
        opt_class = torch.optim.AdamW if optimizer_name == 'AdamW' else torch.optim.Adam
        opt = opt_class(model.parameters(), lr=lr, weight_decay=wd, fused=(device.type == 'cuda'))

        loss_cfg = LossConfig(
            bce_weight=float(bce_weight),
            edge_weight=float(edge_weight),
            temperature=float(temperature)
        )

        # AMP para acelerar 30-45 %
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

        epoch_val_score = None
        n_epochs = HPO_EPOCHS
        for epoch in range(n_epochs):
            model.train()
            opt.zero_grad(set_to_none=True)  # Gradientes limpios al inicio
            train_iter = tqdm(train_loader, desc=f"Trial {trial.number} Train Ep {epoch+1}/{n_epochs}", leave=False, unit='batch', mininterval=2.0)
        
            for batch_idx, batch_data in enumerate(train_iter):
                imgs = batch_data['image'].to(device, non_blocking=True, memory_format=torch.channels_last)
                masks = batch_data['mask'].to(device, non_blocking=True, memory_format=torch.channels_last)
            
                # AMP para velocidad con acumulación
                if scaler is not None:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        preds = model(imgs)
                        loss = sota_loss(preds, masks, loss_cfg)
                        loss = loss / GRADIENT_ACCUMULATION_STEPS  # Escala por acumulación
                
                    scaler.scale(loss).backward()
                
                    # Paso del optimizador cada N lotes
                    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                else:
                    preds = model(imgs)
                    loss = sota_loss(preds, masks, loss_cfg)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                    loss.backward()
                
                    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
            
                # Barra de progreso con menos actualizaciones para evitar sincronización
                if batch_idx % 10 == 0:
                    # Don't sync on every update - mininterval handles refresh rate
                    pass
        
            # Maneja lote final si no divide por pasos de acumulación
            if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            # Valida cada 2 épocas para decisiones de pruning
            if (epoch + 1) % 2 == 0:
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
                    
                        # Usa AMP en validación para velocidad (fp16)
                        out = model(imgs)
                        loss_val = sota_loss(out, masks, loss_cfg)
                        val_losses.append(float(loss_val.item()))
                        mae_vals.append(float(soft_mae_from_logits(out, masks, temperature=loss_cfg.temperature).item()))

                mean_loss = float(np.mean(val_losses)) if val_losses else 0.0
                mean_mae = float(np.mean(mae_vals)) if mae_vals else 1.0

                # Reporta al pruner en pasos intermedios (épocas 2,4,6,8)
                step = ((epoch + 1) // 2) - 1
                # Métrica de comparación: -MAE (no es la loss de entrenamiento)
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
            
                # Limpia caché CUDA tras validación
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
                epoch_val_score = composite_score

        # Limpia memoria entre pruebas para evitar OOM
        del model, opt, scaler, train_loader, val_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
        return epoch_val_score if epoch_val_score is not None else 0.0
    
    except optuna.TrialPruned:
        # Pruning esperado por SuccessiveHalvingPruner
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        raise
    
    except torch.OutOfMemoryError as e:
        print(f"[Trial {trial.number}] OOM Error with batch={batch}: {e}")
        print(f"[Trial {trial.number}] Auto-reducing batch size and retrying...")
        
        # Limpia memoria
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Reintenta con batch menor si es posible
        if batch > 8:
            new_batch = batch // 2
            print(f"[Trial {trial.number}] Retrying with batch={new_batch}")
            # Guarda batch original y reducido
            trial.set_user_attr('original_batch', batch)
            trial.set_user_attr('reduced_batch', new_batch)
            # Se poda para probar batch más pequeño en otro trial
            raise optuna.TrialPruned(f"OOM with batch={batch}, recommend batch<={new_batch}")
        else:
            # Ya está en el mínimo, no se puede reducir más
            raise optuna.TrialPruned(f"OOM even with minimum batch={batch}")
    
    except Exception as e:
        print(f"[Trial {trial.number}] Unexpected error: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        raise


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
    
    # Pruning agresivo: corta pruebas poco prometedoras
    # SuccessiveHalvingPruner mantiene el 50 % en cada ronda (pasos 0,1,2)
    pruner = SuccessiveHalvingPruner(
        min_resource=1,      # Minimum validation steps before pruning
        reduction_factor=2,  # Keep top 50% of trials at each rung
        min_early_stopping_rate=0  # Start pruning immediately
    )
    
    # Maximiza -MAE para comparar trials (menor MAE = mejor calidad)
    study = optuna.create_study(
        direction='maximize', 
        pruner=pruner, 
        storage=storage, 
        study_name=args.study_name, 
        load_if_exists=True
    )

    # Ejecuta la optimización (Ctrl+C controlado)
    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Keeping study results so far.")

    # Muestra el mejor trial solo si hubo ejecuciones
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
