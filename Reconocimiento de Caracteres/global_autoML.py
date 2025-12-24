# ============================
# Configuración global y librerías
# ============================
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.pruners import SuccessiveHalvingPruner
import optuna.visualization as vis
import argparse
from tqdm import tqdm

# Semillas y dispositivo
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DATASET_ROOT = Path("E:/Escuela/Redes Neuronales/NeuralPaint/Reconocimiento de Caracteres/datasets")
BATCH_SIZE = 16
VAL_FRAC = 0.3
NUM_WORKERS = 2   # ajusta según CPU, prueba 0 si se atasca
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Activar benchmark de cuDNN para convoluciones más rápidas
torch.backends.cudnn.benchmark = True

# ============================
# Dataset optimizado con OpenCV
# ============================
class SyntheticGlyphsDataset(Dataset):
    def __init__(self, root: Path, files: List[str], target_size: Tuple[int, int] = (512, 512)):
        self.root = Path(root)
        self.images_dir = self.root / 'images'
        self.masks_dir = self.root / 'masks'
        self.masks_ignore_dir = self.root / 'masks_ignore'
        self.files = list(files)
        self.target_size = target_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]

        # --- Imagen RGB ---
        img_path = str(self.images_dir / fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0

        # --- Máscara ---
        mask_path = str(self.masks_dir / fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        mask_bin = (mask_t > 0.0).float()

        # --- Máscara ignore ---
        ignore_path = str(self.masks_ignore_dir / fname)
        if os.path.exists(ignore_path):
            ignore = cv2.imread(ignore_path, cv2.IMREAD_GRAYSCALE)
            ignore = cv2.resize(ignore, self.target_size, interpolation=cv2.INTER_NEAREST)
        else:
            ignore = np.zeros(self.target_size, dtype=np.uint8)
        ignore_t = torch.from_numpy(ignore).unsqueeze(0).float() / 255.0
        ignore_bin = (ignore_t > 0.0).float()

        return {'image': img_t, 'mask': mask_bin, 'ignore': ignore_bin, 'file': fname}

def build_file_list(root: Path) -> List[str]:
    images_dir = root / 'images'
    if not images_dir.exists():
        return []
    return [p.name for p in sorted(images_dir.glob('*.png'))]

def split_files(files: List[str], val_frac: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    files_shuffled = list(files)
    rng.shuffle(files_shuffled)
    n_val = int(len(files_shuffled) * val_frac)
    return files_shuffled[n_val:], files_shuffled[:n_val]

# ============================
# DataLoaders optimizados
# ============================
# HPO defaults (used inside objective to build local loaders)
HPO_TARGET_SIZE = (128, 128)
HPO_SUBSET_FRAC = 0.1  # 10% del dataset
HPO_BATCH_SIZE = 32
HPO_NUM_WORKERS = 0  # default for safety on Windows; may be overridden by CLI
USE_AMP = True  # enable automatic mixed precision by default when CUDA is available

# Note: we avoid creating DataLoaders at import time on Windows because
# multiprocessing with 'spawn' imports the module in child processes.
# The DataLoaders for HPO will be constructed inside `objective` so each
# trial (possibly running in a separate process) builds its own loaders.

# ============================
# Modelos
# ============================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.up(e2)
        d = self.dec1(d)
        return self.outc(d)

class SimpleFCN(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.conv1 = DoubleConv(in_ch, base)
        self.conv2 = DoubleConv(base, base*2)
        self.conv3 = DoubleConv(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DoubleConv(base*4, base*2)
        self.dec2 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        c1 = self.conv1(x); p1 = self.pool(c1)
        c2 = self.conv2(p1); p2 = self.pool(c2)
        c3 = self.conv3(p2)
        u1 = self.up1(c3); d1 = self.dec1(u1)
        u2 = self.up2(d1); d2 = self.dec2(u2)
        return self.outc(d2)

# ============================
# Métricas
# ============================

def iou_metric(pred, target, ignore_mask=None, eps=1e-7):

    # Binarizar predicciones
    pred_bin = (pred > 0.5).float()
    target = target.float()

    # Aplicar máscara de ignore si existe
    if ignore_mask is not None:
        mask = (ignore_mask < 0.5).float()  # 1 = válido, 0 = ignorado
        pred_bin = pred_bin * mask
        target = target * mask

    # Calcular intersección y unión
    inter = (pred_bin * target).sum(dim=[1, 2, 3])
    union = (pred_bin + target - pred_bin * target).sum(dim=[1, 2, 3])

    # IoU promedio sobre el batch
    return ((inter + eps) / (union + eps)).mean().item()

    
# ============================
# Objective Optuna
# ============================
def objective(trial: optuna.trial.Trial) -> float:
    # Selección de arquitectura y parámetros
    model_type = trial.suggest_categorical('model_type', ['unet', 'fcn'])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    base = trial.suggest_categorical('base', [16, 32, 48])
    weight_decay = trial.suggest_float('wd', 1e-7, 1e-4, log=True)
    # Construcción del modelo
    model = SimpleUNet(in_ch=3, base=base).to(DEVICE) if model_type == 'unet' else SimpleFCN(in_ch=3, base=base).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # Construir loaders locales dentro del proceso del trial.
    # `HPO_NUM_WORKERS` controla `num_workers`. Use 0 for nested trials (safe),
    # or >0 when running this script as an independent worker process.
    files = build_file_list(DATASET_ROOT)
    train_files, val_files = split_files(files, val_frac=VAL_FRAC, seed=SEED)
    hpo_train_files = random.sample(train_files, max(1, int(len(train_files) * HPO_SUBSET_FRAC)))
    hpo_val_files = random.sample(val_files, max(1, int(len(val_files) * HPO_SUBSET_FRAC)))

    train_ds_hpo = SyntheticGlyphsDataset(DATASET_ROOT, hpo_train_files, target_size=HPO_TARGET_SIZE)
    val_ds_hpo = SyntheticGlyphsDataset(DATASET_ROOT, hpo_val_files, target_size=HPO_TARGET_SIZE)

    # IMPORTANT: use num_workers=0 inside trial processes to avoid nested spawn
    train_loader_local = DataLoader(train_ds_hpo, batch_size=HPO_BATCH_SIZE, shuffle=True,
                                    num_workers=HPO_NUM_WORKERS, pin_memory=True)
    val_loader_local = DataLoader(val_ds_hpo, batch_size=HPO_BATCH_SIZE, shuffle=False,
                                  num_workers=HPO_NUM_WORKERS, pin_memory=True)

    # Entrenamiento rápido (proxy)
    n_epochs_hpo = 3
    scaler = torch.amp.GradScaler('cuda') if (globals().get('USE_AMP', True) and DEVICE.type == 'cuda') else None
    for epoch in range(n_epochs_hpo):
        model.train()
        for batch in tqdm(train_loader_local, desc=f'Trial {trial.number}, Epoch {epoch+1} Train', leave=False):
            imgs = batch['image'].to(DEVICE, non_blocking=True)
            masks = batch['mask'].to(DEVICE, non_blocking=True)
            ignores = batch['ignore'].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    pred = model(imgs)
                    loss = loss_fn(pred, masks)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(imgs)
                loss = loss_fn(pred, masks)
                loss.backward()
                opt.step()

        # Validación
        model.eval()
        with torch.no_grad():
            ious = []
            for vb in tqdm(val_loader_local, desc=f'Trial {trial.number}, Epoch {epoch+1} Val', leave=False):
                imgs = vb['image'].to(DEVICE, non_blocking=True)
                masks = vb['mask'].to(DEVICE, non_blocking=True)
                ignores = vb['ignore'].to(DEVICE, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):
                        pred = model(imgs)
                else:
                    pred = model(imgs)
                # model returns logits now; convert to probabilities for IoU
                pred_sig = torch.sigmoid(pred)
                ious.append(iou_metric(pred_sig, masks, ignore_mask=ignores))
            val_iou = float(np.mean(ious)) if ious else 0.0

        # Reportar a Optuna
        trial.report(val_iou, epoch)
        # Reporte legible por época
        print(f"[Trial {trial.number}] Epoch {epoch+1} — val_iou={val_iou:.4f}", flush=True)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_iou

# ============================
# Estudio Optuna con paralelización
# ============================
def main():
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='HPO runner for NeuralPaint recognition')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g. sqlite:///path/to/optuna.db). If omitted, local sqlite in script folder will be used.')
    parser.add_argument('--study-name', type=str, default='neuralpaint_hpo', help='Optuna study name')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials (total across workers)')
    parser.add_argument('--n-jobs', type=int, default=2, help='n_jobs for study.optimize (use 1 for single-process)')
    parser.add_argument('--hpo-batch-size', type=int, default=HPO_BATCH_SIZE)
    parser.add_argument('--hpo-target-size', type=int, default=HPO_TARGET_SIZE[0],
                        help='Square target size for HPO, e.g. 128 or 256')
    parser.add_argument('--hpo-subset-frac', type=float, default=HPO_SUBSET_FRAC)
    parser.add_argument('--hpo-num-workers', type=int, default=0,
                        help='num_workers for DataLoader inside each trial. Use 0 for nested parallel (safe on Windows), or >0 when running independent workers.')
    parser.add_argument('--run-distributed', action='store_true', help='Run as independent distributed worker (set hpo-num-workers>0)')
    args = parser.parse_args()

    # Update globals from args (use globals() to avoid potential syntax/context issues)
    globals().update({
        'HPO_BATCH_SIZE': args.hpo_batch_size,
        'HPO_TARGET_SIZE': (args.hpo_target_size, args.hpo_target_size),
        'HPO_SUBSET_FRAC': args.hpo_subset_frac,
        'HPO_NUM_WORKERS': args.hpo_num_workers,
    })

    # Prepare storage
    if args.storage:
        storage = args.storage
    else:
        db_path = Path(__file__).parent / 'optuna.db'
        storage = f"sqlite:///{db_path.as_posix()}"

    files = build_file_list(DATASET_ROOT)
    print('Found files:', len(files))
    train_files, val_files = split_files(files, val_frac=VAL_FRAC, seed=SEED)
    print(f'Train: {len(train_files)} files, Val: {len(val_files)} files')
    print(f'Device: {DEVICE} (PID {os.getpid()})')

    # Crear (o cargar) estudio Optuna usando RDB backend
    study = optuna.create_study(direction='maximize', pruner=SuccessiveHalvingPruner(),
                               storage=storage, study_name=args.study_name, load_if_exists=True)

    print(f'Using storage: {storage}')
    print('HPO settings:', {'n_trials': args.n_trials, 'n_jobs': args.n_jobs, 'hpo_batch_size': HPO_BATCH_SIZE, 'hpo_target_size': HPO_TARGET_SIZE, 'hpo_num_workers': HPO_NUM_WORKERS})

    # Ejecutar optimización. For distributed mode, launch this script in multiple terminals
    # pointing to the same --storage and each process will pull trials from the DB.
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, timeout=None)

    print('Best trial:', study.best_trial.params)
    print('Best IoU:', study.best_trial.value)

    # Visualizaciones (opcional)
    try:
        fig1 = vis.plot_optimization_history(study)
        fig1.show()
        fig2 = vis.plot_param_importances(study)
        fig2.show()
        fig3 = vis.plot_contour(study)
        fig3.show()
        fig4 = vis.plot_slice(study)
        fig4.show()
    except Exception as e:
        print('Error en visualizaciones:', e)
        print('Asegúrate de tener plotly instalado: pip install plotly')

    print('Optuna HPO completado con selección de arquitectura, paralelización y visualizaciones.')


if __name__ == '__main__':
    main()
