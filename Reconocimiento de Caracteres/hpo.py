from pathlib import Path
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from tqdm import tqdm
import argparse

# HPO defaults
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DATASET_ROOT = Path(__file__).parent / 'datasets'
HPO_TARGET_SIZE = (256, 256)
HPO_SUBSET_FRAC = 0.1
HPO_NUM_WORKERS = 4
import sqlite3


class SyntheticGlyphsDataset(Dataset):
    """Tile-based dataset: yields padded tiles of size `target_size` and metadata for stitching.

    Each item returns dict with keys: 'image' (C,H,W), 'mask' (1,H,W), 'ignore' (1,H,W), 'meta' (file,x,y,crop_w,crop_h,img_w,img_h)
    """
    def __init__(self, root: Path, files: List[str], target_size: Tuple[int, int] = HPO_TARGET_SIZE):
        self.root = Path(root)
        self.images_dir = self.root / 'images'
        self.masks_dir = self.root / 'masks'
        self.masks_ignore_dir = self.root / 'masks_ignore'
        self.files = list(files)
        self.tile_w, self.tile_h = target_size
        # simple LRU cache for full images/masks to avoid repeated IO per tile
        self._img_cache: dict = {}
        self._cache_order: list = []
        self._cache_max: int = 64

        # build index of tiles without loading full images to avoid O(N) memory/IO during init
        import cv2
        self.index = []
        for fname in self.files:
            p = self.images_dir / fname
            if not p.exists():
                continue
            # read header to get shape only (avoid full-image caching here)
            img_hdr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img_hdr is None:
                continue
            h, w = int(img_hdr.shape[0]), int(img_hdr.shape[1])
            xs = list(range(0, max(1, w), self.tile_w))
            ys = list(range(0, max(1, h), self.tile_h))
            for x in xs:
                for y in ys:
                    crop_w = min(self.tile_w, max(0, w - x))
                    crop_h = min(self.tile_h, max(0, h - y))
                    self.index.append({'file': fname, 'x': x, 'y': y, 'crop_w': int(crop_w), 'crop_h': int(crop_h), 'img_w': int(w), 'img_h': int(h)})

    def _load_full(self, fname: str):
        """Load and cache full RGB image, mask and ignore arrays for `fname`.

        Returns (img_rgb, mask_full, ignore_full) as numpy arrays. Uses simple LRU eviction.
        """
        if fname in self._img_cache:
            # move to recent
            try:
                self._cache_order.remove(fname)
            except ValueError:
                pass
            self._cache_order.append(fname)
            return self._img_cache[fname]
        import cv2
        p_img = self.images_dir / fname
        img = cv2.imread(str(p_img), cv2.IMREAD_COLOR)
        if img is None:
            return None, None, None
        # convert BGR->RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # masks
        p_mask = self.masks_dir / fname
        mask_full = None
        if p_mask.exists():
            mask_full = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
        if mask_full is None:
            mask_full = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
        # ignore mask
        p_ign = self.masks_ignore_dir / fname
        ignore_full = None
        if p_ign.exists():
            ignore_full = cv2.imread(str(p_ign), cv2.IMREAD_GRAYSCALE)
        if ignore_full is None:
            ignore_full = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)

        # cache entry
        self._img_cache[fname] = (img_rgb, mask_full, ignore_full)
        self._cache_order.append(fname)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            try:
                del self._img_cache[old]
            except Exception:
                pass
        return img_rgb, mask_full, ignore_full

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        fname = entry['file']
        x = entry['x']; y = entry['y']; crop_w = entry['crop_w']; crop_h = entry['crop_h']
        img_w = entry['img_w']; img_h = entry['img_h']
        # use cached full-image arrays to avoid repeated IO
        img_rgb, mask_full, ignore_full = self._load_full(fname)
        tile = np.ones((self.tile_h, self.tile_w, 3), dtype=np.uint8) * 255
        crop = img_rgb[y:y+crop_h, x:x+crop_w]
        tile[0:crop_h, 0:crop_w] = crop
        img_t = torch.from_numpy(tile).permute(2,0,1).float() / 255.0
        if mask_full is None:
            mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_tile = np.zeros((self.tile_h, self.tile_w), dtype=np.uint8)
        mask_crop = mask_full[y:y+crop_h, x:x+crop_w]
        mask_tile[0:crop_h, 0:crop_w] = mask_crop
        mask_t = torch.from_numpy(mask_tile).unsqueeze(0).float() / 255.0
        mask_bin = (mask_t > 0.5).float()

        ignore_tile = np.zeros((self.tile_h, self.tile_w), dtype=np.uint8)
        if ignore_full is None:
            ignore_full = np.zeros((img_h, img_w), dtype=np.uint8)
        ignore_crop = ignore_full[y:y+crop_h, x:x+crop_w]
        ignore_tile[0:crop_h, 0:crop_w] = ignore_crop
        ignore_t = torch.from_numpy(ignore_tile).unsqueeze(0).float() / 255.0
        ignore_bin = (ignore_t > 0.5).float()

        meta = {'file': fname, 'x': x, 'y': y, 'crop_w': crop_w, 'crop_h': crop_h, 'img_w': img_w, 'img_h': img_h}
        return {'image': img_t, 'mask': mask_bin, 'ignore': ignore_bin, 'meta': meta}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetThin(nn.Module):
    """U-Net delgado de 3 niveles.

    Entrada: tensor (B,3,256,256). Salida: tensor (B,1,256,256) logits.
    Arquitectura: encoders 32 -> 64 -> 128, decoders con skip connections.
    No usa BatchNorm ni Dropout. Activación: ReLU.
    """
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # encoder
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)

        # decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)            # 32
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)           # 64
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)           # 128

        u2 = self.up2(e3)
        # pad/crop if needed then concat
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.outc(d1)


def build_file_list(root: Path):
    images_dir = root / 'images'
    if not images_dir.exists():
        return []
    # Support sharded layouts where images live in numbered subfolders.
    # Return paths relative to `images_dir` so callers can open masks/images using the same relative layout.
    files = []
    for p in sorted(images_dir.rglob('*.png')):
        try:
            rel = p.relative_to(images_dir).as_posix()
        except Exception:
            rel = p.name
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


def iou_metric(pred, target, ignore_mask=None, eps=1e-7):
    pred_bin = (pred > 0.5).float()
    target = target.float()
    if ignore_mask is not None:
        mask = (ignore_mask < 0.5).float()
        pred_bin = pred_bin * mask
        target = target * mask
    inter = (pred_bin * target).sum(dim=[1, 2, 3])
    union = (pred_bin + target - pred_bin * target).sum(dim=[1, 2, 3])
    return ((inter + eps) / (union + eps))


def objective(trial: optuna.trial.Trial) -> float:
    # hyperparameters
    # Study-informed search space (best previous trial: unet, base=48, lr~7.36e-4, wd~1.7e-7)
    # - bias choices toward 48 for `base` by listing it first
    # Reduced search space (user requested): base ∈ {24,32,40}, lr ∈ loguniform(3e-4,1e-3),
    # wd ∈ loguniform(1e-8,1e-6), batch ∈ {16,32}
    base = trial.suggest_categorical('base', [24, 16])
    lr = trial.suggest_float('lr', 3e-4, 1e-3, log=True)
    wd = trial.suggest_float('wd', 1e-8, 1e-6, log=True)
    batch = trial.suggest_categorical('batch', [16, 32])

    device = torch.device('cuda')

    # Fixed HPO dataset: load or generate file lists once and use the same files for all trials
    train_files_all, val_files_all = ensure_hpo_filelists(DATASET_ROOT)
    # Fail fast with a clear message if no files were found (prevents DataLoader RandomSampler ValueError)
    if not train_files_all or not val_files_all:
        raise RuntimeError(
            f"HPO file lists are empty. Check dataset path '{DATASET_ROOT.as_posix()}' and that it contains 'images' with PNG files. "
            "Run the dataset generator and ensure files are present before running HPO."
        )
    # take small subset fraction of the fixed lists for speed (but deterministically)
    def _sample_fixed(lst: List[str], salt: int = 0) -> List[str]:
        """Deterministic subsample of `lst` using fixed seed + salt."""
        k = max(1, int(len(lst) * HPO_SUBSET_FRAC))
        if len(lst) <= k:
            return list(lst)
        rnd_local = random.Random(SEED + int(salt))
        return rnd_local.sample(list(lst), k)

    hpo_train = _sample_fixed(train_files_all, salt=0)
    hpo_val = _sample_fixed(val_files_all, salt=1)

    # persistent DataLoader for speed across trials
    train_ds = SyntheticGlyphsDataset(DATASET_ROOT, hpo_train, target_size=HPO_TARGET_SIZE)
    val_ds = SyntheticGlyphsDataset(DATASET_ROOT, hpo_val, target_size=HPO_TARGET_SIZE)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=HPO_NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=HPO_NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = UNetThin(in_ch=3, base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # Use new API when available; fall back for older PyTorch versions.
    if device.type == 'cuda':
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            try:
                scaler = torch.amp.GradScaler(device_type='cuda')
            except TypeError:
                # Older torch.amp.GradScaler signature — try without keyword
                scaler = torch.amp.GradScaler('cuda')
        else:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Multi-fidelity schedule: report every epoch up to 8 (2->4->8 stages handled by pruner)
    n_epochs = 8
    for epoch in range(n_epochs):
        model.train()
        train_iter = tqdm(train_loader, desc=f"Trial {trial.number} Train Ep {epoch+1}/{n_epochs}", leave=False, unit='batch')
        for batch_data in train_iter:
            imgs = batch_data['image'].to(device, non_blocking=True)
            masks = batch_data['mask'].to(device, non_blocking=True)
            ignores = batch_data.get('ignore', torch.zeros_like(masks)).to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                preds = model(imgs)
                loss_map = loss_fn(preds, masks)
                valid_mask = (ignores < 0.5).float()
                denom = valid_mask.sum(dim=[1,2,3]).clamp_min(1.0)
                per_sample = (loss_map * valid_mask).sum(dim=[1,2,3]) / denom
                loss = per_sample.mean()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            try:
                train_iter.set_postfix({'loss': float(loss.detach().cpu().item())})
            except Exception:
                pass

        # Validate only every 2 epochs to save time; compute BCE, IOU and Dice per tile
        epoch_val_loss = None
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_losses = []
            iou_vals = []
            dice_vals = []
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Trial {trial.number} Val Ep {epoch+1}/{n_epochs}", leave=False, unit='batch')
                for vb in val_iter:
                    imgs = vb['image'].to(device, non_blocking=True)
                    masks = vb['mask'].to(device, non_blocking=True)
                    ignores = vb.get('ignore', torch.zeros_like(masks)).to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        out = model(imgs)

                    # BCE per-sample (as before)
                    loss_map = loss_fn(out, masks)
                    valid_mask = (ignores < 0.5).float()
                    denom = valid_mask.sum(dim=[1,2,3]).clamp_min(1.0)
                    per_sample = (loss_map * valid_mask).sum(dim=[1,2,3]) / denom
                    val_losses.extend([float(x.item()) for x in per_sample])

                    # IOU and Dice per-sample
                    probs = torch.sigmoid(out)
                    pred_bin = (probs > 0.5).float()
                    target = masks.float()
                    pred_masked = pred_bin * valid_mask
                    target_masked = target * valid_mask
                    inter = (pred_masked * target_masked).sum(dim=[1,2,3])
                    union = (pred_masked + target_masked - pred_masked * target_masked).sum(dim=[1,2,3])
                    eps = 1e-7
                    iou_per = ((inter + eps) / (union + eps)).cpu().numpy().tolist()
                    iou_vals.extend([float(x) for x in iou_per])

                    denom_d = (pred_masked + target_masked).sum(dim=[1,2,3])
                    dice_per = ((2.0 * inter + eps) / (denom_d + eps)).cpu().numpy().tolist()
                    dice_vals.extend([float(x) for x in dice_per])

            mean_bce = float(np.mean(val_losses)) if val_losses else 0.0
            mean_iou = float(np.mean(iou_vals)) if iou_vals else 0.0
            mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0

            # report mean BCE as the primary objective (minimize)
            step = (epoch // 2)
            epoch_val_loss = mean_bce
            trial.report(epoch_val_loss, step)
            # attach other metrics as user attributes for later inspection
            trial.set_user_attr(f"val_metrics_step_{step}", {"bce": mean_bce, "iou": mean_iou, "dice": mean_dice})
            print(f"Trial {trial.number} Val step {step}: BCE={mean_bce:.6f}, IOU={mean_iou:.4f}, Dice={mean_dice:.4f}")
            if trial.should_prune():
                raise optuna.TrialPruned()

    return epoch_val_loss


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
    # We minimize validation loss (lower is better)
    study = optuna.create_study(direction='minimize', pruner=SuccessiveHalvingPruner(), storage=storage, study_name=args.study_name, load_if_exists=True)

    # Distributed reservation for exact total trials when using sqlite storage.
    if storage.startswith('sqlite:///'):
        db_file = storage[len('sqlite:///'):]
        db_file = os.path.abspath(db_file)
        # ensure meta table exists
        conn = sqlite3.connect(db_file, timeout=30)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS hpo_meta(key TEXT PRIMARY KEY, val INTEGER)")
        conn.commit()

        def reserve_one(total: int) -> bool:
            # Atomically reserve a single trial slot. Returns True if reserved.
            try:
                cur.execute('BEGIN IMMEDIATE')
                cur.execute("SELECT val FROM hpo_meta WHERE key='allocated'")
                row = cur.fetchone()
                if row is None:
                    allocated = 0
                    cur.execute("INSERT OR REPLACE INTO hpo_meta(key,val) VALUES ('allocated', 0)")
                else:
                    allocated = int(row[0])
                if allocated >= total:
                    conn.rollback()
                    return False
                cur.execute("UPDATE hpo_meta SET val = val + 1 WHERE key='allocated'")
                conn.commit()
                return True
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                return False

        print(f"Coordinating distributed run: target total trials = {args.n_trials}")
        # Each worker reserves one trial at a time and executes it.
        while True:
            ok = reserve_one(args.n_trials)
            if not ok:
                break
            try:
                study.optimize(objective, n_trials=1)
            except Exception as e:
                print('Trial execution error:', e)
                # continue to next reservation
                continue
        conn.close()
    else:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    print('Best trial:', study.best_trial.number, study.best_trial.value)
    print(study.best_trial.params)


if __name__ == '__main__':
    main()
