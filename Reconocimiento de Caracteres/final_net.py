from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import cv2

# performance
torch.backends.cudnn.benchmark = True

# Final model defaults (hyperparameters fixed)
DEFAULT_BASE = 24
DEFAULT_TARGET_SIZE = (256, 256)
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 32
DEFAULT_VAL_FRAC = 0.3
DEFAULT_LR = 0.0006463504999789425
DEFAULT_WD = 1.3408008751202187e-08
# Learning rate used specifically for fine-tuning (not exposed as a CLI flag)
FINE_TUNE_LR = DEFAULT_LR * 0.1
# Early stopping / scheduler defaults
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 0.0005
LR_PLATEAU_PATIENCE = 3
LR_PLATEAU_FACTOR = 0.5


class SyntheticGlyphsDataset(Dataset):
    """Simplified dataset: each item is a full 256x256 image with its mask and optional ignore mask.

    Expects `images/`, `masks/`, `masks_ignore/` under `root` and that each file in `files`
    is a relative path to a PNG of size 256x256. The dataset will load files on-the-fly
    to avoid keeping large state in memory.
    """
    def __init__(self, root: Path, files: List[str], target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE):
        self.root = Path(root)
        self.images_dir = self.root / 'images'
        self.masks_dir = self.root / 'masks'
        self.masks_ignore_dir = self.root / 'masks_ignore'
        self.files = list(files)
        self.tile_w, self.tile_h = target_size
        # optional fast binary cache: one .pt file per sample with tensors {'img','mask','ignore'}
        self.binary_dir = self.root / 'binary'
        # Force binary cache ON — dataset will require .pt preprocessed files
        self.use_binary = True

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        # prefer fast preprocessed binary (.pt) if available
        if self.use_binary:
            p_bin = (self.binary_dir / fname).with_suffix('.pt')
            if not p_bin.exists():
                raise FileNotFoundError(f"Binary cache enabled but missing file: {p_bin}. Run the preprocessor to create datasets/binary/*.pt")
            # PyTorch 2.6+ defaults to weights_only=True; these .pt files may
            # be written either with `torch.save` or with plain `pickle.dump`.
            # Try `torch.load(..., weights_only=False)` first (safe for local
            # trusted files), and fall back to `pickle.load` if the file was
            # written with the preprocessing script that uses `pickle.dump`.
            try:
                data = torch.load(str(p_bin), map_location='cpu', weights_only=False)
            except Exception:
                import warnings, pickle
                warnings.warn(f"torch.load failed for {p_bin!s}; falling back to pickle.load()")
                with open(p_bin, 'rb') as _f:
                    data = pickle.load(_f)
            img_t = data.get('img')
            mask_t = data.get('mask')
            ign_t = data.get('ignore')
            # ensure tensors are float in 0..1
            if isinstance(img_t, np.ndarray):
                img_t = torch.from_numpy(img_t)
            if isinstance(mask_t, np.ndarray):
                mask_t = torch.from_numpy(mask_t)
            if isinstance(ign_t, np.ndarray):
                ign_t = torch.from_numpy(ign_t)
            # expected shapes: img (3,H,W) uint8, mask (1,H,W) uint8
            img = img_t.float() / 255.0
            mask_bin = (mask_t.float() / 255.0) > 0.5
            ign_bin = (ign_t.float() / 255.0) > 0.5
            return {'image': img, 'mask': mask_bin.float(), 'ignore': ign_bin.float(), 'meta': {'file': fname, 'img_w': int(self.tile_w), 'img_h': int(self.tile_h)}}
        p_img = self.images_dir / fname
        img = cv2.imread(str(p_img), cv2.IMREAD_UNCHANGED)
        if img is None:
            img_rgb = np.ones((self.tile_h, self.tile_h, 3), dtype=np.uint8) * 255
        else:
            # assume dataset images are already 256x256 and RGB; convert channels only
            try:
                if img.ndim == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        p_mask = self.masks_dir / fname
        if p_mask.exists():
            mask = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        p_ign = self.masks_ignore_dir / fname
        if p_ign.exists():
            ign = cv2.imread(str(p_ign), cv2.IMREAD_GRAYSCALE)
            if ign is None:
                ign = np.zeros((h, w), dtype=np.uint8)
        else:
            ign = np.zeros((h, w), dtype=np.uint8)

        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        mask_bin = (mask_t > 0.5).float()
        ign_t = torch.from_numpy(ign).unsqueeze(0).float() / 255.0
        ign_bin = (ign_t > 0.5).float()

        meta = {'file': fname, 'img_w': w, 'img_h': h}
        return {'image': img_t, 'mask': mask_bin, 'ignore': ign_bin, 'meta': meta}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Strictly Conv -> ReLU -> Conv -> ReLU (no BatchNorm, no Dropout)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FinalUNet(nn.Module):
    """U-Net de 3 niveles (encoder: base -> 2*base -> 4*base, decoder mirrored).

    Uses ConvTranspose2d for learnable upsampling and explicit skip connections.
    No BatchNorm, no Dropout.
    """
    def __init__(self, in_ch: int = 3, base: int = DEFAULT_BASE):
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
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)

        u2 = self.up2(e3)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)

        u1 = self.up1(d2)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.outc(d1)


def iou_metric_tensor(pred, target, ignore_mask=None, eps=1e-7):
    pred_bin = (pred > 0.5).float()
    target = target.float()
    if ignore_mask is not None:
        mask = (ignore_mask < 0.5).float()
        pred_bin = pred_bin * mask
        target = target * mask
    inter = (pred_bin * target).sum(dim=[1, 2, 3])
    union = (pred_bin + target - pred_bin * target).sum(dim=[1, 2, 3])
    return ((inter + eps) / (union + eps))


def stitch_tiles_to_image(pred_tiles: list, metas: list) -> np.ndarray:
    """Reconstruct full-size mask from list of predicted tiles and corresponding metas.

    pred_tiles: list of numpy arrays shape (H,W) or (1,H,W) with values in [0,1]
    metas: list of meta dicts matching dataset.item()['meta']

    Returns full-size numpy array (img_h, img_w) of stitched probabilities.
    """
    if not metas:
        return np.array([])
    # If metas do not contain tile coords (x,y), assume each pred is a full-image prediction
    img_w = metas[0].get('img_w')
    img_h = metas[0].get('img_h')
    if 'x' not in metas[0]:
        # average multiple full-image predictions if present
        acc = None
        cnt = 0
        for pred in pred_tiles:
            if isinstance(pred, torch.Tensor):
                p = pred.detach().cpu().numpy()
            else:
                p = np.array(pred)
            if p.ndim == 3:
                p = p.squeeze(0)
            if acc is None:
                acc = np.zeros_like(p, dtype=np.float32)
            acc += p.astype(np.float32)
            cnt += 1
        if acc is None:
            return np.array([])
        out = acc / max(1, cnt)
        return out

    # legacy tile-based stitching (kept for compatibility)
    canvas = np.zeros((img_h, img_w), dtype=np.float32)
    count = np.zeros((img_h, img_w), dtype=np.int32)

    for pred, m in zip(pred_tiles, metas):
        if isinstance(pred, torch.Tensor):
            p = pred.detach().cpu().numpy()
        else:
            p = np.array(pred)
        if p.ndim == 3:
            p = p.squeeze(0)
        th, tw = p.shape[:2]
        x, y = m['x'], m['y']
        cw, ch = m['crop_w'], m['crop_h']
        canvas[y:y + ch, x:x + cw] += p[0:ch, 0:cw]
        count[y:y + ch, x:x + cw] += 1

    mask = count > 0
    out = np.zeros_like(canvas)
    out[mask] = canvas[mask] / count[mask]
    return out


def train_and_evaluate(
    dataset_root: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    val_frac: float = DEFAULT_VAL_FRAC,
    num_workers: int = 4,
    resume: str = None,
    base: int = DEFAULT_BASE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WD,
    fine_tune_sizes: List[int] = None,
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    root = Path(dataset_root)
    images_dir = root / 'images'
    # find images recursively to support sharded subfolders (images/0000/...)
    files = sorted(images_dir.rglob('*.png')) if images_dir.exists() else []
    # keep relative paths so SyntheticGlyphsDataset can locate files inside subfolders
    files = [str(p.relative_to(images_dir)) for p in files]
    # If fine-tune sizes provided, filter files using metadata and add 30% random other images
    if fine_tune_sizes:
        import json
        meta_p = root / 'metadata.jsonl'
        selected = set()
        if meta_p.exists():
            with open(meta_p, 'r', encoding='utf8') as mf:
                for line in mf:
                    try:
                        m = json.loads(line)
                    except Exception:
                        continue
                    rel = m.get('file') or m.get('path') or m.get('relpath')
                    if not rel:
                        continue
                    fs = m.get('font_size') or m.get('fontSize') or m.get('size')
                    try:
                        if fs is not None and int(fs) in fine_tune_sizes:
                            # normalize path format
                            selected.add(str(Path(rel)))
                    except Exception:
                        continue
        # map selected to actual files paths (relative)
        selected = {s for s in selected if s in files}
        other = [f for f in files if f not in selected]
        # sample 30% of other images
        add_count = int(len(other) * 0.3)
        random.shuffle(other)
        added = set(other[:add_count])
        files = list(selected.union(added))
        if not files:
            raise ValueError(f'Fine-tune requested for sizes {fine_tune_sizes} but no matching files found in metadata under {meta_p}')
    if not files:
        raise ValueError(f"No images found under {images_dir}. Check --data-root and image extensions (expected .png)")
    random.seed(42)
    random.shuffle(files)
    n_val = int(len(files) * val_frac)
    train_files = files[n_val:]
    val_files = files[:n_val]

    train_ds = SyntheticGlyphsDataset(root, train_files, target_size=target_size)
    val_ds = SyntheticGlyphsDataset(root, val_files, target_size=target_size)

    # On Windows multiprocessing with DataLoader can cause high memory and deadlocks.
    # Force safe defaults on Windows; otherwise enable pin_memory only for CUDA.
    import platform
    # Force conservative Windows-friendly DataLoader settings
    if platform.system() == 'Windows':
        num_workers = 2
    # enforce requested flags
    pin = True
    persistent_workers_flag = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers_flag,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers_flag,
    )

    model = FinalUNet(in_ch=3, base=base).to(device)
    opt = None
    scheduler = None
    # we'll compute BCE per-pixel and mask out ignored pixels (reduction='none')
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    # Use CUDA GradScaler when CUDA is available; else keep None
    if device.type == 'cuda':
        # prefer new API torch.amp.GradScaler(device_type='cuda') but fall back for older torch
        try:
            scaler = torch.amp.GradScaler(device_type='cuda')
        except Exception:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                scaler = torch.cuda.amp.GradScaler()
        # provide a compatibility wrapper for autocast that uses the new API when available
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            def _autocast(enabled):
                return torch.amp.autocast(device_type='cuda', enabled=enabled)
        else:
            def _autocast(enabled):
                return torch.cuda.amp.autocast(enabled=enabled)
        autocast_fn = _autocast
    else:
        scaler = None
        # no-op autocast on CPU
        from contextlib import nullcontext
        def autocast_fn(enabled):
            return nullcontext()
    # `torch.compile()` disabled for Windows environments.

    start_epoch = 0
    best_val_loss = float('inf')
    out_dir = Path(__file__).parent / 'visualizaciones'
    out_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        # resume checkpoints are produced by this script; ensure legacy
        # pickles are accepted by specifying weights_only=False when needed.
        ck = torch.load(resume, map_location=device, weights_only=False)
        # load model weights (checkpoint may be a state-dict or full ck)
        try:
            model.load_state_dict(ck.get('model_state', ck))
        except Exception:
            model.load_state_dict(ck)
        # If not fine-tuning, restore optimizer state as well
        if not fine_tune_sizes:
            try:
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                opt.load_state_dict(ck.get('opt_state', {}))
            except Exception:
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(opt, mode='min', factor=LR_PLATEAU_FACTOR, patience=LR_PLATEAU_PATIENCE, min_lr=1e-9)
        start_epoch = ck.get('epoch', 0) + 1
        best_val_loss = ck.get('best_val_loss', float('inf'))

    # If fine-tuning requested, freeze encoder and create optimizer for decoder-only
    if fine_tune_sizes:
        for p in model.enc1.parameters():
            p.requires_grad = False
        for p in model.enc2.parameters():
            p.requires_grad = False
        for p in model.enc3.parameters():
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError('No trainable parameters found for fine-tuning (all parameters frozen).')
        # use fixed fine-tune learning rate (not exposed to CLI)
        opt = torch.optim.Adam(trainable, lr=FINE_TUNE_LR, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=LR_PLATEAU_FACTOR, patience=LR_PLATEAU_PATIENCE, min_lr=1e-9)
    else:
        if opt is None:
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(opt, mode='min', factor=LR_PLATEAU_FACTOR, patience=LR_PLATEAU_PATIENCE, min_lr=1e-9)

    train_losses = []
    val_losses = []
    val_ious = []
    # store last validation results when validating every N epochs
    last_val_loss = None
    last_val_iou = None
    last_val_ious_list = []

    epochs_no_improve = 0
    # totals for final confusion matrix
    total_TP = total_FP = total_FN = total_TN = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} Train')
        for batch in pbar:
            imgs = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            ignores = batch.get('ignore', torch.zeros_like(masks)).to(device, non_blocking=True)
            opt.zero_grad()
            with autocast_fn(scaler is not None):
                preds = model(imgs)
                # per-pixel BCE
                loss_map = loss_fn(preds, masks)
                valid_mask = (ignores < 0.5).float()
                denom = valid_mask.sum(dim=[1, 2, 3]).clamp_min(1.0)
                per_sample = (loss_map * valid_mask).sum(dim=[1, 2, 3]) / denom
                loss = per_sample.mean()
            # backwards/step with or without GradScaler
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

        # validation: run every 2 epochs to save time
        validation_ran = False
        epoch_val_loss = last_val_loss if last_val_loss is not None else 0.0
        epoch_val_iou = last_val_iou if last_val_iou is not None else 0.0
        if (epoch + 1) % 2 == 0 and len(val_ds) > 0:
            model.eval()
            # accumulate scalar validation totals (no storing per-sample preds)
            sum_val_loss = 0.0
            cnt_val = 0
            sum_val_iou = 0.0
            cnt_iou = 0
            # confusion matrix calculation removed from per-validation step to avoid
            # penalizing training with frequent syncs. Final confusion will be
            # computed once after training.
            with torch.no_grad():
                pbarv = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} Val')
                for vb in pbarv:
                    imgs = vb['image'].to(device, non_blocking=True)
                    masks = vb['mask'].to(device, non_blocking=True)
                    ignores = vb.get('ignore', torch.zeros_like(masks)).to(device, non_blocking=True)
                    metas = vb.get('meta', None)
                    with autocast_fn(scaler is not None):
                        out = model(imgs)
                    # compute masked BCE loss same as training
                    loss_map = loss_fn(out, masks)
                    valid_mask = (ignores < 0.5).float()
                    denom = valid_mask.sum(dim=[1, 2, 3]).clamp_min(1.0)
                    per_sample = (loss_map * valid_mask).sum(dim=[1, 2, 3]) / denom
                    # accumulate scalar losses
                    sum_val_loss += float(per_sample.sum().item())
                    cnt_val += per_sample.size(0)

                    probs = torch.sigmoid(out)  # keep on device
                    # compute IoU and confusion counts on device
                    pred_bin = (probs > 0.5).float()
                    target = masks.float()
                    vm = (valid_mask > 0.5).float()
                    inter = (pred_bin * target * vm).sum(dim=[1, 2, 3])
                    union = ((pred_bin + target - pred_bin * target) * vm).sum(dim=[1, 2, 3])
                    ious = ((inter + 1e-7) / (union + 1e-7))
                    sum_val_iou += float(ious.sum().item())
                    cnt_iou += ious.size(0)

                    # finalize validation scalars from accumulated sums
            epoch_val_loss = float(sum_val_loss / cnt_val) if cnt_val > 0 else 0.0
            val_losses.append(epoch_val_loss)
            epoch_val_iou = float(sum_val_iou / cnt_iou) if cnt_iou > 0 else 0.0
            val_ious.append(epoch_val_iou)
            last_val_loss = epoch_val_loss
            last_val_iou = epoch_val_iou
                    # (confusion matrix accumulation removed from validation loop)
            validation_ran = True
            # scheduler step on validation loss
            try:
                scheduler.step(epoch_val_loss)
            except Exception:
                pass

            # (removed) avoid explicit CUDA cache clearing to prevent extra syncs

        print(f'Epoch {epoch+1} — train_loss={epoch_train_loss:.4f} val_loss={epoch_val_loss:.4f} val_iou={epoch_val_iou:.4f}')

        # Early stopping logic based on validation loss (lower is better)
        if validation_ran:
            if epoch_val_loss + EARLY_STOPPING_DELTA < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = epoch_val_loss
                # save best model
                torch.save(model.state_dict(), out_dir / 'final_model.pth')
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} validations (patience reached).')
                break

        # checkpoint: only save when validation actually ran this epoch
        ck = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'scaler_state': scaler.state_dict() if scaler is not None else {},
            'best_val_loss': best_val_loss,
        }
        if validation_ran:
            torch.save(ck, out_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # compute final confusion matrix once on the full validation set
    total_TP = total_FP = total_FN = total_TN = 0
    if len(val_ds) > 0:
        model.eval()
        with torch.no_grad():
            pbar_final = tqdm(val_loader, desc='Final validation (confusion)')
            for vb in pbar_final:
                imgs = vb['image'].to(device, non_blocking=True)
                masks = vb['mask'].to(device, non_blocking=True)
                ignores = vb.get('ignore', torch.zeros_like(masks)).to(device, non_blocking=True)
                vm = (ignores < 0.5).float()
                with autocast_fn(scaler is not None):
                    out = model(imgs)
                probs = torch.sigmoid(out)
                pred_bin = (probs > 0.5).float()
                target = masks.float()
                pred_bool = pred_bin.bool()
                targ_bool = target.bool()
                vm_bool = vm.bool()
                total_TP += int(((pred_bool & targ_bool) & vm_bool).sum().item())
                total_FP += int(((pred_bool & (~targ_bool)) & vm_bool).sum().item())
                total_FN += int((((~pred_bool) & targ_bool) & vm_bool).sum().item())
                total_TN += int((((~pred_bool) & (~targ_bool)) & vm_bool).sum().item())

    cm = np.array([[total_TN, total_FP], [total_FN, total_TP]], dtype=np.int64)

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    for (j, i), val in np.ndenumerate(cm):
        ax.text(i, j, f'{val}', ha='center', va='center', color='black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1']); ax.set_yticklabels(['True 0', 'True 1'])
    ax.set_title('Confusion matrix (final validation)')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrix_final.png')

    # loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='train_loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(out_dir / 'loss_curve.png')

    # validation IoU curve
    plt.figure()
    plt.plot(range(1, len(val_ious) + 1), val_ious, label='val_iou')
    plt.xlabel('Epoch'); plt.ylabel('Validation IoU'); plt.legend()
    plt.savefig(out_dir / 'val_iou_curve.png')

    # final metrics
    print('Best val loss:', best_val_loss)
    print('Mean validation IoU:', float(np.mean(val_ious)) if val_ious else 0.0)
    print('Saved visuals in', out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=str(Path(__file__).parent / 'datasets'))
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--fine_tune', type=int, nargs='+', default=None, help='List of font sizes to fine-tune on (e.g. --fine_tune 8 10)')
    args = parser.parse_args()
    train_and_evaluate(
        dataset_root=args.data_root,
        epochs=args.epochs,
        batch_size=DEFAULT_BATCH,
        target_size=DEFAULT_TARGET_SIZE,
        val_frac=DEFAULT_VAL_FRAC,
        num_workers=args.num_workers,
        resume=args.resume,
        base=DEFAULT_BASE,
        lr=DEFAULT_LR,
        fine_tune_sizes=args.fine_tune,
        weight_decay=DEFAULT_WD,
    )
