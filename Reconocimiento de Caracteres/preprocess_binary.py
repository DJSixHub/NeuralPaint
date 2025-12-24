from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import torch
import cv2
import numpy as np
from tqdm import tqdm


def process_one(args):
    p_img, rel, images_dir, masks_dir, masks_ignore_dir, out_path = args
    # load image once
    img = cv2.imread(str(p_img), cv2.IMREAD_UNCHANGED)
    if img is None:
        img_rgb = np.ones((256, 256, 3), dtype=np.uint8) * 255
    else:
        # check dimensionality/channels once
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize once using shape[:2]
        if img_rgb.shape[:2] != (256, 256):
            img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    # masks
    p_mask = masks_dir / rel
    if p_mask.exists():
        mask = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((256, 256), dtype=np.uint8)
        else:
            if mask.shape[:2] != (256, 256):
                mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.zeros((256, 256), dtype=np.uint8)
    p_ign = masks_ignore_dir / rel
    if p_ign.exists():
        ign = cv2.imread(str(p_ign), cv2.IMREAD_GRAYSCALE)
        if ign is None:
            ign = np.zeros((256, 256), dtype=np.uint8)
        else:
            if ign.shape[:2] != (256, 256):
                ign = cv2.resize(ign, (256, 256), interpolation=cv2.INTER_NEAREST)
    else:
        ign = np.zeros((256, 256), dtype=np.uint8)

    # convert to tensors uint8 (C,H,W)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
    mask_t = torch.from_numpy(mask).unsqueeze(0).contiguous()
    ign_t = torch.from_numpy(ign).unsqueeze(0).contiguous()

    # write using torch.save to produce PyTorch-compatible .pt dataset files
    torch.save({'img': img_t, 'mask': mask_t, 'ignore': ign_t}, out_path)

    # free large references in worker to avoid holding memory
    del img, img_rgb, mask, ign, img_t, mask_t, ign_t

    return rel


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset images+masks to fast uncompressed .pt files')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).parent / 'datasets')
    parser.add_argument('--out-dir', type=Path, default=None, help='Output dir for binaries (default: <data_root>/binary)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ext', type=str, default='png', help='Image extension to scan')
    args = parser.parse_args()

    data_root: Path = args.data_root
    images_dir = data_root / 'images'
    masks_dir = data_root / 'masks'
    masks_ignore_dir = data_root / 'masks_ignore'
    out_dir = args.out_dir if args.out_dir is not None else data_root / 'binary'
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(images_dir.rglob(f'*.{args.ext}')) if images_dir.exists() else []
    if not files:
        print('No images found under', images_dir)
        return

    # pre-create output subdirectories for each shard to avoid per-task mkdir overhead
    parents = set()
    for p in files:
        rel = str(p.relative_to(images_dir))
        parent = (out_dir / rel).with_suffix('.pt').parent
        parents.add(parent)
    for d in parents:
        d.mkdir(parents=True, exist_ok=True)

    # use process pool to avoid GIL for CPU-bound image IO/resize/save
    # compute full out_path once per file to avoid repeated Path operations in worker
    def iter_args(files, images_dir, masks_dir, masks_ignore_dir, out_dir):
        for p in files:
            rel = str(p.relative_to(images_dir))
            out_path = (out_dir / rel).with_suffix('.pt')
            yield (p, rel, images_dir, masks_dir, masks_ignore_dir, out_path)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for _ in tqdm(ex.map(process_one, iter_args(files, images_dir, masks_dir, masks_ignore_dir, out_dir)), total=len(files)):
            pass

    print('Preprocessing complete. Binaries written to', out_dir)

if __name__ == '__main__':
    main()
