from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

# Import stitch_tiles_to_image from the training script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from train_segmentation import stitch_tiles_to_image

# UNet matching the training script `train_segmentation.py`
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


class FinalUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 24):
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


def preprocess_image(img, target_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img_t


def save_mask_and_overlay(orig_bgr, mask_bin, out_mask_path: Path, out_overlay_path: Path):
    # mask_bin: HxW numpy uint8 0/255
    mask_uint8 = (mask_bin.astype(np.uint8))
    cv2.imwrite(str(out_mask_path), mask_uint8)
    # overlay red mask on original (resized to mask shape)
    h,w = mask_uint8.shape
    orig_resized = cv2.resize(orig_bgr, (w,h), interpolation=cv2.INTER_AREA)
    overlay = orig_resized.copy()
    # create red overlay where mask==255
    overlay[mask_uint8==255] = (0,0,255)
    mixed = cv2.addWeighted(orig_resized, 0.7, overlay, 0.3, 0)
    cv2.imwrite(str(out_overlay_path), mixed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=str(Path(__file__).parent.parent.parent / 'models' / 'segmentation' / 'checkpoint_epoch_70.pth'))
    parser.add_argument('--testing-dir', type=str, default=str(Path(__file__).parent.parent.parent / 'test_images'))
    parser.add_argument('--out-dir', type=str, default=str(Path(__file__).parent.parent.parent / 'outputs' / 'test_predictions'))
    parser.add_argument('--target-size', type=int, nargs=2, default=[256,256])
    parser.add_argument('--stride', type=int, default=0, help='Sliding window stride in pixels; default 0 means 50 percent overlap')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--base', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    testing_dir = Path(args.testing_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if (args.device is None and torch.cuda.is_available()) or args.device=='cuda' else 'cpu')

    # autocast wrapper compatible with torch.amp or torch.cuda.amp
    if device.type == 'cuda' and hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        def autocast_fn(enabled):
            return torch.amp.autocast(device_type='cuda', enabled=enabled)
    else:
        try:
            def autocast_fn(enabled):
                return torch.cuda.amp.autocast(enabled=enabled)
        except Exception:
            from contextlib import nullcontext
            def autocast_fn(enabled):
                return nullcontext()

    # load checkpoint first so we can infer `base` (avoid shape mismatches)
    if not model_path.exists():
        print('Model file not found:', model_path)
        return
    # allow loading legacy pickles/checkpoints produced locally
    ck = torch.load(str(model_path), map_location='cpu', weights_only=False)
    # extract state_dict whether checkpoint contains it under different keys
    if isinstance(ck, dict) and ('model_state' in ck or 'model_state_dict' in ck):
        sd = ck.get('model_state', ck.get('model_state_dict'))
    else:
        sd = ck

    # try to infer `base` from the first conv weight in enc1
    inferred_base = None
    for k, v in sd.items():
        if k.endswith('enc1.net.0.weight') or k.endswith('enc1.net.0.weight'):
            inferred_base = v.shape[0]
            break
    # fallback: look for outc weight and infer base from its input channels
    if inferred_base is None:
        for k, v in sd.items():
            if k.endswith('outc.weight'):
                # outc.weight shape: [1, base, 1, 1]
                inferred_base = v.shape[1]
                break
    if inferred_base is None:
        inferred_base = args.base

    # build model with inferred base
    model = FinalUNet(in_ch=3, base=int(inferred_base)).to(device)
    try:
        model.load_state_dict(sd)
    except Exception as e:
        # last resort: attempt loading the full checkpoint dict if it contains model_state under another key
        if isinstance(ck, dict) and 'state_dict' in ck:
            model.load_state_dict(ck['state_dict'])
        else:
            raise

    model.eval()

    img_paths = sorted([p for p in testing_dir.glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg')])
    if not img_paths:
        print('No images found in', testing_dir)
        return

    with torch.no_grad():
        for p in tqdm(img_paths, desc='Predict'):
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            H, W = img_bgr.shape[:2]
            tile_w, tile_h = tuple(args.target_size)
            # default stride: 50% overlap
            if args.stride and args.stride > 0:
                stride_x = stride_y = int(args.stride)
            else:
                stride_x = max(1, tile_w // 2)
                stride_y = max(1, tile_h // 2)

            pred_tiles = []
            metas = []

            # iterate sliding window to fully cover image with given stride
            for y in range(0, H, stride_y):
                for x in range(0, W, stride_x):
                    cw = min(tile_w, W - x)
                    ch = min(tile_h, H - y)
                    crop = img_bgr[y:y+ch, x:x+cw]
                    # pad crop on right/bottom to tile size (no rescaling)
                    pad_right = tile_w - cw
                    pad_bottom = tile_h - ch
                    if pad_right != 0 or pad_bottom != 0:
                        crop_padded = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(255,255,255))
                    else:
                        crop_padded = crop
                    crop_rgb = cv2.cvtColor(crop_padded, cv2.COLOR_BGR2RGB)
                    img_t = torch.from_numpy(crop_rgb).permute(2,0,1).float() / 255.0
                    img_t = img_t.unsqueeze(0).to(device)
                    # forward
                    if device.type == 'cuda':
                        with autocast_fn(True):
                            logits = model(img_t)
                    else:
                        logits = model(img_t)
                    probs = torch.sigmoid(logits)
                    probs_np = probs.squeeze().cpu().numpy()  # (tile_h, tile_w)
                    # store full tile probabilities; stitching will crop to cw,ch
                    pred_tiles.append(probs_np[np.newaxis, :, :])
                    metas.append({'x': x, 'y': y, 'crop_w': cw, 'crop_h': ch, 'img_w': W, 'img_h': H})

            # stitch and threshold
            stitched = stitch_tiles_to_image(pred_tiles, metas)
            if stitched.size == 0:
                print(f'{p.name}: empty stitched output')
                continue
            mask_bin = (stitched > args.threshold).astype(np.uint8) * 255

            out_mask = out_dir / (p.stem + '_mask.png')
            out_overlay = out_dir / (p.stem + '_overlay.png')
            save_mask_and_overlay(img_bgr, mask_bin, out_mask, out_overlay)
            # print summary
            print(f'{p.name}: mean_prob={stitched.mean():.4f} pred_coverage={(mask_bin.sum()/255.0)/(mask_bin.size):.4f}')

    print('Saved predictions to', out_dir)

if __name__ == '__main__':
    main()
