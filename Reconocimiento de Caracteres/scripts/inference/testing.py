from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ResUNet de etapa 1 usado en segmentación (coincide con train_segmentation_clean.py).
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

    def __init__(self, in_ch: int = 3, base: int = 24, use_se: bool = True):
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


def stitch_tiles_to_image(pred_tiles, metas):
    # Ensambla los parches deslizantes en una sola máscara completa.
    if not pred_tiles:
        return np.array([])
    
    # Usa dimensiones de la primera meta.
    img_h = metas[0]['img_h']
    img_w = metas[0]['img_w']
    
    # Acumuladores de salida y pesos.
    output = np.zeros((img_h, img_w), dtype=np.float32)
    weights = np.zeros((img_h, img_w), dtype=np.float32)
    
    for tile, meta in zip(pred_tiles, metas):
        x, y = meta['x'], meta['y']
        crop_h, crop_w = meta['crop_h'], meta['crop_w']
        
        # Recorte válido del tile (quita padding).
        tile_crop = tile[0, :crop_h, :crop_w]
        
        # Acumula valores y pesos.
        output[y:y+crop_h, x:x+crop_w] += tile_crop
        weights[y:y+crop_h, x:x+crop_w] += 1.0
    
    # Promedia regiones solapadas.
    weights = np.maximum(weights, 1.0)
    output /= weights
    
    return output


def preprocess_image(img, target_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img_t


def save_outputs(orig_bgr, output_mask, out_original_path: Path, out_mask_path: Path):
    # Guarda la imagen original y la máscara cruda de salida.
    cv2.imwrite(str(out_original_path), orig_bgr)
    mask_uint8 = (output_mask * 255).astype(np.uint8)
    cv2.imwrite(str(out_mask_path), mask_uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=str(Path(__file__).parent.parent.parent / 'models' / 'segmentation' / 'fine_tuning_checkpoint_epoch_10.pth'))
    parser.add_argument('--testing-dir', type=str, default=str(Path(__file__).parent.parent.parent / 'test_images'))
    parser.add_argument('--out-dir', type=str, default=str(Path(__file__).parent.parent.parent / 'outputs' / 'test_predictions'))
    parser.add_argument('--target-size', type=int, nargs=2, default=[256,256])
    parser.add_argument('--stride', type=int, default=0, help='Sliding window stride in pixels; default 0 means 50 percent overlap')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--base', type=int, default=24, help='Stage 1 uses base=24')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    testing_dir = Path(args.testing_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if (args.device is None and torch.cuda.is_available()) or args.device=='cuda' else 'cpu')

    # Envoltura autocast para torch.amp o torch.cuda.amp.
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

    # Carga el checkpoint.
    if not model_path.exists():
        print('Model file not found:', model_path)
        return
    
    ck = torch.load(str(model_path), map_location='cpu', weights_only=False)
    
    # Extrae state_dict del checkpoint.
    if isinstance(ck, dict) and 'model_state' in ck:
        sd = ck['model_state']
    elif isinstance(ck, dict) and 'model_state_dict' in ck:
        sd = ck['model_state_dict']
    else:
        sd = ck

    # Infiera base según los canales de la primera conv.
    inferred_base = args.base
    for k, v in sd.items():
        if 'enc1.conv1.weight' in k:
            inferred_base = v.shape[0]
            break

    print(f'Using base={inferred_base} (Stage 1 trained with base=24)')
    
    # Construye el ResUNet de etapa 1.
    model = ResUNet(in_ch=3, base=int(inferred_base), use_se=True).to(device)
    try:
        model.load_state_dict(sd)
    except Exception as e:
        # Último recurso: cargar state_dict desde otra clave.
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
            # Stride por defecto: solape del 50%.
            if args.stride and args.stride > 0:
                stride_x = stride_y = int(args.stride)
            else:
                stride_x = max(1, tile_w // 2)
                stride_y = max(1, tile_h // 2)

            pred_tiles = []
            metas = []

            # Calcula posiciones de tiles para minimizar padding.
            y_positions = list(range(0, H, stride_y))
            x_positions = list(range(0, W, stride_x))
            
            # Ajusta últimas posiciones para no exceder bordes.
            if y_positions and (y_positions[-1] + tile_h > H):
                y_positions[-1] = max(0, H - tile_h)
            if x_positions and (x_positions[-1] + tile_w > W):
                x_positions[-1] = max(0, W - tile_w)
            
            # Elimina posiciones duplicadas.
            y_positions = sorted(set(y_positions))
            x_positions = sorted(set(x_positions))

            # Desliza ventana con posiciones ajustadas.
            for y in y_positions:
                for x in x_positions:
                    cw = min(tile_w, W - x)
                    ch = min(tile_h, H - y)
                    crop = img_bgr[y:y+ch, x:x+cw]
                    
                    # Usa padding reflectivo para evitar artefactos grises.
                    pad_right = tile_w - cw
                    pad_bottom = tile_h - ch
                    if pad_right != 0 or pad_bottom != 0:
                        crop_padded = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)
                    else:
                        crop_padded = crop
                    crop_rgb = cv2.cvtColor(crop_padded, cv2.COLOR_BGR2RGB)
                    img_t = torch.from_numpy(crop_rgb).permute(2,0,1).float() / 255.0
                    img_t = img_t.unsqueeze(0).to(device)
                    # Inferencia.
                    if device.type == 'cuda':
                        with autocast_fn(True):
                            logits = model(img_t)
                    else:
                        logits = model(img_t)
                    probs = torch.sigmoid(logits)
                    probs_np = probs.squeeze().cpu().numpy()  # (tile_h, tile_w)
                    # Guarda probabilidades completas; el pegado recorta a cw,ch.
                    pred_tiles.append(probs_np[np.newaxis, :, :])
                    metas.append({'x': x, 'y': y, 'crop_w': cw, 'crop_h': ch, 'img_w': W, 'img_h': H})

            # Stitch tiles back to original image size
            stitched = stitch_tiles_to_image(pred_tiles, metas)
            if stitched.size == 0:
                print(f'{p.name}: empty stitched output')
                continue
            
            # Save original image + network output
            out_original = out_dir / (p.stem + '_original.png')
            out_mask = out_dir / (p.stem + '_output.png')
            save_outputs(img_bgr, stitched, out_original, out_mask)
            
            # Print summary
            mean_prob = stitched.mean()
            print(f'{p.name}: mean_prob={mean_prob:.4f}, saved to {out_mask.name}')

    print('Saved predictions to', out_dir)

if __name__ == '__main__':
    main()
