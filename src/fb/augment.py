"""Augmentations for GONG H-alpha filament segmentation.

CPU (geometric, per sample):  foreground-biased crop, D4, optional free rotation + mild zoom about
                              the crop centre (chirality is not scored, so mirrors are label-safe).
GPU (photometric, per batch): applied to the normalised frame *before* the classical filter bank,
                              so every derived channel sees a physically consistent image.
  gain/offset   exposure differences between the six GONG stations
  tone curve    non-linear FITS->JPEG transfer / gamma differences
  seeing        Gaussian blur (bad seeing) or unsharp mask (good seeing)
  veil          smooth low-frequency additive field: thin cloud, residual limb darkening, vignetting
  noise         sensor / JPEG-like noise
"""
import math, numpy as np, cv2, torch, torch.nn.functional as F
from filters import gblur

H = W = 2048


def crop_pad(a, y0, x0, S):
    out = np.zeros((S, S), a.dtype)
    ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + S, H), min(x0 + S, W)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


def geo_sample(rs, arrays, cy, cx, S, free_rot):
    """arrays: list of (H,W) arrays (image float16, target uint8, valid uint8). Returns float32 (S,S) each."""
    if free_rot:
        ang, sc = rs.uniform(0, 360), math.exp(rs.uniform(-0.16, 0.16))
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, sc)
        M[0, 2] += S / 2 - cx; M[1, 2] += S / 2 - cy
        out = [cv2.warpAffine(np.asarray(a, np.float32), M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0) for a in arrays]
    else:
        y0, x0 = int(cy) - S // 2, int(cx) - S // 2
        out = [crop_pad(a, y0, x0, S).astype(np.float32) for a in arrays]
    k = rs.randint(4); fl = rs.rand() < 0.5
    out = [np.rot90(o, k) for o in out]
    if fl: out = [o[:, ::-1] for o in out]
    return [np.ascontiguousarray(o) for o in out]


def _u(B, lo, hi, dev):
    return torch.empty(B, 1, 1, 1, device=dev).uniform_(lo, hi)


def photometric(x, valid, strong):
    """x: (B,1,S,S) normalised frame (0 off-disk). Returns augmented frame, still 0 off-disk."""
    B, dev = x.shape[0], x.device
    if strong:
        # tone curve in intensity space I = 1 + s*x (quiet Sun = 1, s ~ rms contrast)
        s = 0.12
        gam = torch.exp(_u(B, -0.45, 0.45, dev)); on = (torch.rand(B, 1, 1, 1, device=dev) < 0.5).float()
        gam = on * gam + (1 - on)
        x = ((1 + s * x).clamp_min(0.05) ** gam - 1) / (s * gam)
        for b in range(B):
            r = float(torch.rand(1))
            if r < 0.30:                                       # worse seeing
                x[b:b + 1] = gblur(x[b:b + 1], float(torch.empty(1).uniform_(0.6, 1.8)))
            elif r < 0.45:                                     # better seeing
                x[b:b + 1] = x[b:b + 1] + float(torch.empty(1).uniform_(0.3, 1.0)) * (x[b:b + 1] - gblur(x[b:b + 1], 1.5))
        # veil: smooth random field, amplitude up to ~0.6 sigma
        g = torch.randn(B, 1, 5, 5, device=dev) * _u(B, 0.0, 0.6, dev) * (torch.rand(B, 1, 1, 1, device=dev) < 0.4).float()
        x = x + F.interpolate(g, size=x.shape[-2:], mode="bicubic", align_corners=True)
        x = x + torch.randn_like(x) * _u(B, 0.0, 0.15, dev) * (torch.rand(B, 1, 1, 1, device=dev) < 0.3).float()
    x = x * _u(B, 0.85, 1.15, dev) + _u(B, -0.15, 0.15, dev)
    return x * valid
