"""Solar-disk geometry and photometric normalisation for GONG H-alpha frames (2048x2048, 8-bit).

fit_disk   : limb = steepest radial intensity drop along rays -> robust least-squares circle (cx, cy, R)
flat_field : divide by the radial median centre-to-limb profile (removes limb darkening), then
             robust z-score inside the disk. Output is ~N(0,1) on quiet Sun, exactly 0 off-disk,
             so linear filters see no artificial edge at the limb.
"""
import numpy as np, cv2
from scipy import ndimage as ndi

H = W = 2048
VALID_FRAC = 0.985          # pixels with r < VALID_FRAC * R are treated as on-disk


def _circle_lsq(x, y):
    """Kasa algebraic circle fit."""
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    (a, c, d), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a / 2, c / 2
    return cx, cy, np.sqrt(d + cx * cx + cy * cy)


def fit_disk(img, n_rays=720, r_lo=780, r_hi=1010):
    """Return (cx, cy, R) of the solar limb. img: uint8/float (H, W)."""
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    cx, cy = W / 2, H / 2
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    for _ in range(2):                                   # second pass re-centres the rays
        rr = np.arange(r_lo, r_hi, 0.5, dtype=np.float32)
        xs = cx + np.cos(th)[:, None] * rr[None]
        ys = cy + np.sin(th)[:, None] * rr[None]
        prof = cv2.remap(g, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
        prof = ndi.gaussian_filter1d(prof, 2.0, axis=1)
        d = prof[:, 4:] - prof[:, :-4]                   # radial derivative over 2 px
        k = d.argmin(1)                                  # steepest drop = limb
        r_l = rr[k + 2]
        px, py = cx + np.cos(th) * r_l, cy + np.sin(th) * r_l
        ok = (px > 2) & (px < W - 3) & (py > 2) & (py < H - 3) & (d.min(1) < -3)
        for _ in range(4):                               # iterative outlier rejection
            fx, fy, fr = _circle_lsq(px[ok], py[ok])
            res = np.abs(np.hypot(px - fx, py - fy) - fr)
            ok2 = ok & (res < max(2.0, 2.5 * np.median(res[ok])))
            if ok2.sum() < 60: break
            ok = ok2
        cx, cy, R = fx, fy, fr
    if not (820 < R < 980 and abs(cx - W / 2) < 120 and abs(cy - H / 2) < 120):
        cx, cy, R = W / 2, H / 2, 900.0                  # implausible fit -> nominal GONG geometry
    return float(cx), float(cy), float(R)


def radius_map(cx, cy):
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    return np.hypot(xx - cx, yy - cy)


def flat_field(img, geom, nbins=180):
    """Limb-darkening removal + robust z-score. Returns (xn float32 (H,W), valid bool (H,W))."""
    cx, cy, R = geom
    img = img.astype(np.float32)
    r = radius_map(cx, cy)
    valid = r < VALID_FRAC * R
    b = np.minimum((r / R * nbins).astype(np.int32), nbins)          # bin `nbins` = off-disk
    med = ndi.median(img, labels=b, index=np.arange(nbins))
    med = ndi.median_filter(med, 5, mode="nearest")                   # smooth the profile itself
    clv = np.interp(r, (np.arange(nbins) + 0.5) * R / nbins, med).astype(np.float32)
    flat = img / np.maximum(clv, 1.0)
    v = flat[valid]
    p16, p50, p84 = np.percentile(v[::7], [16, 50, 84])
    xn = (flat - p50) / max(0.5 * (p84 - p16), 1e-3)
    xn = np.clip(xn, -8, 8)
    xn[~valid] = 0
    return xn.astype(np.float32), valid


def zscore_disk(img):
    """Teacher-style normalisation (z-score over img>15), kept as the A/B baseline."""
    img = img.astype(np.float32)
    disk = img > 15
    x = (img - img[disk].mean()) / (img[disk].std() + 1e-6)
    x[~disk] = 0
    return x, disk
