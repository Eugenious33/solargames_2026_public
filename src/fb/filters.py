"""Classical multi-scale filter bank as a fixed, differentiable torch module.

Runs on GPU inside the model (no CPU preprocessing at inference): the network sees the
normalised H-alpha frame plus hand-crafted "filament detectors" that classical solar /
vessel-segmentation pipelines rely on:

  lc{s}     local contrast  (x - mean_s) / std_s           masked (on-disk) Gaussian statistics
  sw{j}     starlet (a-trous B3-spline wavelet) detail plane j, sign flipped: dark -> positive
  ridge{s}  Lindeberg dark-ridge   s^2 * max(lambda_1, 0)   of the Gaussian Hessian
  aniso{s}  ridge minus blob part  s^2 * max(lambda_1 - max(lambda_2, 0), 0)
  frangi    multi-scale Frangi vesselness for dark tubular structures
  gabor{l}  max over orientations of an elongated even Gabor (wavelength l), dark-line phase
  gcos/gsin orientation (double angle) of the strongest Gabor response, weighted by strength
  bth{k}    black top-hat (closing - image), k x k structuring element

All filters take x: (B,1,H,W) ~N(0,1) on quiet Sun and 0 off-disk, valid: (B,1,H,W) in {0,1}.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


# ----------------------------------------------------------------------------- primitives
def _gauss1d(sigma, trunc=3.0):
    r = max(1, int(trunc * sigma + 0.5))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-t * t / (2 * sigma * sigma))
    return k / k.sum()


def _sep_conv(x, k, dilation=1):
    """Separable conv of (B,C,H,W) with a 1-D kernel, reflect padding."""
    C = x.shape[1]
    r = (len(k) // 2) * dilation
    k = k.to(x)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C, dilation=dilation)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C, dilation=dilation)
    return x


def gblur(x, sigma):
    """Gaussian blur; large sigmas are computed on a box-decimated pyramid level (cheap, accurate)."""
    f = 1
    while sigma / f > 6 and min(x.shape[-2:]) // (2 * f) >= 16:
        f *= 2
    if f == 1:
        return _sep_conv(x, _gauss1d(sigma))
    Hh, Ww = x.shape[-2:]
    ph, pw = (-Hh) % f, (-Ww) % f
    xs = F.avg_pool2d(F.pad(x, (0, pw, 0, ph), mode="reflect"), f)
    s = math.sqrt(max(sigma * sigma - (f * f - 1) / 12.0, 1e-3)) / f   # box prefilter variance
    xs = _sep_conv(xs, _gauss1d(s))
    return F.interpolate(xs, scale_factor=f, mode="bilinear", align_corners=False)[..., :Hh, :Ww]


def mblur(x, valid, sigma):
    """Normalised (masked) Gaussian mean: ignores off-disk pixels instead of bleeding zeros in."""
    return gblur(x * valid, sigma) / gblur(valid, sigma).clamp_min(1e-3)


# ----------------------------------------------------------------------------- filters
def local_contrast(x, valid, sigma):
    mu = mblur(x, valid, sigma)
    var = (mblur(x * x, valid, sigma) - mu * mu).clamp_min(1e-4)
    return (x - mu) / var.sqrt() * valid


_B3 = torch.tensor([1., 4., 6., 4., 1.]) / 16.


def starlet(x, valid, n_scales):
    """A-trous starlet transform. Returns detail planes w_1..w_n (dark structures positive)."""
    out, c = [], x
    for j in range(n_scales):
        num = _sep_conv(c * valid, _B3, dilation=2 ** j)
        den = _sep_conv(valid, _B3, dilation=2 ** j).clamp_min(1e-3)
        c_next = num / den
        out.append((c_next - c) * valid)        # = -(c - c_next): absorption features > 0
        c = c_next
    return out


def hessian_eigs(x, valid, sigma):
    """Scale-normalised eigenvalues (l1 >= l2) of the Gaussian Hessian."""
    L = mblur(x, valid, sigma)
    Lp = F.pad(L, (1, 1, 1, 1), mode="replicate")
    c = Lp[..., 1:-1, 1:-1]
    Lxx = Lp[..., 1:-1, 2:] - 2 * c + Lp[..., 1:-1, :-2]
    Lyy = Lp[..., 2:, 1:-1] - 2 * c + Lp[..., :-2, 1:-1]
    Lxy = 0.25 * (Lp[..., 2:, 2:] - Lp[..., 2:, :-2] - Lp[..., :-2, 2:] + Lp[..., :-2, :-2])
    half_tr = 0.5 * (Lxx + Lyy)
    root = (0.25 * (Lxx - Lyy) ** 2 + Lxy * Lxy + 1e-12).sqrt()
    s2 = sigma * sigma
    return s2 * (half_tr + root), s2 * (half_tr - root)


def frangi_dark(l1, l2, beta=0.5, c=0.25):
    """Frangi vesselness for dark ridges: l1 (across) large positive, |l2| (along) small."""
    rb2 = (l2 / l1.clamp_min(1e-6)) ** 2
    s2 = l1 * l1 + l2 * l2
    v = torch.exp(-rb2 / (2 * beta * beta)) * (1 - torch.exp(-s2 / (2 * c * c)))
    return torch.where(l1 > 0, v, torch.zeros_like(v))


def gabor_kernels(lam=12.0, n_orient=12, sigma_ratio=0.45, gamma=0.35, trunc=2.5):
    """Even (cosine) Gabor bank, elongated along the line (gamma < 1), zero-mean, unit positive mass.
    A dark bar of width ~lam/2 gives a strongly *negative* response."""
    sx = sigma_ratio * lam; sy = sx / gamma
    r = int(math.ceil(trunc * sy))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    ks = []
    for i in range(n_orient):
        th = math.pi * i / n_orient
        xa = xx * math.cos(th) + yy * math.sin(th)        # across the line
        ya = -xx * math.sin(th) + yy * math.cos(th)       # along the line
        env = torch.exp(-(xa * xa / (2 * sx * sx) + ya * ya / (2 * sy * sy)))
        g = env * torch.cos(2 * math.pi * xa / lam)
        g = g - env * (g.sum() / env.sum())               # remove DC
        ks.append(g / g.clamp_min(0).sum())
    return torch.stack(ks)[:, None]                       # (n_orient, 1, k, k)


def black_tophat(x, k):
    r = k // 2
    dil = F.max_pool2d(x, k, 1, r)
    clo = -F.max_pool2d(-dil, k, 1, r)
    return clo - x


# ----------------------------------------------------------------------------- module
class FilterBank(nn.Module):
    """Fixed classical front-end. `spec` is a list of channel names (see module docstring);
    'x' passes the input through. Example: ['x', 'lc32', 'sw3', 'sw4', 'ridge3', 'gabor24']."""

    GABOR_BASE = 12.0     # base wavelength in px; 24 / 48 are computed on 2x / 4x decimated levels

    def __init__(self, spec, n_orient=12):
        super().__init__()
        self.spec = list(spec)
        self.n_orient = n_orient
        self.register_buffer("gk", gabor_kernels(self.GABOR_BASE, n_orient), persistent=False)
        self.out_channels = len(self.spec)
        self._gk_fft = {}

    # -- gabor on a pyramid level: returns (max response, argmax orientation index)
    def _gabor(self, x, level):
        f = 2 ** level
        xs = (F.avg_pool2d(x, f) if f > 1 else x).float()
        h, w = xs.shape[-2:]; ks = self.gk.shape[-1]
        if (h, w, xs.device) not in self._gk_fft:              # kernel spectra for this grid (cached)
            pad = torch.zeros(self.gk.shape[0], h, w, device=xs.device); pad[:, :ks, :ks] = self.gk[:, 0]
            self._gk_fft = {(h, w, xs.device): torch.fft.rfft2(torch.roll(pad, (-(ks // 2), -(ks // 2)), (1, 2))).conj()}
        # correlation with all orientations in the Fourier domain: 79x79 kernels at the cost of a few FFTs
        r = -torch.fft.irfft2(torch.fft.rfft2(xs[:, 0])[:, None] * self._gk_fft[(h, w, xs.device)][None], s=(h, w))   # dark line -> positive
        m, idx = r.max(1, keepdim=True)
        m = m.clamp_min(0)
        if f > 1:
            m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)
            idx = F.interpolate(idx.float(), size=x.shape[-2:], mode="nearest").long()
        return m, idx

    @torch.no_grad()
    def forward(self, x, valid):
        x = x.float(); valid = valid.float()
        cache, out = {}, []

        def eig(s):
            if ("h", s) not in cache: cache[("h", s)] = hessian_eigs(x, valid, s)
            return cache[("h", s)]

        def gab(lam):
            if ("g", lam) not in cache:
                cache[("g", lam)] = self._gabor(x, int(round(math.log2(lam / self.GABOR_BASE))))
            return cache[("g", lam)]

        for name in self.spec:
            if name == "x":
                o = x
            elif name.startswith("lc"):
                o = local_contrast(x, valid, float(name[2:])).clamp(-6, 6)
            elif name.startswith("sw"):
                j = int(name[2:])
                if "sw" not in cache or len(cache["sw"]) < j: cache["sw"] = starlet(x, valid, j)
                o = cache["sw"][j - 1] * 2.0
            elif name.startswith("ridge"):
                o = eig(float(name[5:]))[0].clamp_min(0) * 4.0
            elif name.startswith("aniso"):
                l1, l2 = eig(float(name[5:]))
                o = (l1 - l2.clamp_min(0)).clamp_min(0) * 4.0
            elif name == "frangi":
                o = torch.stack([frangi_dark(*eig(s)) for s in (3.0, 4.5, 7.0, 10.0, 14.0)]).amax(0) * 3.0
            elif name.startswith("gabor"):
                o = gab(float(name[5:]))[0] * 2.0
            elif name in ("gcos", "gsin"):
                ms = torch.stack([gab(l)[0] for l in (12.0, 24.0, 48.0)])
                ix = torch.stack([gab(l)[1] for l in (12.0, 24.0, 48.0)])
                best = ms.argmax(0, keepdim=True)
                m = ms.gather(0, best)[0]; th = ix.gather(0, best)[0].float() * (math.pi / self.n_orient)
                o = m * (torch.cos(2 * th) if name == "gcos" else torch.sin(2 * th)) * 2.0
            elif name.startswith("bth"):
                o = black_tophat(gblur(x, 1.5), int(name[3:]))
            else:
                raise ValueError(f"unknown filter '{name}'")
            out.append(o * valid)
        return torch.cat(out, 1)


ALL = ["x", "lc16", "lc32", "lc64", "lc128", "sw2", "sw3", "sw4", "sw5", "sw6", "sw7",
       "ridge2", "ridge3", "ridge4.5", "ridge7", "ridge10", "ridge14", "ridge20",
       "aniso4.5", "aniso7", "aniso10", "aniso14", "frangi",
       "gabor12", "gabor24", "gabor48", "gabor96", "bth15", "bth31", "bth61"]
