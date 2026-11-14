"""Learnable Fourier front-end: the network itself decides which spatial frequencies to pass.

  x --FFT--> X(f) --[ K learnable transfer functions G_k(f) ]--> iFFT --> K channels

Each G_k is a log-Gabor-like filter whose *radial profile is free*: sigmoid of learnable knots,
log-spaced in period (DC is always removed). Two families:
  isotropic   G = R_k(|f|)                          "which scale"             periods 5..256 px
  oriented    G = R_s(|f|) * A(angle - theta_o)     "which scale + direction" periods 10..256 px,
              n_orient directions; even part -> dark-line response, odd part -> edge response;
              max over orientations.
The filters are defined in physical frequency (cycles / px), so the same weights apply to a training
crop and to the full 2048 x 2048 frame. A frequency-domain product is a convolution with an arbitrarily
large kernel at O(N log N) -- what filament-scale (20-100 px) structure needs. Because the pass-bands
are band-limited, responses are synthesised on a 2x / 4x decimated grid by cropping the spectrum
(exact, no aliasing) and bilinearly upsampled, which makes the stem ~10x cheaper than full-res iFFTs.
The learned radial profiles are directly plottable: they show what the model found worth filtering.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def _init_profiles(periods, knots_logf, sigma_log):
    out = []
    for p in periods:                                              # log-Gaussian bump centred at 1/p
        g = torch.exp(-(knots_logf - math.log(1 / p)) ** 2 / (2 * sigma_log ** 2)).clamp(1e-3, 1 - 1e-3)
        out.append(torch.log(g / (1 - g)))                         # inverse sigmoid
    return torch.stack(out)


class SpectralStem(nn.Module):
    ISO_DEC, ORI_DEC = 2, 4                                        # decimation of the synthesis grid

    def __init__(self, iso_periods=(12., 24., 48., 96.), ori_periods=(24., 48., 96.), n_orient=8,
                 n_knots=20, sigma_log=0.55, ang_width=0.35):
        super().__init__()
        self.n_iso, self.n_sc, self.n_orient, self.n_knots = len(iso_periods), len(ori_periods), n_orient, n_knots
        self.rng = {"iso": (math.log(1 / 256.), math.log(1 / 5.)), "ori": (math.log(1 / 256.), math.log(1 / 10.))}
        self.radial_iso = nn.Parameter(_init_profiles(iso_periods, torch.linspace(*self.rng["iso"], n_knots), sigma_log))
        self.radial_ori = nn.Parameter(_init_profiles(ori_periods, torch.linspace(*self.rng["ori"], n_knots), sigma_log))
        self.log_w = nn.Parameter(torch.full((self.n_sc,), math.log(ang_width)))       # angular width per scale
        self.out_channels = self.n_iso + 2 * self.n_sc
        self.register_buffer("scale", torch.ones(self.out_channels))                   # data-dependent init, then fixed
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        self._grid = {}

    def _freq(self, kind, h, w, d, dev):
        """Frequency geometry of the (h/d, w/d) synthesis grid, in cycles per full-resolution pixel."""
        key = (kind, h, w, dev)
        if key not in self._grid:
            lo, hi = self.rng[kind]
            fy = (torch.fft.fftfreq(h // d, device=dev) / d)[:, None]; fx = (torch.fft.rfftfreq(w // d, device=dev) / d)[None, :]
            rho = torch.sqrt(fx * fx + fy * fy)
            pos = (torch.log(rho.clamp_min(1e-6)) - lo) / (hi - lo) * (self.n_knots - 1)
            inside = ((pos >= 0) & (pos <= self.n_knots - 1)).float()
            pos = pos.clamp(0, self.n_knots - 1 - 1e-4)
            s = (d - 1) / 2                                         # sample at block centres -> aligned bilinear upsampling
            phase = torch.exp(2j * math.pi * (fy * s + fx * s))
            self._grid[key] = (pos.floor().long(), pos - pos.floor(), inside, torch.atan2(fy.expand_as(rho), fx.expand_as(rho)), phase)
        return self._grid[key]

    @staticmethod
    def _crop(X, h, w, d):
        """Low-frequency block of an rfft2 spectrum = spectrum of the d-times decimated band-limited signal."""
        hh, ww = h // d, w // d
        return torch.cat([X[:, : hh // 2, : ww // 2 + 1], X[:, h - hh // 2:, : ww // 2 + 1]], 1) / (d * d)

    def _radial(self, param, i0, t, inside):
        r = torch.sigmoid(param)
        return (r[:, i0] * (1 - t) + r[:, i0 + 1] * t) * inside

    def forward(self, x, valid):
        with torch.autocast("cuda", enabled=False):
            x = x.float(); valid = valid.float()
            B, _, h, w = x.shape
            if len(self._grid) > 4: self._grid = {}
            X = torch.fft.rfft2(x[:, 0])                                           # (B, h, w/2+1)
            # ---- isotropic band-passes
            d = self.ISO_DEC; i0, t, inside, _, ph = self._freq("iso", h, w, d, x.device)
            R = self._radial(self.radial_iso, i0, t, inside)
            iso = -torch.fft.irfft2(self._crop(X, h, w, d)[:, None] * (R * ph)[None], s=(h // d, w // d))   # dark -> positive
            iso = F.interpolate(iso, size=(h, w), mode="bilinear", align_corners=False)
            # ---- oriented filters
            d = self.ORI_DEC; i0, t, inside, theta, ph = self._freq("ori", h, w, d, x.device)
            R = self._radial(self.radial_ori, i0, t, inside)
            Xo = self._crop(X, h, w, d)[:, None]
            dth = theta[None] - (torch.arange(self.n_orient, device=x.device) * (math.pi / self.n_orient))[:, None, None]
            sgn = 1j * torch.sign(torch.cos(dth))
            outs = []
            for s in range(self.n_sc):
                G = R[s][None] * torch.exp(-torch.sin(dth) ** 2 / (2 * torch.exp(self.log_w[s]) ** 2)) * ph      # (n_orient, hh, wf)
                even = torch.fft.irfft2(Xo * G[None], s=(h // d, w // d))          # line response
                odd = torch.fft.irfft2(Xo * (sgn * G)[None], s=(h // d, w // d))   # edge response
                outs += [(-even).amax(1, keepdim=True).clamp_min(0), odd.abs().amax(1, keepdim=True)]
            ori = F.interpolate(torch.cat(outs, 1), size=(h, w), mode="bilinear", align_corners=False)
            y = torch.cat([iso, ori], 1) * valid
            if self.training and not bool(self.calibrated):                        # one-off scale calibration
                with torch.no_grad():
                    std = ((y * y).sum((0, 2, 3)) / valid.sum().clamp_min(1)).sqrt().clamp_min(1e-4)
                    self.scale.copy_(0.5 / std); self.calibrated.fill_(True)
            return y * self.scale[None, :, None, None]

    @torch.no_grad()
    def profiles(self):
        """Learned radial transfer functions for plotting / the report."""
        per = lambda k: (1 / torch.exp(torch.linspace(*self.rng[k], self.n_knots))).tolist()
        return {"iso": {"periods_px": per("iso"), "gains": torch.sigmoid(self.radial_iso).cpu().tolist()},
                "ori": {"periods_px": per("ori"), "gains": torch.sigmoid(self.radial_ori).cpu().tolist()},
                "ang_width_rad": torch.exp(self.log_w).cpu().tolist()}
