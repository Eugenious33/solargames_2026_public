"""FilamentNet = [fixed classical filter bank] + [optional learnable Fourier stem] + segmentation network.

Input: normalised frame x (B,1,H,W) and on-disk mask. Everything, including the hand-crafted and the
learned frequency filters, runs on the GPU inside the module; inference needs no CPU preprocessing.
"""
import os
import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from filters import FilterBank
from spectral import SpectralStem

ARCHS = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus}


class FeatureMixer(nn.Module):
    """Learnable per-pixel block on top of the filter / Fourier responses, before the segmentation net.

      soft clip      t * tanh(f / t), t learnable per channel     -> tames outlier responses
      channel gate   sigmoid(g) per channel, global                -> learns which filters to drop (readable)
      pixel MLP      1x1 conv cin->h->h->k with GELU               -> a fully connected net on every pixel
      pixel head     1x1 conv k->1, own auxiliary loss             -> a filament classifier from filters alone

    Output (k mixed channels + the pixel-head logit) is concatenated with the frame and fed to the U-Net.

    mix=2 adds a depthwise k x k conv per channel before the MLP (each filter response sees its own
    neighbourhood at full resolution) and is wide (no narrow bottleneck).
    """

    def __init__(self, cin, hidden=32, k=8, spatial=0):
        super().__init__()
        self.log_t = nn.Parameter(torch.full((1, cin, 1, 1), 1.4))           # t = e^1.4 ~ 4
        self.gate = nn.Parameter(torch.full((1, cin, 1, 1), 2.0))            # sigmoid(2) ~ 0.88 at start
        layers = []
        if spatial:                                                          # depthwise: x2 channels, own kernel per channel
            layers += [nn.Conv2d(cin, 2 * cin, spatial, padding=spatial // 2, groups=cin), nn.GELU()]
            c0 = 2 * cin
        else:
            c0 = cin
        layers += [nn.Conv2d(c0, hidden, 1), nn.GELU(), nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, k, 1)]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Conv2d(k, 1, 1)
        self.out_channels = k + 1

    def forward(self, f):
        t = self.log_t.exp()
        f = t * torch.tanh(f / t) * torch.sigmoid(self.gate)
        z = self.mlp(f)
        aux = self.head(F.gelu(z))
        return torch.cat([z, aux], 1), aux


class FilamentNet(nn.Module):
    def __init__(self, spec, enc="resnet34", arch="unet", lsf=False, weights="imagenet", mix=False):
        super().__init__()
        self.fb = FilterBank(spec)
        self.stem = SpectralStem() if lsf else None
        self.spec = list(spec)
        nf = len(spec) + (self.stem.out_channels if lsf else 0)
        self.mix = int(mix)
        self.mixer = None
        assert self.spec[0] == "x" or not mix, "mix needs spec starting with x"
        if self.mix == 1:                                       # frame bypasses the mixer, filters / Fourier go through it
            assert nf > 1, "mix 1 needs filter / Fourier channels"
            self.mixer = FeatureMixer(nf - 1)
            nf = 1 + self.mixer.out_channels
        elif self.mix == 2:                                     # all channels incl. frame, depthwise + wide MLP
            self.mixer = FeatureMixer(nf, hidden=64, k=16, spatial=5)
            nf = 1 + self.mixer.out_channels
        self.last_aux = None                                    # pixel-head logits of the last forward (train-time aux loss)
        self.net = ARCHS[arch](enc, encoder_weights=weights, in_channels=nf, classes=1)
        if self.mix == 2:                                       # full-resolution skip: smp's decoder never sees stride-1 features
            dc = self.net.segmentation_head[0].in_channels
            self.head = nn.Sequential(nn.Conv2d(dc + nf, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))

    def channel_names(self):
        names = list(self.spec)
        if self.stem is not None:
            names += [f"lsf_iso{i}" for i in range(self.stem.n_iso)] + [f"lsf_even{i}" for i in range(self.stem.n_sc)] \
                     + [f"lsf_odd{i}" for i in range(self.stem.n_sc)]
        return names

    def features(self, x, valid, margin=0):
        f = self.fb(x, valid)                                   # fixed filters, fp32, no grad
        if self.stem is not None:
            f = torch.cat([f, self.stem(x, valid)], 1)          # learnable filters, fp32, with grad
        if margin:
            f = f[..., margin:-margin, margin:-margin]          # drop the context border used by the filters
        if self.mixer is not None:
            with torch.autocast("cuda", dtype=torch.float16):
                m, self.last_aux = self.mixer(f[:, 1:] if self.mix == 1 else f)
            m = m.float()
            f = torch.cat([f[:, :1], m], 1)
        return f.contiguous(memory_format=torch.channels_last)

    def forward(self, x, valid, margin=0):
        f = self.features(x, valid, margin)
        # SOLAR_AMP=bf16: DaViT's channel attention sums over every token of the frame and overflows fp16 on a full
        # 2048 frame (NaN maps, run dvt_softs_f1); bf16 has the fp32 range. Default fp16 keeps every other run as it was.
        with torch.autocast("cuda", dtype=torch.bfloat16 if os.environ.get("SOLAR_AMP") == "bf16" else torch.float16):
            if self.mix != 2:
                return self.net(f).float()
            d = self.net.decoder(self.net.encoder(f))
            return self.head(torch.cat([d, f.to(d.dtype)], 1)).float()


def build(args, weights="imagenet"):
    """args: dict with spec (comma string), enc, and optionally arch / lsf / mix."""
    return FilamentNet(args["spec"].split(","), args["enc"], args.get("arch", "unet"), bool(args.get("lsf", 0)), weights,
                       int(args.get("mix", 0)))


def load(path, dev="cuda"):
    """Load an EMA checkpoint written by train.py (also accepts the first bare-U-Net checkpoints)."""
    ck = torch.load(path, map_location=dev)
    net = build(ck["args"], weights=None).to(dev).to(memory_format=torch.channels_last).eval()
    sd = ck["model"]
    if not any(k.startswith("net.") for k in sd): sd = {"net." + k: v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net, ck["args"]
