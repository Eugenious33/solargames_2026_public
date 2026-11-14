"""Figures for the demo notebook and the report: frames, annotator records, probability maps, predicted masks.

Everything returns a matplotlib Figure; nothing here is used by training or inference.
"""
import csv, json, numpy as np, cv2
import matplotlib
import matplotlib.pyplot as plt
from pycocotools import mask as mu
from data import DATA, CACHE, H, W, load_sets, poly_masks, open_cache
import pq as PQ

COLORS = [(0.20, 0.85, 0.35), (0.25, 0.60, 1.00), (1.00, 0.75, 0.10), (1.00, 0.30, 0.30)]
# outline colours that stay visible on a gray solar frame (no gray, no dark blue)
BRIGHT = ["#ff7f0e", "#2ee62e", "#ff3030", "#19d3f3", "#ff77ff", "#ffe100", "#b388ff", "#00e6a8"]


def raw(name, split="train"):
    """Original 8-bit frame as distributed."""
    sub = "train/train_images" if split == "train" else "test/test_images"
    return cv2.imread(f"{DATA}/{sub}/{name}", cv2.IMREAD_GRAYSCALE)


def _outline(ax, mask, color, lw=1.0):
    for c in cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
        c = c[:, 0, :]; ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=color, lw=lw)


def _crop(ax, img, box, title):
    x0, y0, x1, y1 = box
    ax.imshow(img[y0:y1, x0:x1], cmap="gray", extent=(x0, x1, y1, y0), interpolation="nearest"); ax.set_title(title, fontsize=9); ax.axis("off")


def busiest_box(masks, size=700):
    """Square window of `size` px that holds the most annotated pixels."""
    m = np.zeros((H, W), np.float32)
    for q in masks: m += q
    s = cv2.blur(m, (size, size)); y, x = np.unravel_index(int(s.argmax()), s.shape)
    x0, y0 = int(np.clip(x - size // 2, 0, W - size)), int(np.clip(y - size // 2, 0, H - size))
    return x0, y0, x0 + size, y0 + size


def annotators(name, size=700, pred=None):
    """One train frame: the image and what every annotator record marked in the same window.
    pred (optional): our masks for the same frame, shown in a last panel."""
    sets = load_sets()[name]; masks = [poly_masks(s) for s in sets]; img = raw(name)
    box = busiest_box([q for ms in masks for q in ms], size); n = len(sets) + 1 + (pred is not None)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 3.8))
    _crop(axs[0], img, box, name)
    panels = [(f"annotator record {k + 1}: {len(ms)} filaments in the frame", ms, COLORS[k % len(COLORS)]) for k, ms in enumerate(masks)]
    if pred is not None: panels.append((f"our prediction: {len(pred)} filaments in the frame", pred, COLORS[3]))
    for ax, (title, ms, col) in zip(axs[1:], panels):
        _crop(ax, img, box, title)
        for q in ms: _outline(ax, q, col)
        ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1])
    fig.tight_layout(); return fig


def preprocessing(name, split="train"):
    """Raw frame, the fitted limb, and the flat-fielded frame the networks see."""
    names, X = open_cache(split, "flat"); i = names.index(name); img = raw(name, split)
    geom = json.load(open(f"{CACHE}/{split}_meta.json"))["geom"][i]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
    axs[0].imshow(img, cmap="gray"); axs[0].add_patch(plt.Circle(geom[:2], geom[2], fill=False, color="tab:red", lw=0.8))
    axs[0].set_title(f"raw frame, fitted limb: centre ({geom[0]:.1f}, {geom[1]:.1f}), R {geom[2]:.1f} px", fontsize=9); axs[0].axis("off")
    axs[1].imshow(np.asarray(X[i], np.float32), cmap="gray", vmin=-4, vmax=4)
    axs[1].set_title("limb darkening removed, robust z-score, off-disk = 0", fontsize=9); axs[1].axis("off")
    fig.tight_layout(); return fig


def read_submission(path):
    """CSV -> {frame stem: [bool masks]}."""
    d = {}
    for r in csv.DictReader(open(path)):
        d.setdefault(r["filament_id"].split("_")[0], []).append(mu.decode({"size": [H, W], "counts": r["segmentation_rle"].encode("ascii")}).astype(bool))
    return d


def prediction(stem, sub, prob=None, size=800):
    """A test frame with the submitted masks; with `prob` (H,W float) also the blended probability map."""
    img = raw(f"{stem}.jpeg", "test"); masks = sub.get(stem, []); box = busiest_box(masks, size) if masks else (624, 624, 1424, 1424)
    n = 3 if prob is not None else 2; fig, axs = plt.subplots(1, n, figsize=(4.4 * n, 4.6))
    _crop(axs[0], img, box, f"{stem} (test)")
    if prob is not None:
        x0, y0, x1, y1 = box; axs[1].imshow(prob[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=1, extent=(x0, x1, y1, y0)); axs[1].set_title("blended probability", fontsize=9); axs[1].axis("off")
    _crop(axs[-1], img, box, f"{len(masks)} predicted filaments in the frame")
    for k, q in enumerate(masks): _outline(axs[-1], q, BRIGHT[k % len(BRIGHT)])
    axs[-1].set_xlim(box[0], box[2]); axs[-1].set_ylim(box[3], box[1])
    fig.tight_layout(); return fig


def pipeline_steps(prob, rule, box):
    """Post-processing on one window: probability -> components of prob > lo -> kept objects, trimmed -> one piece."""
    lab0, area, peak, mean = PQ.components(prob, rule["lo"]); keep = PQ.keep_mask(area, peak, mean, rule, PQ.radius(lab0, len(area)))
    lab, _ = PQ.trim(prob, lab0, area, rule["lo"], rule["inner"]); final = PQ.instances(prob, rule)
    x0, y0, x1, y1 = box; ext = (x0, x1, y1, y0); fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    axs[0].imshow(prob[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=1, extent=ext); axs[0].set_title("1. blended probability", fontsize=9)
    c = np.where(lab0 > 0, np.where(keep[lab0], 2, 1), 0)[y0:y1, x0:x1]
    axs[1].imshow(c, cmap=matplotlib.colors.ListedColormap(["black", "dimgray", "gold"]), vmin=0, vmax=2, extent=ext, interpolation="nearest")
    axs[1].set_title(f"2. components of prob > {rule['lo']}: kept (yellow) / dropped (gray)", fontsize=9)
    t = (np.where(keep[lab], lab, 0) > 0)[y0:y1, x0:x1]
    axs[2].imshow(t, cmap="gray", extent=ext, interpolation="nearest"); axs[2].set_title(f"3. trimmed to prob > {rule['inner']:.2f}", fontsize=9)
    out = np.zeros((H, W), np.int32)
    for k, (m, _) in enumerate(final, 1): out[m] = k
    axs[3].imshow(np.ma.masked_equal(out[y0:y1, x0:x1], 0) % 10, cmap="tab10", vmin=0, vmax=9, extent=ext, interpolation="nearest")
    axs[3].set_facecolor("black"); axs[3].set_title("4. one piece per object (final instances)", fontsize=9)
    for a in axs: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); return fig


def score_distributions(npz):
    """Histograms of the best IoU and Dice of every ground-truth segment (output of evaluate.py)."""
    g = np.load(npz)["gt"]; fig, axs = plt.subplots(1, 2, figsize=(9, 3.2)); bins = np.linspace(0, 1, 21)
    for ax, col, nm in ((axs[0], 1, "IoU"), (axs[1], 2, "Dice")):
        ax.hist(g[:, col], bins, color="tab:blue"); ax.set_xlabel(f"best {nm} of a ground-truth segment"); ax.set_ylabel("segments")
        ax.set_title(f"{nm}: {np.mean(g[:, col] == 0):.0%} not found, mean over matched {g[g[:, 1] > 0.5, col].mean():.3f}", fontsize=9)
    axs[0].axvline(0.5, color="tab:red", lw=1, ls="--"); axs[1].axvline(2 / 3, color="tab:red", lw=1, ls="--")
    fig.tight_layout(); return fig
