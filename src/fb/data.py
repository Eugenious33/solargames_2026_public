"""Data access + on-disk cache of normalised frames/targets (float16 / uint8 memmaps on fast local disk).

build:  python data.py            # everything: train + test frames and all training targets (~27 GB, 1-2 min on 24 cores)
        python data.py test       # test frames only (enough to reproduce the submission from the shipped weights)
        python data.py train:flat # train frames without targets (enough to re-predict the out-of-fold maps)
        python data.py exact      # only (re)build the exact-raster targets tfullc / tsoftc
"""
import os, sys, json, random, collections, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disk import fit_disk, flat_field, zscore_disk

# Locations are environment-driven so the same code runs on any machine and inside the Docker image
# (docker/run.sh mounts the project at /work and the cache at /cache).
ROOT = os.environ.get("SOLAR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = f"{ROOT}/data/MAGFiLO_1.0_Kaggle_2026"
ANN = f"{DATA}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CACHE = os.environ.get("SOLAR_CACHE") or os.path.expanduser("~/.cache/solar2026")
# trained weights: <RUNS>/<run name>/ema.pt (U-Net) or model.pt (Mask R-CNN). The public repository ships them in weights/.
RUNS = os.environ.get("SOLAR_RUNS") or (f"{ROOT}/weights" if os.path.isdir(f"{ROOT}/weights") else f"{ROOT}/fb/runs")
H = W = 2048
DISK = (1023.1, 1024.1, 903.9)     # measured: GONG frames are pre-registered (std 0.4 px / 0.9 px)


def load_sets():
    """file_name -> list of annotator sets; each set = list of polygon segmentations."""
    d = json.load(open(ANN))
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a["segmentation"])
    files = collections.defaultdict(list)
    for im in d["images"]:
        files[im["file_name"]].append(by_img.get(im["id"], []))
    return files


def split_names(files, n_val=141, seed=42, fold=0):
    """(train, val) by physical file. Fold k holds out names[k*n_val:(k+1)*n_val] of one fixed shuffle
    (the last fold also takes the remainder), so folds 0..4 cover all 707 frames exactly once.
    Fold 0's first 70 names == the teacher's hold-out (same shuffle)."""
    names = sorted(files)
    random.Random(seed).shuffle(names)
    lo = fold * n_val
    hi = len(names) if lo + 2 * n_val > len(names) else lo + n_val
    return names[:lo] + names[hi:], names[lo:hi]


def union(segs):
    """LEGACY raster (cv2.fillPoly): 11% more area than the masks the host scores against, because fillPoly
    also paints every pixel the outline touches. Targets `tfull` / `tsoft` and all runs before 2026-11-11
    used it; kept only so those runs stay reproducible. New training uses `union_exact`."""
    m = np.zeros((H, W), np.uint8)
    for s in segs:
        cv2.fillPoly(m, [np.array(p, np.float32).reshape(-1, 2).round().astype(np.int32) for p in s], 1)
    return m


def poly_masks(segs):
    """Instance masks rasterised exactly like the host scorer (pycocotools)."""
    from pycocotools import mask as mu
    return [mu.decode(mu.merge(mu.frPyObjects(s, H, W))).astype(bool) for s in segs]


def union_exact(segs):
    """Union of one annotator set, rasterised exactly like the host scorer (pycocotools). Targets `tfullc` / `tsoftc`."""
    from pycocotools import mask as mu
    polys = [p for s in segs for p in s]
    if not polys: return np.zeros((H, W), np.uint8)
    return mu.decode(mu.merge(mu.frPyObjects(polys, H, W)))


def _prep_exact(args):
    i, sets = args
    mm = lambda name: np.load(f"{CACHE}/train_{name}.npy", mmap_mode="r+")
    full = union_exact(max(sets, key=len))
    mm("tfullc")[i] = full
    mm("tsoftc")[i] = np.round(255 * np.mean([union_exact(s) for s in sets], 0)).astype(np.uint8)
    return int(full.sum())


def build_exact():
    """Add the exact-raster targets to an existing cache without touching frames or legacy targets."""
    files = load_sets(); names = json.load(open(f"{CACHE}/train_meta.json"))["names"]
    assert names == sorted(files)
    for name in ("tfullc", "tsoftc"):
        np.lib.format.open_memmap(f"{CACHE}/train_{name}.npy", "w+", np.uint8, (len(names), H, W)).flush()
    with ProcessPoolExecutor(min(12, os.cpu_count() or 8)) as ex:
        px = list(ex.map(_prep_exact, [(i, files[fn]) for i, fn in enumerate(names)], chunksize=4))
    old = np.load(f"{CACHE}/train_tfull.npy", mmap_mode="r") if os.path.exists(f"{CACHE}/train_tfull.npy") else None
    msg = f"exact targets cached for {len(names)} frames; filament pixels {sum(px)}"
    if old is not None:
        k = range(0, len(names), 7); a_old = sum(int(np.count_nonzero(old[i])) for i in k)
        msg += f"; legacy cv2 targets are {a_old / max(sum(px[i] for i in k), 1):.3f}x this area (every 7th frame)"
    print(msg, flush=True)


def _prep(args):
    split, i, fn, sets, zs = args
    sub = "train/train_images" if split == "train" else "test/test_images"
    img = cv2.imread(f"{DATA}/{sub}/{fn}", cv2.IMREAD_GRAYSCALE)
    geom = fit_disk(img)
    xn, _ = flat_field(img, geom)
    mm = lambda name, dt: np.load(f"{CACHE}/{split}_{name}.npy", mmap_mode="r+")
    mm("flat", np.float16)[i] = xn.astype(np.float16)
    if zs: mm("zs", np.float16)[i] = zscore_disk(img)[0].astype(np.float16)
    if sets is not None:
        mm("tfull", np.uint8)[i] = union(max(sets, key=len))
        mm("tsoft", np.uint8)[i] = np.round(255 * np.mean([union(s) for s in sets], 0)).astype(np.uint8)
    return geom


def build(split, frames_only=False):
    """Cache one split. frames_only: just the flat-fielded frames (what inference needs); otherwise also the plain
    z-score frames (A/B baseline) and, for train, the legacy targets."""
    os.makedirs(CACHE, exist_ok=True)
    if split == "train":
        files = load_sets(); names = sorted(files)
    else:
        files = None; names = sorted(os.listdir(f"{DATA}/test/test_images"))
    n = len(names); targets = files is not None and not frames_only
    for name, dt in [("flat", np.float16)] + ([] if frames_only else [("zs", np.float16)]) + ([("tfull", np.uint8), ("tsoft", np.uint8)] if targets else []):
        np.lib.format.open_memmap(f"{CACHE}/{split}_{name}.npy", "w+", dt, (n, H, W)).flush()
    with ProcessPoolExecutor(min(24, os.cpu_count() or 8)) as ex:
        geoms = list(ex.map(_prep, [(split, i, fn, files[fn] if targets else None, not frames_only) for i, fn in enumerate(names)], chunksize=4))
    json.dump({"names": names, "geom": geoms}, open(f"{CACHE}/{split}_meta.json", "w"))
    g = np.array(geoms)
    print(split, n, "frames cached; disk fit mean", g.mean(0).round(2), "std", g.std(0).round(2), flush=True)


def open_cache(split, name):
    meta = json.load(open(f"{CACHE}/{split}_meta.json"))
    return meta["names"], np.load(f"{CACHE}/{split}_{name}.npy", mmap_mode="r")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["train", "test", "exact"]):   # `python data.py exact` = only the exact-raster targets
        if s == "exact": build_exact()
        elif s == "test": build("test", frames_only=True)       # nothing downstream uses the z-score copy of the test frames
        else: build(s.split(":")[0], frames_only=s.endswith(":flat"))
