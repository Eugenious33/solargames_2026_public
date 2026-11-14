"""Full-frame (2048x2048, no tiling / resizing) inference with D4 TTA and model averaging.

usage: python infer.py --runs runA,runB --split test --tag ens1      -> <cache>/pred/ens1_test.npy (uint8 probs)
       python infer.py --runs runA --split val --fold 2 --tta hflip --tag oof
            hold-out frames of fold 2 (the frames runA never saw). With --tag oof the maps are written where
            train.py writes them (<cache>/runs/runA/val_probs.npy), so `oof.py assemble` can rebuild the
            out-of-fold maps from shipped weights without retraining.
"""
import argparse, os, sys, time, json, numpy as np, torch
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, CACHE, RUNS, DISK, H, W, load_sets, split_names, open_cache
from disk import VALID_FRAC, radius_map
import model as MODEL

p = argparse.ArgumentParser()
p.add_argument("--runs", required=True); p.add_argument("--root", default=RUNS)
p.add_argument("--split", default="test", choices=["test", "val"]); p.add_argument("--tag", required=True)
p.add_argument("--tta", default="d4", choices=["none", "hflip", "d4"]); p.add_argument("--nval", type=int, default=141)
p.add_argument("--fold", type=int, default=0, help="--split val: which block of the fixed file shuffle is predicted")
a = p.parse_args()
dev = "cuda"
r_full = radius_map(DISK[0], DISK[1])
ONDISK = torch.from_numpy(r_full < DISK[2] + 3).to(dev)
VIEWS = {"none": [(0, False)], "hflip": [(0, False), (0, True)], "d4": [(k, f) for k in range(4) for f in (False, True)]}[a.tta]

models = []
for name in a.runs.split(","):
    m, ar = MODEL.load(f"{a.root}/{name}/ema.pt", dev)
    valid = torch.from_numpy((r_full < VALID_FRAC * DISK[2]) if ar["base"] == "flat" else np.ones((H, W), bool))[None, None].to(dev).float()
    models.append((m, ar["base"], valid))

src = "test" if a.split == "test" else "train"
arrays = {b: open_cache(src, b) for b in {m[1] for m in models}}
names = next(iter(arrays.values()))[0]
if a.split == "val":
    keep = split_names(load_sets(), a.nval, fold=a.fold)[1]
else:
    keep = names
pos = {n: i for i, n in enumerate(names)}


@torch.no_grad()
def predict(i):
    acc = 0
    for m, base, valid in models:
        x = torch.from_numpy(np.asarray(arrays[base][1][i], np.float32))[None, None].to(dev)
        for k, fl in VIEWS:
            xi, vi = torch.rot90(x, k, (-2, -1)), torch.rot90(valid, k, (-2, -1))
            if fl: xi, vi = xi.flip(-1), vi.flip(-1)
            o = torch.sigmoid(m(xi, vi))
            if fl: o = o.flip(-1)
            acc = acc + torch.rot90(o, -k, (-2, -1))
    return (acc / (len(models) * len(VIEWS)))[0, 0] * ONDISK


os.makedirs(f"{CACHE}/pred", exist_ok=True)
out = np.zeros((len(keep), H, W), np.uint8); t0 = time.time()
for j, n in enumerate(keep):
    out[j] = torch.round(predict(pos[n]) * 255).byte().cpu().numpy()
if a.split == "val" and a.tag == "oof":                       # same place and layout as train.py's hold-out maps
    assert "," not in a.runs, "--tag oof takes one run"; dst = f"{CACHE}/runs/{a.runs}"; os.makedirs(dst, exist_ok=True)
    np.save(f"{dst}/val_probs.npy", out); json.dump(keep, open(f"{dst}/val_names.json", "w")); where = f"{dst}/val_probs.npy"
else:
    where = f"{CACHE}/pred/{a.tag}_{a.split}.npy"; np.save(where, out)
    json.dump({"names": keep, "runs": a.runs.split(","), "tta": a.tta}, open(f"{CACHE}/pred/{a.tag}_{a.split}.json", "w"))
print(f"{len(keep)} frames, {len(models)} models x {len(VIEWS)} views: {(time.time()-t0)/len(keep)*1000:.0f} ms/frame -> {where}")
