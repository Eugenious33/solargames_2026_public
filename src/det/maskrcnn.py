"""Instance-level ensemble member: Mask R-CNN on native-resolution crops of the pre-filtered frame.

Follows the public "detector" recipe (Mask R-CNN R50-FPN v2, small anchors 16..256, mask RoI 28,
fullest annotator set, D4 + gain augmentation) with two changes:
  * input is a 3-channel composite of classical filters instead of gray x3:
        [flat-fielded frame, starlet band-pass (scales 4+5), Hessian dark-ridge sigma 10]
  * trained on native-resolution 1024 crops and applied to the full 2048 frame (the public version
    downsamples to 1024, where a median filament is ~5 px wide).
Output per frame: an "objectness" map  max_n(score_n * mask_n)  stored exactly like a semantic
probability map, so it can be scored alone (score.py) or blended with the U-Net members.

  python src/det/maskrcnn.py prep                    # composite cache (GPU, ~2 min); `prep --only test` = test frames only
  python src/det/maskrcnn.py train --name det_r50    # train + predict the hold-out
  python src/det/maskrcnn.py val   --name det_r50    # (re)predict the hold-out from saved weights
  python src/det/maskrcnn.py test  --name det_r50    # predict the test frames
"""
import argparse, os, sys, json, time, math, threading, queue, numpy as np, cv2, torch
for _d in ("fb", "fb_exp"):                                    # an experiment copy of the fb modules (second machine) wins when present
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", _d)
    if os.path.exists(os.path.join(_p, "data.py")): sys.path.insert(0, _p)
from data import ROOT, CACHE, RUNS, DISK, H, W, load_sets, split_names, open_cache
from disk import VALID_FRAC, radius_map
from filters import FilterBank

SPEC = ["x", "sw4", "sw5", "ridge10"]
p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["prep", "train", "val", "test"])
p.add_argument("--name", default="det_r50"); p.add_argument("--iters", type=int, default=6000)
p.add_argument("--bs", type=int, default=2); p.add_argument("--crop", type=int, default=1024)
p.add_argument("--lr", type=float, default=0.01); p.add_argument("--nval", type=int, default=141)
p.add_argument("--seed", type=int, default=0); p.add_argument("--workers", type=int, default=6)
p.add_argument("--fold", type=int, default=0, help="held-out block of the fixed file shuffle (0..4)")
p.add_argument("--only", default="", choices=["", "train", "test"], help="prep: build the composite cache of one split only")
p.add_argument("--init", default="", help="train: warm start from a finished run (name) or a model.pt path instead of the COCO weights (E32)")
p.add_argument("--ema", type=float, default=0.0, help="train: keep an exponential moving average of the weights with this decay and save / predict with it (E32)")
a = p.parse_args()
dev = "cuda"; cv2.setNumThreads(0); torch.manual_seed(a.seed)
r_full = radius_map(DISK[0], DISK[1])
VALID = torch.from_numpy(r_full < VALID_FRAC * DISK[2])[None, None].to(dev).float()
ONDISK = torch.from_numpy(r_full < DISK[2] + 3).to(dev)


def composite(split):
    """(N,3,H,W) uint8: flat frame, starlet band-pass, dark ridge."""
    names, X = open_cache(split, "flat")
    fb = FilterBank(SPEC).to(dev)
    out = np.lib.format.open_memmap(f"{CACHE}/{split}_det3.npy", "w+", np.uint8, (len(names), 3, H, W))
    for i in range(len(names)):
        f = fb(torch.from_numpy(np.asarray(X[i], np.float32))[None, None].to(dev), VALID)[0]
        c = torch.stack([(f[0].clamp(-4, 4) + 4) / 8, ((f[1] + f[2]).clamp(-2, 2) + 2) / 4, (f[3] / 4).clamp(0, 1)])
        out[i] = (c * 255).round().byte().cpu().numpy()
    out.flush(); print(split, len(names), "composites cached", flush=True)


def build(pretrained=False):
    """pretrained=True starts from the COCO weights (training); inference loads our own weights and downloads nothing."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None, weights_backbone=None,
                                 min_size=a.crop, max_size=a.crop, image_mean=[0.5, 0.5, 0.0], image_std=[0.25, 0.25, 0.25],
                                 box_detections_per_img=150, box_score_thresh=0.05)
    # half-size anchors (16..256 instead of 32..512): still 3 per location, so the COCO RPN head is reused as is
    m.rpn.anchor_generator = AnchorGenerator(((16,), (32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 5)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    m.roi_heads.mask_roi_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
    return m.to(dev)


@torch.no_grad()
def predict(model, img_u8):
    """Full 2048 frame -> objectness map uint8 (H,W) and detections."""
    model.eval(); model.transform.min_size = (H,); model.transform.max_size = W
    x = torch.from_numpy(img_u8).to(dev).float() / 255
    acc = torch.zeros((H, W), device=dev)
    for fl in (False, True):                                    # hflip TTA
        xi = x.flip(-1) if fl else x
        with torch.autocast("cuda", dtype=torch.float16):
            o = model([xi])[0]
        m = torch.zeros((H, W), device=dev)
        for i in range(0, len(o["scores"]), 8):                 # chunked: N full-frame masks do not fit 8 GB at once
            m = torch.maximum(m, (o["masks"][i:i + 8, 0].float() * o["scores"][i:i + 8, None, None]).amax(0))
        del o
        acc = torch.maximum(acc, m.flip(-1)) if fl else m
    return ((acc * ONDISK) * 255).round().byte().cpu().numpy()


if a.cmd == "prep":
    for s in ([a.only] if a.only else ["train", "test"]): composite(s)
    sys.exit(0)

out = f"{RUNS}/{a.name}"; fast = f"{CACHE}/runs/{a.name}"; os.makedirs(out, exist_ok=True); os.makedirs(fast, exist_ok=True)
if a.cmd == "test":
    model = build(); model.load_state_dict(torch.load(f"{out}/model.pt", map_location=dev))
    names, X = open_cache("test", "det3"); os.makedirs(f"{CACHE}/pred", exist_ok=True)
    P = np.zeros((len(names), H, W), np.uint8)
    for i in range(len(names)): P[i] = predict(model, np.array(X[i]))
    np.save(f"{CACHE}/pred/{a.name}_test.npy", P); json.dump({"names": names, "runs": [a.name]}, open(f"{CACHE}/pred/{a.name}_test.json", "w"))
    print("test predicted"); sys.exit(0)

log = open(f"{out}/log.txt", "a" if a.cmd == "val" else "w")
def say(*s):
    msg = " ".join(str(v) for v in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()
files = load_sets(); tr_names, va_names = split_names(files, a.nval, fold=a.fold)
names, Xmm = open_cache("train", "det3"); pos = {n: i for i, n in enumerate(names)}

def predict_holdout(model):
    t0 = time.time(); P = np.zeros((len(va_names), H, W), np.uint8)
    for k, n in enumerate(va_names): P[k] = predict(model, np.array(Xmm[pos[n]]))
    np.save(f"{fast}/val_probs.npy", P); json.dump(va_names, open(f"{fast}/val_names.json", "w"))
    say(f"hold-out predicted: {(time.time()-t0)/len(va_names)*1000:.0f} ms/frame -> {fast}/val_probs.npy")

if a.cmd == "val":
    model = build(); model.load_state_dict(torch.load(f"{out}/model.pt", map_location=dev))
    predict_holdout(model); os._exit(0)

# ----------------------------------------------------------------------------- train
say("args", json.dumps(vars(a)))
tr_idx = [pos[n] for n in tr_names]
POLY = [[np.array(s[0], np.float32).reshape(-1, 2) for s in max(files[n], key=len)] for n in tr_names]   # fullest set
C = a.crop

def sample(rs):
    k = rs.randint(len(tr_idx)); polys = POLY[k]
    if polys and rs.rand() < 0.8:                               # centre near a random filament
        c = polys[rs.randint(len(polys))].mean(0) + rs.randint(-250, 251, 2)
    else:
        c = rs.randint(C // 2, H - C // 2 + 1, 2)
    x0, y0 = (int(np.clip(v - C // 2, 0, H - C)) for v in c)
    img = np.asarray(Xmm[tr_idx[k]][:, y0:y0 + C, x0:x0 + C])
    masks = []
    for pl in polys:
        q = pl - (x0, y0)
        if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() >= C or q[:, 1].min() >= C: continue
        m = np.zeros((C, C), np.uint8); cv2.fillPoly(m, [q.round().astype(np.int32)], 1)
        if m.sum() >= 24: masks.append(m)
    kr = rs.randint(4); fl = rs.rand() < 0.5                    # D4
    img = np.rot90(img, kr, (1, 2)); masks = [np.rot90(m, kr) for m in masks]
    if fl: img = img[:, :, ::-1]; masks = [m[:, ::-1] for m in masks]
    img = np.ascontiguousarray(img).astype(np.float32) / 255
    img[:2] = np.clip((img[:2] - 0.5) * rs.uniform(0.85, 1.15) + 0.5 + rs.uniform(-0.03, 0.03), 0, 1)
    boxes, keep = [], []
    for m in masks:
        ys, xs = np.nonzero(m)
        if xs.max() > xs.min() and ys.max() > ys.min():
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]); keep.append(np.ascontiguousarray(m))
    t = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), "labels": torch.ones(len(boxes), dtype=torch.int64),
         "masks": torch.from_numpy(np.stack(keep)) if keep else torch.zeros((0, C, C), dtype=torch.uint8)}
    return torch.from_numpy(img), t

q = queue.Queue(maxsize=16); stop = False
def worker(tid):
    rs = np.random.RandomState(a.seed * 1000 + tid)
    while not stop:
        b = [sample(rs) for _ in range(a.bs)]
        while not stop:
            try: q.put(b, timeout=0.5); break
            except queue.Full: pass
for t in range(a.workers): threading.Thread(target=worker, args=(t,), daemon=True).start()

model = build(pretrained=not a.init)
if a.init:
    model.load_state_dict(torch.load(a.init if os.path.isfile(a.init) else f"{RUNS}/{a.init}/model.pt", map_location=dev)); say(f"warm start from {a.init}")
ema = None
if a.ema > 0:
    import copy
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters(): p_.requires_grad_(False)
params = [p_ for p_ in model.parameters() if p_.requires_grad]
opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")
t0 = time.time(); run = None; model.train()
for it in range(1, a.iters + 1):
    lr = a.lr * min(1, it / 300) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * it / a.iters)))
    for g in opt.param_groups: g["lr"] = lr
    b = q.get()
    imgs = [x.to(dev) for x, _ in b]; tg = [{k: v.to(dev) for k, v in t.items()} for _, t in b]
    with torch.autocast("cuda", dtype=torch.float16):
        losses = model(imgs, tg)
    loss = sum(losses.values())
    if not torch.isfinite(loss): continue
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.0); scaler.step(opt); scaler.update()
    if ema is not None:
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
    run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
    if it % 250 == 0: say(f"it {it} {(time.time()-t0)/60:.1f}min loss {run:.3f} " + " ".join(f"{k[5:]}={v.item():.3f}" for k, v in losses.items()))
stop = True
if ema is not None: model = ema
torch.save(model.state_dict(), f"{out}/model.pt")
say(f"trained {a.iters} iters in {(time.time()-t0)/60:.1f} min")
del opt, scaler, losses, loss, imgs, tg; torch.cuda.empty_cache()        # training buffers must go before full-frame inference
predict_holdout(model)
os._exit(0)
