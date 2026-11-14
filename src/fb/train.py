"""Train a native-resolution semantic U-Net on [normalised frame + classical filter channels] and score it
with the host-style pooled PQ on a file-level hold-out.

Every arm of an ablation shares steps / batch / crop / seed, only the listed switches differ:
  --base flat|zs      limb-darkening flat-field + robust z-score   vs   plain z-score over img>15 (teacher)
  --spec x,sw5,...    channels produced by the fixed FilterBank front-end ('x' = frame only)
  --aug basic|strong  D4 + gain/offset   vs   + free rotation/zoom, tone curve, seeing, veil, noise
  --target tfullc|tsoftc  fullest annotator set   vs   mean of annotator sets (soft consensus), exact host raster
  --init run            warm start from a finished run (e.g. to move a model from the legacy raster to the exact one)

example: python train.py --name fb8 --spec x,lc128,sw4,sw5,sw6,ridge7,ridge10,gabor24,gabor48
"""
import argparse, os, sys, time, json, math, threading, queue, copy, numpy as np, cv2, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, CACHE, RUNS, DISK, H, W, load_sets, split_names, poly_masks, open_cache
from disk import VALID_FRAC, radius_map
import model as MODEL
from augment import geo_sample, photometric
import pq as PQ

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--out", default=RUNS)
p.add_argument("--base", default="flat", choices=["flat", "zs"])
p.add_argument("--spec", default="x")
p.add_argument("--aug", default="basic", choices=["basic", "strong"])
p.add_argument("--target", default="tfullc", choices=["tfullc", "tsoftc", "tfull", "tsoft"],
               help="tfullc / tsoftc = exact host raster (pycocotools); tfull / tsoft = legacy cv2 raster, 8-11% fatter")
p.add_argument("--init", default="", help="warm start: name of a finished run whose ema.pt initialises the network")
p.add_argument("--enc", default="resnet34")
p.add_argument("--arch", default="unet", choices=["unet", "unetpp"])
p.add_argument("--lsf", type=int, default=0, help="1 = add the learnable Fourier stem (spectral.py)")
p.add_argument("--mix", type=int, default=0, help="1 = learnable per-pixel mixer (soft clip, channel gate, MLP, pixel head) on the filter channels")
p.add_argument("--aux", type=float, default=0.3, help="weight of the mixer pixel-head loss")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--crop", type=int, default=768)
p.add_argument("--margin", type=int, default=128)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--nval", type=int, default=141)
p.add_argument("--fold", type=int, default=0, help="which block of the fixed file shuffle is held out (0..4)")
p.add_argument("--workers", type=int, default=8)
p.add_argument("--twins", type=float, default=0.0, help="probability of swapping a train frame for one of its temporal twins")
p.add_argument("--twin_min_corr", type=float, default=0.6)
p.add_argument("--train_frac", type=float, default=1.0, help="use only this share of the training frames (learning curve)")
# switches of the 2026-11-12 "lift" experiments (E32); the defaults reproduce every earlier run
p.add_argument("--dice_w", type=float, default=1.0, help="weight of the soft Dice term; 0 = BCE only")
p.add_argument("--sched", default="cos", choices=["cos", "flat"], help="flat = constant lr after warm-up, cosine decay only over the last --anneal steps")
p.add_argument("--anneal", type=int, default=0, help="--sched flat: number of final steps with cosine decay to 2%% of lr")
p.add_argument("--ckpt_every", type=int, default=0, help="also save the EMA weights every N steps as ema_<step>.pt (dose-response of training length)")
p.add_argument("--rec_w", type=int, default=0, help="1 = draw train frames in proportion to their number of annotator records (the metric pools records)")
p.add_argument("--enc_lr_k", type=float, default=1.0, help="learning-rate multiplier of the encoder (E32: transformer / ConvNeXt encoders are fine-tuned at a lower rate than the decoder)")
p.add_argument("--iw", default="", help="per-pixel weights of the BCE term: name of a cache train_<name>.npy (uint8, weight = 1 + v / 64) built by "
                                        "src/analysis/lift_20261112/build_iw.py so that small annotated objects count as much as large ones (E32); '' = off")
a = p.parse_args()
out = f"{a.out}/{a.name}"; os.makedirs(out, exist_ok=True)
fast = f"{CACHE}/runs/{a.name}"; os.makedirs(fast, exist_ok=True)
log = open(f"{out}/log.txt", "w")
def say(*s):
    msg = " ".join(str(v) for v in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()
say("args", json.dumps(vars(a)))
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)                                         # parallelism comes from our own threads / processes
dev = "cuda"

# ------------------------------------------------------------------ data
files = load_sets()
tr_names, va_names = split_names(files, a.nval, fold=a.fold)
tr_names = tr_names[:max(8, int(round(len(tr_names) * a.train_frac)))]   # split_names order is already shuffled
names, Xmm = open_cache("train", a.base)
_, Ymm = open_cache("train", a.target)
pos = {n: i for i, n in enumerate(names)}
t0 = time.time()
tr_idx = np.array(sorted(pos[n] for n in tr_names))
X = Xmm[tr_idx]; Y = Ymm[tr_idx]                               # into RAM
WT = open_cache("train", a.iw)[1][tr_idx] if a.iw else None    # --iw: weight code per pixel, 0 = weight 1
fg = [np.nonzero(y[::8, ::8]) for y in Y]
TW = [[] for _ in tr_idx]                                       # per train frame: its twin images (same labels)
if a.twins > 0:
    tm = json.load(open(f"{CACHE}/twin_meta.json")); Tmm = np.load(f"{CACHE}/twin_flat.npy", mmap_mode="r")
    loc = {names[i]: k for k, i in enumerate(tr_idx)}           # hold-out parents are absent -> their twins are never used
    use = [j for j, (pn, c) in enumerate(zip(tm["parent"], tm["corr"])) if pn in loc and c is not None and c >= a.twin_min_corr]
    Tw = Tmm[np.array(use)]
    for k, j in enumerate(use): TW[loc[tm["parent"][j]]].append(k)
    say(f"twins: {len(use)} extra views for {sum(bool(t) for t in TW)}/{len(TW)} train frames (corr >= {a.twin_min_corr})")
r_full = radius_map(DISK[0], DISK[1])
VALID = ((r_full < VALID_FRAC * DISK[2]) if a.base == "flat" else np.ones((H, W), bool)).astype(np.uint8)
ONDISK = torch.from_numpy(r_full < DISK[2] + 3).to(dev)        # no GT beyond the limb -> zero prob there
say(f"train {len(tr_idx)} / val {len(va_names)} frames in RAM, {time.time()-t0:.0f}s")

C, M, B = a.crop, (a.margin if (a.spec != "x" or a.lsf) else 0), a.bs
S = C + 2 * M
strong = a.aug == "strong"
NREC = np.array([len(files[names[i]]) for i in tr_idx], np.float64); PREC = NREC / NREC.sum()   # --rec_w: frame weight = its records

def make_batch(rs):
    xs, ys, vs, ws = [], [], [], []
    for _ in range(B):
        i = int(rs.choice(len(X), p=PREC)) if a.rec_w else rs.randint(len(X)); y = Y[i]
        if rs.rand() < 0.7 and len(fg[i][0]):                  # foreground-centred crop
            k = rs.randint(len(fg[i][0])); j = int(300 * C / 1024)
            cy, cx = fg[i][0][k] * 8 + rs.randint(-j, j + 1), fg[i][1][k] * 8 + rs.randint(-j, j + 1)
        else:
            cy, cx = rs.randint(C // 2, H - C // 2 + 1), rs.randint(C // 2, W - C // 2 + 1)
        cy, cx = int(np.clip(cy, C // 2, H - C // 2)), int(np.clip(cx, C // 2, W - C // 2))
        img = Tw[TW[i][rs.randint(len(TW[i]))]] if (TW[i] and rs.rand() < a.twins) else X[i]
        x_, y_, v_, *w_ = geo_sample(rs, [img, y, VALID] + ([WT[i]] if a.iw else []), cy, cx, S, strong and rs.rand() < 0.5)
        xs.append(x_); ys.append(y_); vs.append(v_); ws.extend(w_)
    yb = np.stack(ys)[:, None]
    if a.target.startswith("tsoft"): yb = yb / 255.0
    b = (torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(yb.astype(np.float32)), torch.from_numpy((np.stack(vs) > 0.5).astype(np.float32))[:, None])
    return b + ((torch.from_numpy((1 + np.stack(ws)[:, None] / 64).astype(np.float32)),) if a.iw else ())   # outside the frame: weight 1

q = queue.Queue(maxsize=24); stop = False
def worker(tid):
    rs = np.random.RandomState(a.seed * 1000 + tid)
    while not stop:
        b = make_batch(rs)
        while not stop:
            try: q.put(b, timeout=0.5); break
            except queue.Full: pass
for t in range(a.workers): threading.Thread(target=worker, args=(t,), daemon=True).start()

# ------------------------------------------------------------------ model
model = MODEL.build(vars(a)).to(dev).to(memory_format=torch.channels_last)
if a.init:                                                       # a run name under --out, or a path to a checkpoint
    model.load_state_dict(torch.load(a.init if os.path.isfile(a.init) else f"{a.out}/{a.init}/ema.pt", map_location=dev)["model"])
    say(f"warm start from {a.init}")
ema = copy.deepcopy(model).eval()
for q_ in ema.parameters(): q_.requires_grad_(False)
stem_p = list(model.stem.parameters()) if model.stem is not None else []
groups = [{"params": list(model.net.parameters()), "weight_decay": 1e-4, "k": 1.0}]
if a.enc_lr_k != 1.0:                                            # encoder at its own rate, decoder and head at --lr
    enc_ids = {id(q_) for q_ in model.net.encoder.parameters()}
    groups = [{"params": list(model.net.encoder.parameters()), "weight_decay": 1e-4, "k": a.enc_lr_k},
              {"params": [q_ for q_ in model.net.parameters() if id(q_) not in enc_ids], "weight_decay": 1e-4, "k": 1.0}]
if stem_p: groups.append({"params": stem_p, "weight_decay": 0.0, "k": 10.0})            # few scale-like params: faster lr, no decay
if model.mixer is not None:
    gp = [model.mixer.log_t, model.mixer.gate]
    groups.append({"params": gp, "weight_decay": 0.0, "k": 10.0})                         # per-channel clip / gate: like the stem
    groups.append({"params": [q_ for q_ in model.mixer.parameters() if all(q_ is not g for g in gp)], "weight_decay": 1e-4, "k": 1.0})
seen = {id(q_) for g in groups for q_ in g["params"]}
rest = [q_ for q_ in model.parameters() if q_.requires_grad and id(q_) not in seen]          # e.g. the full-resolution head
if rest: groups.append({"params": rest, "weight_decay": 1e-4, "k": 1.0})
say(f"params: {sum(q_.numel() for g in groups for q_ in g['params'])/1e6:.2f}M in {len(groups)} groups")
opt = torch.optim.AdamW(groups, lr=a.lr)
scaler = torch.amp.GradScaler("cuda")
dice = smp.losses.DiceLoss("binary")

t0 = time.time(); model.train(); run = 0.0
for step in range(1, a.steps + 1):
    ph = step / a.steps if a.sched == "cos" else max(0, step - (a.steps - a.anneal)) / max(a.anneal, 1)
    lr = a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * ph))) * min(1, step / 100)
    for g in opt.param_groups: g["lr"] = lr * g["k"]
    x, y, v, *wb = q.get()
    x, y, v = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), v.to(dev, non_blocking=True)
    wb = wb[0].to(dev, non_blocking=True) if wb else None
    with torch.no_grad():
        x = photometric(x, v, strong)
        y = y[..., M:S - M, M:S - M] if M else y
        if wb is not None and M: wb = wb[..., M:S - M, M:S - M]
    lo = model(x, v, M)                                          # filters fp32, network fp16, logits fp32
    if wb is None: loss = F.binary_cross_entropy_with_logits(lo, y)
    else: loss = (F.binary_cross_entropy_with_logits(lo, y, reduction="none") * wb).sum() / wb.sum()   # --iw: weighted mean of the pixel BCE
    if a.dice_w > 0: loss = loss + a.dice_w * dice(lo, y)
    if model.mixer is not None:                                 # the pixel head must classify from filters alone
        loss = loss + a.aux * F.binary_cross_entropy_with_logits(model.last_aux.float(), y)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    with torch.no_grad():
        d = min(0.999, (1 + step) / (10 + step))
        for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1 - d)
        for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
    run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
    if step % 250 == 0: say(f"step {step} {(time.time()-t0)/60:.1f}min loss {run:.4f} lr {lr:.2e} queue {q.qsize()}")
    if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
        torch.save({"model": ema.state_dict(), "args": vars(a), "step": step}, f"{out}/ema_{step}.pt")
stop = True
train_min = (time.time() - t0) / 60
torch.save({"model": ema.state_dict(), "args": vars(a)}, f"{out}/ema.pt")
say(f"trained {a.steps} steps in {train_min:.1f} min")
if ema.stem is not None:
    prof = ema.stem.profiles()
    json.dump(prof, open(f"{out}/spectral_profiles.json", "w"))
    say("learned pass-bands, period px of max gain:", {k: [round(prof[k]["periods_px"][int(np.argmax(g))]) for g in prof[k]["gains"]] for k in ("iso", "ori")},
        "angular width rad", [round(v, 2) for v in prof["ang_width_rad"]])

if ema.mixer is not None:
    g = torch.sigmoid(ema.mixer.gate).flatten().tolist(); t = ema.mixer.log_t.exp().flatten().tolist()
    gates = {n: dict(gate=round(gv, 3), clip=round(tv, 2)) for n, gv, tv in zip(ema.channel_names()[(1 if ema.mix == 1 else 0):], g, t)}
    json.dump(gates, open(f"{out}/mixer_gates.json", "w"), indent=1)
    say("mixer channel gates:", {n: v["gate"] for n, v in sorted(gates.items(), key=lambda kv: -kv[1]["gate"])})

# ------------------------------------------------------------------ hold-out: full 2048 frame, hflip TTA
@torch.no_grad()
def predict(xn):
    x = torch.from_numpy(np.asarray(xn, np.float32))[None, None].to(dev)
    v = torch.from_numpy(VALID)[None, None].to(dev).float()
    pr = 0
    for fl in (False, True):
        xi, vi = (x.flip(-1), v.flip(-1)) if fl else (x, v)
        o = torch.sigmoid(ema(xi, vi))
        pr = pr + (o.flip(-1) if fl else o)
    return ((pr / 2)[0, 0] * ONDISK).cpu().numpy()

t0 = time.time()
probs = np.zeros((len(va_names), H, W), np.uint8)
for k, n in enumerate(va_names):
    probs[k] = np.round(predict(Xmm[pos[n]]) * 255).astype(np.uint8)
np.save(f"{fast}/val_probs.npy", probs)
json.dump(va_names, open(f"{fast}/val_names.json", "w"))
say(f"predicted {len(va_names)} val frames in {time.time()-t0:.0f}s ({(time.time()-t0)/len(va_names)*1000:.0f} ms/frame incl. filters + TTA)")

def _score(k):
    n = va_names[k]
    gts = [[np.flatnonzero(m) for m in poly_masks(s)] for s in files[n]]
    pr = probs[k].astype(np.float32) / 255
    t = np.asarray(Ymm_full[pos[n]]).ravel() > 0
    hp = np.bincount(probs[k].ravel()[t], minlength=256); hn = np.bincount(probs[k].ravel()[~t], minlength=256)
    return PQ.frame_stats(pr, gts), hp, hn

_, Ymm_full = open_cache("train", "tfullc" if a.target.endswith("c") else "tfull")   # pixel AP against the raster the run was trained on
with ProcessPoolExecutor(min(24, os.cpu_count() or 8)) as ex:
    res = list(ex.map(_score, range(len(va_names)), chunksize=2))
st = np.stack([r[0] for r in res])                              # (n_val, n_cfg, 4)
hp = sum(r[1] for r in res).astype(np.float64); hn = sum(r[2] for r in res).astype(np.float64)
cp, cn = np.cumsum(hp[::-1]), np.cumsum(hn[::-1])
rec, prec = cp / cp[-1], cp / np.maximum(cp + cn, 1)
ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
bdice = float(np.max(2 * cp / (cp + cn + cp[-1])))

def best(sub):
    acc = st[sub].sum(0); pqs = np.array([PQ.pq(r) for r in acc]); j = int(pqs.argmax())
    return dict(pq=float(pqs[j]), cfg=PQ.GRID[j], acc=[float(v) for v in acc[j]]), pqs
b_all, pq_all = best(slice(None)); b_70, _ = best(slice(0, 70))   # (teacher-70 is meaningful for fold 0 only)
# honest number: pick the keep-rule on one half of the hold-out, score on the other half (and vice versa)
h1, h2 = np.arange(0, len(va_names), 2), np.arange(1, len(va_names), 2)
j1 = int(np.argmax([PQ.pq(r) for r in st[h1].sum(0)])); j2 = int(np.argmax([PQ.pq(r) for r in st[h2].sum(0)]))
pq_cross = PQ.pq(st[h2].sum(0)[j1] + st[h1].sum(0)[j2])
ref = PQ.GRID.index(PQ.REF)
result = dict(name=a.name, args=vars(a), train_min=train_min, pixel_ap=ap, pixel_best_dice=bdice,
              pq_best=b_all, pq_cross=float(pq_cross), pq_ref_cfg=float(pq_all[ref]), pq_teacher70=b_70)
json.dump(result, open(f"{out}/result.json", "w"), indent=1)
np.save(f"{fast}/val_stats.npy", st)
say(f"RESULT {a.name}: pixelAP {ap:.4f} bestDice {bdice:.4f} | PQ best {b_all['pq']:.4f} {b_all['cfg']} sumIoU/TP/FP/FN {[round(v,1) for v in b_all['acc']]}"
    f" | PQ cross-fit {pq_cross:.4f} | PQ@ref {pq_all[ref]:.4f} | PQ teacher-70 {b_70['pq']:.4f}")
os._exit(0)
