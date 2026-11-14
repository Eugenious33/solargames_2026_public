"""Host-style pooled Panoptic Quality + PQ-aware component selection.

Mirrors the organisers' self-evaluation notebook: the same prediction set is scored against every
annotator record of the frame, TP = IoU > 0.5 (no 1:1 matching), TP/FP/FN and sum-IoU are pooled
globally (micro), frames/predictions without GT records are ignored.
Semantic map -> instances = 8-connected components of prob > lo, kept when
peak >= peak, mean >= mean, area >= area (strict rules cut FPs: every extra object costs +0.5 in
the denominator for every annotator record).

`inner` trims the kept object to its pixels with prob > inner (0 = no trimming). The object is still
defined, and judged, by the `lo` component; only its outline shrinks. A filament is ~11 px wide, so
one pixel of extra width costs several points of IoU, and models trained on the legacy cv2 raster or
with a Dice loss draw masks wider than the host's pycocotools masks.
"""
import numpy as np, cv2

DISK = (1023.1, 1024.1, 903.9)     # same constants as data.DISK: every GONG frame is pre-registered

GRID = [dict(lo=lo, peak=pk, mean=mn, area=ar, inner=inn)
        for lo in (0.2, 0.3, 0.4, 0.5) for pk in (0.5, 0.9) for mn in (0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96)
        for ar in (100, 200, 400, 600) for inn in (0.0, 0.7, 0.8, 0.9, 0.95)]
REF = dict(lo=0.4, peak=0.9, mean=0.7, area=200, inner=0.0)        # fixed rule reported by every run (comparable across runs)


def components(prob, lo):
    """prob float32 (H,W) -> (label map int32, area[n], peak[n], mean[n]); index 0 = background."""
    n, lab = cv2.connectedComponents((prob > lo).astype(np.uint8), connectivity=8)
    idx = np.flatnonzero(lab)
    l = lab.ravel()[idx]; p = prob.ravel()[idx]
    area = np.bincount(l, minlength=n).astype(np.float64)
    mean = np.bincount(l, weights=p, minlength=n) / np.maximum(area, 1)
    order = np.argsort(p, kind="stable")                      # last write wins -> max
    peak = np.zeros(n); peak[l[order]] = p[order]
    return lab, area, peak, mean


def radius(lab, n):
    """r/R of every component centroid (index 0 = background -> 0)."""
    ys, xs = np.nonzero(lab); l = lab[ys, xs]; cnt = np.maximum(np.bincount(l, minlength=n), 1)
    cx = np.bincount(l, weights=xs, minlength=n) / cnt; cy = np.bincount(l, weights=ys, minlength=n) / cnt
    r = np.hypot(cx - DISK[0], cy - DISK[1]) / DISK[2]; r[0] = 0
    return r


def keep_mask(area, peak, mean, cfg, r=None):
    """cfg["zones"] (optional) = [[r_hi, mean_min, area_min], ...] in increasing r_hi: thresholds that depend on
    the distance from disk centre. Experts label far fewer filaments near the limb (0.34 per record at
    r/R 0.85-0.92 against 0.64 that a uniform rule keeps), so the limb ring needs a stricter rule."""
    if cfg.get("zones") is None:
        k = (area >= cfg["area"]) & (peak >= cfg["peak"]) & (mean >= cfg["mean"])
    else:
        z = np.minimum(np.searchsorted([q[0] for q in cfg["zones"]], r, side="right"), len(cfg["zones"]) - 1)
        mn = np.array([q[1] for q in cfg["zones"]])[z]; ar = np.array([q[2] for q in cfg["zones"]])[z]
        k = (area >= ar) & (peak >= cfg.get("peak", 0.0)) & (mean >= mn)
    k[0] = False
    return k


def trim(prob, lab, area, lo, inner):
    """Label map and areas after trimming every component to prob > inner (no-op when inner <= lo)."""
    if inner <= lo: return lab, area
    lt = np.where(prob > inner, lab, 0)
    return lt, np.bincount(lt.ravel(), minlength=len(area)).astype(np.float64)


def one_piece(m, c0, p, width=2):
    """Make an emitted mask obey the annotation rule of the dataset: one piece, no holes (MAGFiLO, Sci. Data 2024).

    Trimming to prob > inner leaves 13% of the objects in several pieces (the ground truth: 0.4%). Label-free repair:
      1. fill holes;
      2. drop islands smaller than max(30 px, 5% of the object);
      3. connect the remaining pieces, largest first, along the cheapest path through the probability map
         (cost 1 - prob, restricted to the object's own lo-component), drawn 2*width+1 px wide.
    m = trimmed mask, c0 = the lo-component it came from, p = probabilities; all three are the same crop.
    Out-of-fold effect (src/analysis/morph_sq.py): PQ 0.4564 -> 0.4570, objects with 2+ pieces 13% -> 0%, area +0.4%."""
    from scipy import ndimage as ndi
    from skimage.graph import MCP_Geometric
    m = ndi.binary_fill_holes(m)
    n, l, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    a = st[:, cv2.CC_STAT_AREA]; big = [k for k in range(1, n) if a[k] >= max(30, 0.05 * m.sum())]
    if not big: return m
    m = np.isin(l, big)
    if len(big) == 1: return m
    cost = np.where(c0, 1.02 - p, 50.0).astype(np.float64); disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    first = max(big, key=lambda k: a[k]); done = l == first; out = m.copy()
    for k in sorted((k for k in big if k != first), key=lambda k: -a[k]):
        mcp = MCP_Geometric(cost)
        src = np.argwhere(done & ~cv2.erode(done.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)); dst = np.argwhere(l == k)
        costs, _ = mcp.find_costs([tuple(s) for s in src], [tuple(e) for e in dst])
        path = np.zeros(m.shape, np.uint8)
        for y, x in mcp.traceback(tuple(dst[np.argmin(costs[dst[:, 0], dst[:, 1]])])): path[y, x] = 1
        out |= (cv2.dilate(path, disk) > 0) & c0; done |= l == k
    return out


def instances(prob, cfg):
    """Final instance masks [(bool mask, mean prob)], pixel-disjoint by construction, never empty.
    cfg["one_piece"] = w (optional): every mask is repaired to a single piece without holes, see `one_piece`."""
    lab0, area, peak, mean = components(prob, cfg["lo"])
    keep = keep_mask(area, peak, mean, cfg, radius(lab0, len(area)) if cfg.get("zones") is not None else None)
    lab, area_in = trim(prob, lab0, area, cfg["lo"], cfg.get("inner", 0.0))
    js = np.flatnonzero(keep & (area_in > 0))
    if not cfg.get("one_piece"): return [(lab == j, float(mean[j])) for j in js]
    from scipy import ndimage as ndi
    sls = ndi.find_objects(lab0); taken = np.zeros(prob.shape, bool); out = []
    for j in js:
        sl = sls[j - 1]; cs = (slice(max(sl[0].start - 4, 0), sl[0].stop + 4), slice(max(sl[1].start - 4, 0), sl[1].stop + 4))
        q = one_piece(lab[cs] == j, lab0[cs] == j, prob[cs], int(cfg["one_piece"])) & ~taken[cs]
        full = np.zeros(prob.shape, bool); full[cs] = q; taken[cs] |= q; out.append((full, float(mean[j])))
    return out


def frame_stats(prob, gt_sets, grid=GRID):
    """Per-frame PQ sufficient statistics for every keep-rule in `grid`.
    gt_sets: list (annotator records) of lists of flat pixel-index arrays (one per GT instance).
    Returns float array (len(grid), 4) = [sumIoU, TP, FP, FN] pooled over the frame's records."""
    out = np.zeros((len(grid), 4))
    for lo in sorted({c["lo"] for c in grid}):
        lab0, area, peak, mean = components(prob, lo)
        for inner in sorted({c.get("inner", 0.0) for c in grid if c["lo"] == lo}):
            lab, area_in = trim(prob, lab0, area, lo, inner)
            flat = lab.ravel()
            hits = []                                         # per record: (gt_i, comp_j, iou) with IoU > 0.5
            for gts in gt_sets:
                h = []
                for gi, gidx in enumerate(gts):
                    c = np.bincount(flat[gidx], minlength=len(area)); c[0] = 0
                    for j in np.flatnonzero(c):
                        iou = c[j] / (len(gidx) + area_in[j] - c[j])
                        if iou > 0.5: h.append((gi, j, iou))
                hits.append(h)
            for ci, cfg in enumerate(grid):
                if cfg["lo"] != lo or cfg.get("inner", 0.0) != inner: continue
                k = keep_mask(area, peak, mean, cfg) & (area_in > 0); nk = int(k.sum())
                for gts, h in zip(gt_sets, hits):
                    hk = [(gi, j, iou) for gi, j, iou in h if k[j]]
                    tp = len(hk)
                    out[ci] += (sum(i for _, _, i in hk), tp, nk - len({j for _, j, _ in hk}), len(gts) - len({g for g, _, _ in hk}))
    return out


def pq(acc):
    s, tp, fp, fn = acc
    return s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)


def _gather(rule, probs):
    """Components of every map that pass the rule's area thresholds: mean prob, zone, and their pixel probabilities."""
    lo = rule["lo"]; edges = [q[0] for q in rule["zones"]]; ar_z = np.array([q[2] for q in rule["zones"]])
    M, Z, PIX, nf = [], [], [], 0
    for prob in probs:
        nf += 1
        lab, area, peak, mean = components(prob, lo)
        z = np.minimum(np.searchsorted(edges, radius(lab, len(area)), side="right"), len(edges) - 1)
        ok = np.flatnonzero(area >= ar_z[z]); ok = ok[ok > 0]
        idx = np.flatnonzero(lab); l = lab.ravel()[idx]; pv = prob.ravel()[idx]
        o = np.argsort(l, kind="stable"); l, pv = l[o], pv[o]
        M.append(mean[ok]); Z.append(z[ok]); PIX += [pv[a:b] for a, b in zip(np.searchsorted(l, ok), np.searchsorted(l, ok, side="right"))]
    return np.concatenate(M), np.concatenate(Z), PIX, nf


def rule_stats(rule, probs):
    """What the rule keeps on these maps, without labels: (objects per frame, trimmed area per object)."""
    M, Z, PIX, nf = _gather(rule, probs)
    k = np.flatnonzero(M >= np.array([q[1] for q in rule["zones"]])[Z]); inner = max(rule["inner"], rule["lo"])
    return len(k) / nf, sum(int((PIX[j] > inner).sum()) for j in k) / max(len(k), 1)


def calibrate(rule, probs, target=None):
    """Transfer a keep-rule fitted on out-of-fold maps (one model per frame) to maps of a different sharpness
    (the test-time average of several models and D4 views). Uses no labels.

    Averaging changes the probability scale: component means drop where the models disagree, and the
    prob > inner core gets narrower because every model has to agree on a pixel. Thresholds taken as they are
    keep fewer and thinner objects than the rule was fitted to keep (measured on a clean 4-model hold-out:
    4.8 objects per frame instead of 6.5, PQ 0.425 instead of 0.447). We restore two statistics:
      1. objects kept per frame   -> one common shift of the mean-probability thresholds
      2. kept area per object     -> the trimming threshold `inner`
    target = (objects per frame, area per object). Best source: `rule_stats` of the unchanged rule on
    single-model maps of the SAME frames (paired, so the frame population cancels out). Default: the
    out-of-fold statistics stored in the rule. Area thresholds are scale-free and stay as they are.
    Returns (new rule, report dict)."""
    tgt_n, tgt_a = target or (rule["oof"]["kept_per_frame"], rule["oof"]["area_per_frame"] / rule["oof"]["kept_per_frame"])
    mn_z = np.array([q[1] for q in rule["zones"]]); lo = rule["lo"]
    M, Z, PIX, nf = _gather(rule, probs)
    deltas = np.linspace(-0.2, 0.2, 801)
    kept = np.array([(M >= mn_z[Z] - d).sum() / nf for d in deltas])
    d = float(deltas[int(np.abs(kept - tgt_n).argmin())])
    k = np.flatnonzero(M >= mn_z[Z] - d)
    pix = np.sort(np.concatenate([PIX[j] for j in k])) if len(k) else np.zeros(1)
    want = int(round(tgt_a * len(k)))                          # number of pixels the kept objects should have in total
    inner = max(float(pix[len(pix) - want]) if 0 < want < len(pix) else lo, lo)
    k0 = np.flatnonzero(M >= mn_z[Z])                          # what the unchanged rule keeps here (for the report)
    raw_n, raw_a = len(k0) / nf, sum(int((PIX[j] > max(rule["inner"], lo)).sum()) for j in k0) / max(len(k0), 1)
    new = dict(rule); new["inner"] = inner
    new["zones"] = [[q[0], float(q[1] - d) if q[1] < 1.5 else q[1], q[2]] for q in rule["zones"]]
    rep = dict(frames=nf, target_kept_per_frame=float(tgt_n), kept_per_frame_as_is=raw_n, mean_shift=d,
               target_area_per_object=float(tgt_a), area_per_object_as_is=float(raw_a), inner_from=rule["inner"], inner_to=inner)
    return new, rep
