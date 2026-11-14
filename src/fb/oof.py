"""Out-of-fold probability maps on all 707 train frames: assemble them, score blends, fit the final keep-rule.

Every frame is predicted by the one model that never saw it, so numbers measured here are honest for the
whole training set (not just one 141-frame hold-out).

  python oof.py assemble --tag unet_s --runs f0run,f1run,f2run,f3run,f4run     # run k must have held out fold k
  python oof.py score --maps unet_s                                            # one map, uniform keep-rule grid
  python oof.py score --maps unet_s,det --weights 0.5,0.5                      # weighted average of several maps
  python oof.py rule  --maps unet_s,det --weights 0.7,0.3 --out fb/rules/ud.json   # + limb zones, writes the rule

`score` reports the best keep-rule on all frames and the honest number: the rule is chosen on four folds
and applied to the fifth, for every fold, then pooled.
`rule` additionally lets the thresholds depend on the distance from disk centre (experts label far fewer
filaments near the limb), reports uniform vs zoned honestly, and writes the rule that submit.py applies.
"""
import argparse, os, sys, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, CACHE, H, W, load_sets, split_names, poly_masks
import pq as PQ

NFOLD = 5
D = f"{CACHE}/oof"
# zone rule search space
MS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 2.0)    # 2.0 = drop the zone
AS = (100, 150, 200, 300, 400, 600, 800)
EDGES = ((), (0.85,), (0.85, 0.92), (0.80, 0.90), (0.75, 0.85, 0.92))
C_AREA, C_MEAN, C_R, C_NREC, C_S, C_T, C_AIN = 3, 6, 7, 8, 9, 10, 4


def meta():
    files = load_sets(); names = json.load(open(f"{CACHE}/train_meta.json"))["names"]
    fold_of = {n: k for k in range(NFOLD) for n in split_names(files, fold=k)[1]}
    return files, names, np.array([fold_of[n] for n in names])


def assemble(tag, runs):
    files, names, _ = meta(); pos = {n: i for i, n in enumerate(names)}
    os.makedirs(D, exist_ok=True)
    out = np.lib.format.open_memmap(f"{D}/{tag}.npy", "w+", np.uint8, (len(names), H, W))
    for k, r in enumerate(runs):
        P = np.load(f"{CACHE}/runs/{r}/val_probs.npy", mmap_mode="r")
        vn = split_names(files, fold=k)[1]; assert len(vn) == len(P), (r, len(vn), len(P))
        f = f"{CACHE}/runs/{r}/val_names.json"
        if os.path.exists(f): assert json.load(open(f)) == vn, f"{r} is not fold {k}"
        for j, n in enumerate(vn): out[pos[n]] = P[j]
    out.flush(); json.dump({"runs": runs}, open(f"{D}/{tag}.json", "w"))
    print(f"{D}/{tag}.npy: {len(names)} frames from {runs}", flush=True)


def blend(maps, weights, i):
    if len(maps) == 1: return np.load(f"{D}/{maps[0]}.npy", mmap_mode="r")[i].astype(np.float32) / 255
    return sum(w * np.load(f"{D}/{m}.npy", mmap_mode="r")[i].astype(np.float32) for m, w in zip(maps, weights)) / (255 * sum(weights))


def _score(args):
    i, maps, weights, segs = args
    gts = [[np.flatnonzero(m) for m in poly_masks(s)] for s in segs]
    return PQ.frame_stats(blend(maps, weights, i), gts)


def score(maps, weights, verbose=True):
    files, names, fold = meta()
    with ProcessPoolExecutor(min(20, os.cpu_count() or 8)) as ex:
        st = np.stack(list(ex.map(_score, [(i, maps, weights, files[n]) for i, n in enumerate(names)], chunksize=4)))
    acc = st.sum(0); pqs = np.array([PQ.pq(r) for r in acc]); j = int(pqs.argmax())
    cross = np.zeros(4)
    for k in range(NFOLD):                                     # rule from the other four folds
        jk = int(np.argmax([PQ.pq(r) for r in st[fold != k].sum(0)])); cross += st[fold == k].sum(0)[jk]
    res = dict(maps=maps, weights=weights, pq_best=float(pqs[j]), cfg=PQ.GRID[j], pq_honest=float(PQ.pq(cross)),
               acc=[float(v) for v in acc[j]], pq_no_trim=float(max(p for p, c in zip(pqs, PQ.GRID) if c["inner"] == 0)))
    if verbose:
        s, tp, fp, fn = acc[j]
        print(f"{'+'.join(f'{w:g}*{m}' for m, w in zip(maps, weights)):40s} PQ best {pqs[j]:.4f}  honest {res['pq_honest']:.4f}  (no trim: {res['pq_no_trim']:.4f})"
              f"  TP {tp:.0f} FP {fp:.0f} FN {fn:.0f} meanIoU {s/max(tp,1):.3f}  {PQ.GRID[j]}", flush=True)
    return res, pqs


# ----------------------------------------------------------------------------- component table + zone rule
def frame_table(prob, segs, pairs, i=0):
    """One row per candidate component: [frame, pair, -, area, area_in, peak, mean, r/R, n_rec, sum matched IoU, n matched].
    pairs = list of (lo, inner). Also returns the number of GT instances pooled over annotator records."""
    gts = [np.flatnonzero(m) for s in segs for m in poly_masks(s)]
    rows = []
    for pi, (lo, inner) in enumerate(pairs):
        lab0, area, peak, mean = PQ.components(prob, lo)
        n = len(area); r = PQ.radius(lab0, n)
        lab, area_in = PQ.trim(prob, lab0, area, lo, inner); flat = lab.ravel()
        s = np.zeros(n); t = np.zeros(n)
        for g in gts:
            c = np.bincount(flat[g], minlength=n); c[0] = 0
            for j in np.flatnonzero(c):
                iou = c[j] / (len(g) + area_in[j] - c[j])
                if iou > 0.5: s[j] += iou; t[j] += 1
        for j in np.flatnonzero((area >= AS[0]) & (area_in > 0)):          # area[0] (background) is 0, so j >= 1
            rows.append((i, pi, 0, area[j], area_in[j], peak[j], mean[j], r[j], len(segs), s[j], t[j]))
    return np.array(rows, np.float64).reshape(-1, 11), len(gts)


def _table(args):
    i, maps, weights, segs, pairs = args
    return frame_table(blend(maps, weights, i), segs, pairs, i)


def _tabs(T, edges):
    """Per zone: sums [S, TP, FP] for every (mean, area) threshold pair -> array (zones, len(MS), len(AS), 3)."""
    zone = np.searchsorted(edges, T[:, C_R], side="right") if edges else np.zeros(len(T), int)
    out = np.zeros((len(edges) + 1, len(MS), len(AS), 3))
    for z in range(len(edges) + 1):
        Tz = T[zone == z]
        for mi, m in enumerate(MS):
            for ai, a_ in enumerate(AS):
                k = (Tz[:, C_MEAN] >= m) & (Tz[:, C_AREA] >= a_)
                out[z, mi, ai] = (Tz[k, C_S].sum(), Tz[k, C_T].sum(), (Tz[k, C_NREC] - Tz[k, C_T]).sum())
    return out


def _pq3(v, ngt):
    return v[..., 0] / np.maximum(v[..., 1] + 0.5 * v[..., 2] + 0.5 * (ngt - v[..., 1]), 1e-9)


def fit_zones(T, ngt, edges):
    """Coordinate ascent over the (mean, area) thresholds of every zone. -> ([(mean, area) per zone], pooled PQ)."""
    tabs = _tabs(T, edges); nz = len(tabs)
    choice = [(MS.index(0.9), AS.index(200))] * nz
    for _ in range(4):
        for z in range(nz):
            other = sum((tabs[y][choice[y]] for y in range(nz) if y != z), np.zeros(3))
            choice[z] = np.unravel_index(int(_pq3(tabs[z] + other, ngt).argmax()), tabs[z].shape[:2])
    return [(MS[c[0]], AS[c[1]]) for c in choice], float(_pq3(sum(tabs[z][choice[z]] for z in range(nz)), ngt))


def apply_zones(T, edges, params):
    zone = np.searchsorted(edges, T[:, C_R], side="right") if edges else np.zeros(len(T), int)
    mn = np.array([p_[0] for p_ in params])[zone]; ar = np.array([p_[1] for p_ in params])[zone]
    return (T[:, C_MEAN] >= mn) & (T[:, C_AREA] >= ar)


def best_rule(T, ngt, pairs, edge_sets):
    best = (-1,)
    for pi in range(len(pairs)):
        Tp = T[T[:, 1] == pi]
        for e in edge_sets:
            params, v = fit_zones(Tp, ngt, e)
            if v > best[0]: best = (v, pi, e, params)
    return best


def rule(maps, weights, out, top=3):
    files, names, fold = meta()
    res, pqs = score(maps, weights)
    order = np.argsort(-pqs); pairs = []
    for j in order:                                            # the best few (lo, inner) combinations of the uniform grid
        pr = (PQ.GRID[j]["lo"], PQ.GRID[j]["inner"])
        if pr not in pairs: pairs.append(pr)
        if len(pairs) == top: break
    with ProcessPoolExecutor(min(20, os.cpu_count() or 8)) as ex:
        r = list(ex.map(_table, [(i, maps, weights, files[n], pairs) for i, n in enumerate(names)], chunksize=4))
    T = np.concatenate([x[0] for x in r]); ngt_f = np.array([x[1] for x in r]); rf = fold[T[:, 0].astype(int)]
    honest = {}
    for fam, es in (("uniform", EDGES[:1]), ("zones", EDGES)):
        acc = np.zeros(3); n = 0
        for k in range(NFOLD):
            _, pi, e, params = best_rule(T[rf != k], ngt_f[fold != k].sum(), pairs, es)
            Tk = T[(rf == k) & (T[:, 1] == pi)]; kp = apply_zones(Tk, e, params)
            acc += (Tk[kp, C_S].sum(), Tk[kp, C_T].sum(), (Tk[kp, C_NREC] - Tk[kp, C_T]).sum())
        honest[fam] = float(_pq3(acc, ngt_f.sum()))
    v, pi, e, params = best_rule(T, ngt_f.sum(), pairs, EDGES)
    vu, piu, _, pu = best_rule(T, ngt_f.sum(), pairs, EDGES[:1])
    Tp = T[T[:, 1] == pi]; kp = apply_zones(Tp, e, params)
    zones = [[float(hi), float(m), float(a_)] for hi, (m, a_) in zip(list(e) + [9.0], params)]
    rl = dict(maps=maps, weights=weights, lo=pairs[pi][0], inner=pairs[pi][1], peak=0.0, zones=zones,
              oof=dict(pq_fit=v, pq_honest_zones=honest["zones"], pq_honest_uniform=honest["uniform"], pq_fit_uniform=vu,
                       uniform=dict(lo=pairs[piu][0], inner=pairs[piu][1], mean=pu[0][0], area=pu[0][1]),
                       kept_per_frame=float(kp.sum() / len(names)), area_per_frame=float(Tp[kp, C_AIN].sum() / len(names)),
                       kept_by_zone=[float(((Tp[kp, C_R] >= lo_) & (Tp[kp, C_R] < hi_)).sum() / len(names)) for lo_, hi_ in zip([0.0] + list(e), list(e) + [9.0])]))
    print(f"uniform rule : fit {vu:.4f}  honest {honest['uniform']:.4f}   lo {pairs[piu][0]} inner {pairs[piu][1]} mean>={pu[0][0]} area>={pu[0][1]}")
    print(f"zoned rule   : fit {v:.4f}  honest {honest['zones']:.4f}   lo {rl['lo']} inner {rl['inner']} zones [r_hi, mean, area] {zones}")
    print(f"kept per frame {rl['oof']['kept_per_frame']:.2f} (by zone {[round(x, 2) for x in rl['oof']['kept_by_zone']]}), kept area per frame {rl['oof']['area_per_frame']:.0f} px")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); json.dump(rl, open(out, "w"), indent=1); print("rule ->", out)
    return rl


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["assemble", "score", "rule"])
    p.add_argument("--tag", default=""); p.add_argument("--runs", default="")
    p.add_argument("--maps", default=""); p.add_argument("--weights", default=""); p.add_argument("--out", default="")
    a = p.parse_args()
    if a.cmd == "assemble":
        assemble(a.tag, a.runs.split(","))
    else:
        maps = a.maps.split(",")
        ws = [[float(v) for v in s.split(",")] for s in a.weights.split(";")] if a.weights else [[1.0] * len(maps)]
        if a.cmd == "score":
            for w in ws: score(maps, w)
        else:                                                  # several weight sets (`;`-separated): keep the honestly best one
            rules = [rule(maps, w, "") for w in ws]
            best = max(rules, key=lambda r: r["oof"]["pq_honest_zones"])
            if len(rules) > 1: print("best weights by honest PQ:", best["weights"], f"{best['oof']['pq_honest_zones']:.4f}")
            if a.out:
                os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(best, open(a.out, "w"), indent=1); print("rule ->", a.out)
