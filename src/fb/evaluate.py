"""Score the production path on the out-of-fold maps: pq.instances with a fitted rule, exactly what submit.py runs.

Every train frame is predicted by the model that never saw it (oof.py), the instances are built by the same
function that writes the submission, and they are scored like the host does (every annotator record, IoU > 0.5,
pooled). Besides PQ this prints what the jury rubric asks for and what the report quotes:

  * distributions of IoU and Dice over the ground-truth segments;
  * one-to-many / many-to-one relations between ground-truth and predicted segments;
  * annotator agreement on the frames with three records: one annotator scored against another, our prediction
    scored against the same records, and how many segments we find depending on how many annotators drew them.

  python evaluate.py rules/final.json [out.npz]

Relations. A predicted segment and a ground-truth segment are 'related' when the smaller of the two lies at least
half inside the other. one-to-many: one ground-truth segment related to 2+ predictions (we split it);
many-to-one: one prediction related to 2+ ground-truth segments of the same record (we merged them).
"""
import os, sys, json, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import H, W, poly_masks
import pq as PQ, oof as OOF

RULE = json.load(open(sys.argv[1]))
files, names, fold = OOF.meta()


def iou(a, b):
    """IoU of two segments given as sorted flat pixel-index arrays."""
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter) if inter else 0.0


def pair_stats(P, G):
    """[sumIoU, TP, FP, FN] of segment set P scored against segment set G (IoU > 0.5)."""
    s = tp = 0.0; pm = set(); gm = set()
    for gi, g in enumerate(G):
        for pi, p in enumerate(P):
            v = iou(g, p)
            if v > 0.5: s += v; tp += 1; pm.add(pi); gm.add(gi)
    return np.array([s, tp, len(P) - len(pm), len(G) - len(gm)])


def _f(i):
    inst = PQ.instances(OOF.blend(RULE["maps"], RULE["weights"], i), RULE)
    lab = np.zeros((H, W), np.int32)
    for k, (m, _) in enumerate(inst, 1):
        assert not lab[m].any(), "instances overlap"; lab[m] = k
    n = len(inst) + 1; flat = lab.ravel(); area = np.bincount(flat, minlength=n).astype(np.float64); area[0] = 0
    pieces = [cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1 for m, _ in inst]
    G = [[np.flatnonzero(g) for g in poly_masks(segs)] for segs in files[names[i]]]
    acc = np.zeros(4); gt_rows = []; rel = np.zeros(4)          # rel: GT units, one-to-many GT, prediction units, many-to-one predictions
    found = []                                                  # per record: was each ground-truth segment matched by us
    for rec in G:
        hit = set(); nrel_pred = np.zeros(n, int); f = []
        for gi in rec:
            c = np.bincount(flat[gi], minlength=n); c[0] = 0; j = np.flatnonzero(c)
            v = c[j] / (len(gi) + area[j] - c[j]); h = v > 0.5
            acc[0] += v[h].sum(); acc[1] += h.sum(); acc[3] += not h.any(); hit.update(j[h].tolist()); f.append(bool(h.any()))
            best = float(v.max()) if len(j) else 0.0
            r = j[c[j] >= 0.5 * np.minimum(area[j], len(gi))]; nrel_pred[r] += 1
            gt_rows.append((len(gi), best, 2 * best / (1 + best), len(r)))
            rel[0] += 1; rel[1] += len(r) >= 2
        acc[2] += len(inst) - len(hit); rel[2] += len(inst); rel[3] += (nrel_pred >= 2).sum(); found.append(f)
    ann = np.zeros(4); agree = []                               # three-record frames: annotator vs annotator, agreement per segment
    if len(G) == 3:
        for a in range(3):
            for b in range(3):
                if a != b: ann += pair_stats(G[a], G[b])
            for gi, g in enumerate(G[a]):
                others = sum(any(iou(g, o) > 0.5 for o in G[b]) for b in range(3) if b != a)
                agree.append((others, found[a][gi]))
    return acc, np.array(gt_rows, np.float64).reshape(-1, 4), rel, pieces, len(G), ann, np.array(agree, np.float64).reshape(-1, 2)


if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(_f, range(len(names)), chunksize=4))
    acc = sum(r[0] for r in res); s, tp, fp, fn = acc
    print(f"rule {sys.argv[1]} (one_piece = {RULE.get('one_piece', 0)}), {len(names)} out-of-fold frames, {int(sum(len(files[n]) for n in names))} annotator records")
    print(f"PQ {PQ.pq(acc):.4f} = SQ {s / tp:.4f} x RQ {tp / (tp + 0.5 * fp + 0.5 * fn):.4f};  TP {tp:.0f}  FP {fp:.0f}  FN {fn:.0f}")
    print("per fold: " + "  ".join(f"{PQ.pq(sum(res[i][0] for i in np.flatnonzero(fold == k))):.4f}" for k in range(5)))
    G = np.concatenate([r[1] for r in res]); rel = sum(r[2] for r in res); pc = np.concatenate([r[3] for r in res])
    print(f"predicted objects {len(pc)} ({len(pc) / len(names):.2f} per frame); with 2+ pieces: {(pc >= 2).mean():.1%}")
    edges = (0, 1e-9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)
    for nm, col in (("IoU", 1), ("Dice", 2)):
        hst = np.histogram(G[:, col], edges)[0] / len(G)
        print(f"best {nm} of every ground-truth segment ({len(G)}): none {hst[0]:.1%} | " + "  ".join(f"{a:.1f}-{min(b, 1.0):.1f}: {v:.1%}" for a, b, v in zip(edges[1:-1], edges[2:], hst[1:]))
              + f" | mean {G[:, col].mean():.3f}, median {np.median(G[:, col]):.3f}, mean over matched {G[G[:, 1] > 0.5, col].mean():.3f}")
    print(f"one-to-many (a ground-truth segment related to 2+ predictions): {rel[1]:.0f} of {rel[0]:.0f} ground-truth segments ({rel[1] / rel[0]:.1%})")
    print(f"many-to-one (a prediction related to 2+ ground-truth segments of one record): {rel[3]:.0f} of {rel[2]:.0f} prediction x record units ({rel[3] / rel[2]:.1%})")
    three = [r for r in res if r[4] == 3]; ann = sum(r[5] for r in three); mo = sum(r[0] for r in three); A = np.concatenate([r[6] for r in three])
    print(f"frames with three annotator records ({len(three)}): one annotator against another PQ {PQ.pq(ann):.4f} (SQ {ann[0] / ann[1]:.3f}); "
          f"our prediction against the same records PQ {PQ.pq(mo):.4f} (SQ {mo[0] / mo[1]:.3f})")
    for k, nm in ((2, "all three annotators"), (1, "two annotators"), (0, "one annotator only")):
        m = A[:, 0] == k; print(f"  segments drawn by {nm:20s}: {m.mean():.1%} of the segments on these frames, we find {A[m, 1].mean():.1%}")
    if len(sys.argv) > 2: np.savez(sys.argv[2], acc=acc, gt=G, rel=rel, pieces=pc, annotators=ann, agree=A)
