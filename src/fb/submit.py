"""Probability maps -> pixel-disjoint instances -> Kaggle CSV, followed by a strict format validation.

Two ways to give the keep-rule:
  --cfg lo,peak,mean,area[,inner]      one uniform rule, applied as is (how the first submissions were made)
  --rule fb/rules/x.json               rule fitted by oof.py (blend weights, trimming, limb zones), transferred to
                                       the test-time ensemble with the label-free calibration of pq.calibrate

  python submit.py --tag ens1 --cfg 0.4,0.9,0.85,400 --out sub/ens1.csv
  python submit.py --rule fb/rules/ud.json --ens strong5,detf0+detf1+detf2+detf3+detf4 \
                   --single "uf0,uf1,uf2,uf3,uf4;detf0,detf1,detf2,detf3,detf4" --out sub/ud.csv

--ens     one entry per map family of the rule, in the rule's order; `a+b+c` = plain average of saved test maps
--scale   n,a: multiply the paired target by these factors (leaderboard probes of object count / mask width)
--single  per family, the single-model test maps (same TTA as the out-of-fold maps). The unchanged rule is applied
          to blend k = family maps number k; the average count / area it keeps is the calibration target.
          Without --single the out-of-fold statistics stored in the rule are the target (--calib oof), or the
          rule is used unchanged (--calib none).
Rules enforced (host scorer): header, unique ids `<stem>_<k>`, stems in test,
RLE decodes at 2048x2048, no empty masks (an empty RLE is a guaranteed FP), no overlaps, no quoting.
"""
import argparse, os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, CACHE
import pq as PQ
from subio import rle, write_and_validate, agreement

p = argparse.ArgumentParser()
p.add_argument("--tag", default=""); p.add_argument("--cfg", default="", help="lo,peak,mean,area[,inner]")
p.add_argument("--rule", default=""); p.add_argument("--ens", default=""); p.add_argument("--single", default="")
p.add_argument("--calib", default="paired", choices=["paired", "oof", "none"])
p.add_argument("--scale", default="1,1", help="multiply the calibration target (objects per frame, area per object); 1,1.15 = same objects, 15%% more area (width probe)")
p.add_argument("--out", required=True); p.add_argument("--ref", default="", help="another submission CSV to compare the result with (reproduction check)")
a = p.parse_args()


def load(tag):
    names = json.load(open(f"{CACHE}/pred/{tag}_test.json"))["names"]
    return names, np.load(f"{CACHE}/pred/{tag}_test.npy", mmap_mode="r")


def family(entry):
    """`a+b+c` -> (names, list of memmaps)."""
    got = [load(t) for t in entry.split("+")]
    assert all(g[0] == got[0][0] for g in got), f"frame order differs inside {entry}"
    return got[0][0], [g[1] for g in got]


if a.rule:
    rule = json.load(open(a.rule)); w = rule["weights"]
    fams = [family(e) for e in a.ens.split(",")]; names = fams[0][0]
    assert len(fams) == len(rule["maps"]) and all(f[0] == names for f in fams), "--ens must list one entry per map family of the rule"

    def ens(i):
        return sum(wk * np.mean([m[i] for m in f[1]], 0) for wk, f in zip(w, fams)).astype(np.float32) / (255 * sum(w))

    if a.calib == "none":
        cal, rep = rule, None
    else:
        target = None
        if a.calib == "paired":
            sing = [[load(t) for t in fam.split(",")] for fam in a.single.split(";")]
            assert len(sing) == len(w) and len({len(s) for s in sing}) == 1 and all(s[0] == names for f in sing for s in f), "--single: same number of maps per family"
            st = [PQ.rule_stats(rule, (sum(wk * f[k][1][i].astype(np.float32) for wk, f in zip(w, sing)) / (255 * sum(w)) for i in range(len(names))))
                  for k in range(len(sing[0]))]
            target = tuple(np.mean(st, 0) * np.array([float(v) for v in a.scale.split(",")]))
            print("single-model blends, rule as is: " + "  ".join(f"[{n:.2f}/frame, {ar:.0f} px]" for n, ar in st), flush=True)
        cal, rep = PQ.calibrate(rule, (ens(i) for i in range(len(names))), target)
        print(f"calibration ({a.calib}): the unchanged rule would keep {rep['kept_per_frame_as_is']:.2f} objects/frame of {rep['area_per_object_as_is']:.0f} px; "
              f"target {rep['target_kept_per_frame']:.2f} of {rep['target_area_per_object']:.0f} px -> mean thresholds shifted by {-rep['mean_shift']:+.3f}, "
              f"inner {rep['inner_from']} -> {rep['inner_to']:.3f}", flush=True)
    cfg = cal; prob = ens
    json.dump(dict(rule=a.rule, ens=a.ens, single=a.single, calib=a.calib, report=rep, applied=cal), open(a.out.rsplit(".", 1)[0] + ".rule.json", "w"), indent=1)
else:
    v = [float(x) for x in a.cfg.split(",")]
    cfg = dict(lo=v[0], peak=v[1], mean=v[2], area=v[3], inner=v[4] if len(v) > 4 else 0.0)
    names, P = load(a.tag)
    prob = lambda i: P[i].astype(np.float32) / 255

rows = []
for i, n in enumerate(names):
    inst = PQ.instances(prob(i), cfg)
    inst.sort(key=lambda t: -t[1])
    rows += [(f"{n.rsplit('.', 1)[0]}_{k}", rle(m)) for k, (m, _) in enumerate(inst, 1)]
write_and_validate(rows, a.out)
if a.ref: agreement(a.out, a.ref)
