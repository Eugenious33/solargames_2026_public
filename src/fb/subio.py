"""Submission CSV writer + strict validator + agreement between two submission files."""
import os, re, csv, numpy as np
from pycocotools import mask as mu
from data import DATA, H, W


def rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def write_and_validate(rows, out):
    """rows: list of (filament_id, rle string). Writes the CSV, re-reads it from disk and checks every rule
    the host scorer relies on. Raises AssertionError on any violation."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write("filament_id,segmentation_rle\n")
        for fid, r in rows: f.write(f"{fid},{r}\n")
    test_stems = {f.rsplit(".", 1)[0] for f in os.listdir(f"{DATA}/test/test_images")}
    lines = open(out).read().split("\n")
    assert lines[0] == "filament_id,segmentation_rle" and lines[-1] == "", "header / trailing newline"
    seen, cover, areas = set(), {}, []
    for ln in lines[1:-1]:
        assert '"' not in ln and ln.count(",") == 1, f"quoting/commas: {ln[:40]}"
        fid, r = ln.split(",")
        assert re.fullmatch(r"\d{14}[A-Z][a-z]_\d+", fid), fid
        assert fid not in seen, f"duplicate id {fid}"; seen.add(fid)
        stem = fid.split("_")[0]; assert stem in test_stems, stem
        assert all(48 <= ord(c) <= 111 for c in r), "RLE charset"
        m = mu.decode({"size": [H, W], "counts": r.encode("ascii")})
        assert m.shape == (H, W) and m.any(), f"empty/invalid mask {fid}"      # an empty RLE is a guaranteed FP
        c = cover.setdefault(stem, np.zeros((H, W), np.uint8)); c += m
        assert c.max() <= 1, f"overlapping masks in {stem}"
        areas.append(int(m.sum()))
    per = np.array([sum(1 for s in seen if s.startswith(st)) for st in sorted(test_stems)])
    print(f"VALID {out}: {len(seen)} instances on {int((per > 0).sum())}/{len(test_stems)} frames; per frame median {np.median(per):.0f} "
          f"mean {per.mean():.1f} max {per.max()}; area p5/p50/p95 {np.percentile(areas, [5, 50, 95]).round(0)}")


def agreement(out, ref):
    """PQ between two submission files (`ref` plays the ground truth). Used to check that a re-run of the pipeline
    reproduces a submitted file: identical files give 1.0."""
    def load(path):
        d = {}
        for r in csv.DictReader(open(path)):
            d.setdefault(r["filament_id"].split("_")[0], []).append(mu.decode({"size": [H, W], "counts": r["segmentation_rle"].encode("ascii")}).astype(bool))
        return d
    P, G = load(out), load(ref); s = tp = 0.0; npred = sum(len(v) for v in P.values()); ngt = sum(len(v) for v in G.values())
    for stem, gts in G.items():
        for g in gts:
            for m in P.get(stem, []):
                inter = np.count_nonzero(m & g)
                if inter and inter / (m.sum() + g.sum() - inter) > 0.5: s += inter / (m.sum() + g.sum() - inter); tp += 1
    v = s / max(tp + 0.5 * (npred - tp) + 0.5 * (ngt - tp), 1e-9)
    print(f"agreement of {out} with {ref}: PQ {v:.4f}; matched {tp:.0f} of {npred} / {ngt} masks")
    return v
