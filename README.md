# Solar filament instance segmentation on GONG H-alpha frames

Code, weights and a demo notebook for our solution to the IEEE BigData Cup 2026 challenge
*Filament Segmentation 2026* (MAGFiLO dataset). The repository reproduces the submitted predictions from the
stored weights, without retraining.

| Result | Panoptic Quality |
|---|---|
| Out-of-fold on the 707 training frames (1154 annotator records) | **0.464** (SQ 0.686 x RQ 0.677) |
| The same without the last repair step, thresholds fitted on four folds and applied to the fifth | 0.459 |
| Public leaderboard | 0.41 |
| For scale: one annotator against another annotator of the same frame | 0.33 |

Two files were submitted as finals; this repository reproduces both. `submission/submission.csv` (public LB 0.41) is the
final solution described below. `submission/first_final.csv` (public LB 0.40) is our first final: a hard and a soft
U-Net with the same ResNet-34 encoder plus Mask R-CNN (rule `rules/first_final.json`, weights `unet_hard_f*`,
`unet_soft_f*`); the commands are at the end of this file.

## Quick start

1. Unpack the competition data so that `data/MAGFiLO_1.0_Kaggle_2026/{train,test}/` exists.
2. Create the environment: either `pip install -r requirements.txt` (Python 3.11, CUDA GPU with 8 GB or more),
   or build the image: `docker/build.sh`.
3. Run `demo.ipynb` from the repository root, for example

   ```bash
   docker/run.sh jupyter nbconvert --to notebook --execute --inplace demo.ipynb
   ```

The notebook preprocesses the test frames, runs the 20 stored networks, post-processes the maps, writes
`submission/reproduced.csv`, validates it and compares it with the submitted file `submission/submission.csv`.
It then re-predicts the held-out training frames and prints the out-of-fold evaluation. Training is included as an
optional step (`RETRAIN = True`).

## The pipeline

```text
8-bit frame 2048 x 2048
  -> limb fit, division by the centre-to-limb profile, robust z-score          src/fb/disk.py, data.py
  -> four model families, five folds each, native resolution, no tiling
       unet_r34      U-Net / ResNet-34,         target = mean of annotator records   src/fb/model.py, train.py, infer.py
       unet_effv2s   U-Net / EfficientNetV2-S,  same target
       unet_pvtv2    U-Net / PVTv2-B2 (vision transformer), same target
       maskrcnn      Mask R-CNN / ResNet-50 FPN v2 on three filter channels       src/det/maskrcnn.py, src/fb/filters.py
  -> weighted average of the four probability maps (0.233 / 0.233 / 0.233 / 0.30)
  -> objects = connected components of prob > 0.2
  -> keep / drop by mean probability and area, thresholds depend on r/R         src/fb/pq.py
  -> outline = pixels of the object with prob > 0.5, repaired to one piece
  -> thresholds transferred to the ensemble without labels (calibration)
  -> CSV with RLE masks, validated                                              src/fb/submit.py, subio.py
```

Why it looks like this:

- **Different encoders, not more models.** Five models with one encoder average out their noise, but more models of
  the same kind add little. Encoders of different design make different mistakes: out-of-fold PQ is 0.452 with one
  encoder, 0.460 with two, 0.464 with three. The transformer (PVTv2) differs most from the two convolutional
  encoders. A fourth encoder (ConvNeXt, HRNet, HGNetV2, Twins-SVT, DaViT, TinyViT or NextViT) added at most 0.001.
- **Annotators disagree.** A frame has one to three independent annotator records and a prediction is scored against
  each of them. Two annotators of the same frame agree with PQ 0.33. The soft target (mean of the records) lets the
  network output how likely a pixel is to be marked, and the keep / drop rule then keeps an object only when its
  expected matched IoU pays for the half point it costs in the denominator if it is not matched.
- **Thin objects.** A median filament is about 11 px wide. One pixel of extra width costs several points of IoU, so
  everything runs at the native resolution and the outline is taken from the confident core of each object.
- **The limb.** Annotators mark far fewer filaments near the limb, so the thresholds depend on the distance from
  disk centre.
- **One piece.** An annotated filament is one piece without holes. Our masks are repaired to obey the same rule.

## What we tried and left out

Every idea was tested against the same hold-out protocol (thresholds fitted without the scored frames) before it could
enter the pipeline. The code of the main lines is kept in the repository.

| Idea | Result | Decision |
|---|---|---|
| Flat-field: divide the frame by its centre-to-limb profile | pixel AP of brightness alone 0.011 -> 0.171 | in the pipeline (`disk.py`, `data.py`) |
| Classical filter bank as extra U-Net input channels (`filters.py`): Hessian dark ridge, starlet wavelet planes, elongated even Gabor filters over orientations, Frangi vesselness, local contrast, black top-hat | Strong alone (pixel AP per channel: dark ridge 0.335, starlet 0.313, Gabor 0.307), but a U-Net with 8 filter channels scores PQ 0.406 vs 0.421 on the plain frame (same 6k-step recipe): the network learns these features itself | dropped for the U-Nets; three of them (frame, starlet band-pass, dark ridge) form the input of Mask R-CNN, where a different input adds diversity to the blend |
| Learnable Fourier front-end (`spectral.py`, `--lsf 1`): log-Gabor-like isotropic and oriented transfer functions with free radial profiles, learned in the frequency domain | PQ 0.416 vs 0.421; the learned pass-bands barely moved (periods 11 / 21 / 49 / 91 px from the initial 12 / 24 / 48 / 96). With the filter bank and a learnable per-pixel mixer (`--mix 1`), long training: 0.433 vs 0.431 | dropped |
| More data: pseudo-labelled frames from the public GONG archive (tested only, not used) | learning curve: +0.008-0.010 per doubling of the training set; the archive FITS frames look different from the competition JPEGs (the same network scores 0.357 on them vs 0.406) | dropped |
| Larger network (ResNeXt-101, 4x parameters) | same PQ; no gap between training and unseen frames for any model | dropped |
| A fourth encoder (ConvNeXt, HRNet, HGNetV2, Twins-SVT, DaViT, TinyViT, NextViT) | at most +0.001 in the blend, although several are stronger alone | dropped |
| SAM 2.1 as a contour refiner | 0.451 vs 0.456 | dropped |
| Network conditioned on the annotator (FiLM) | 0.4485 vs 0.4510 even with the annotator known | dropped |
| Re-labelling the training set with our own network | -0.007 | dropped |
| Gap network for touching filaments: a U-Net trained on the gap between neighbouring objects of each record, used to cut the blend | it finds the gaps (pixel AP 0.25, 670x chance), but every way of cutting loses (-0.004 / -0.001): the annotators themselves disagree whether two pieces are one filament | dropped |

## Repository layout

```text
demo.ipynb             the whole pipeline, step by step
requirements.txt       exact package versions
src/fb/                preprocessing, U-Net training and inference, post-processing, evaluation
src/det/maskrcnn.py    Mask R-CNN member
rules/final.json       fitted post-processing rule (blend weights, thresholds, rings)
rules/first_final.json the rule of the first final submission
weights/               unet_r34_f0..4, unet_effv2s_f0..4, unet_pvtv2_f0..4 (ema.pt), maskrcnn_f0..4 (model.pt);
                       unet_hard_f0..4, unet_soft_f0..4 for the first final
submission/            the files submitted to Kaggle and the calibrated rules they were made with
docker/                Dockerfile, build.sh, run.sh
```

Every module starts with a description of what it does and how to run it. Paths come from two optional environment
variables: `SOLAR_ROOT` (default: the repository) and `SOLAR_CACHE` (default: `~/.cache/solar2026`).

The weights are stored as float16 (45-56 MB per U-Net, 92 MB per Mask R-CNN) and cast to float32 on load. The first
final was produced from exactly these files and is reproduced mask for mask. The final solution was produced from the
float32 originals; the float16 copies reproduce it with PQ 0.999 between the two files (1178 of 1179 masks matched,
one extra mask).

## Step by step

```bash
python src/fb/data.py test                                   # cache the preprocessed test frames
python src/fb/infer.py --runs unet_r34_f0 --tta hflip --tag r34_f0_h      # single models (calibration targets) ...
python src/fb/infer.py --runs unet_r34_f0,unet_r34_f1,unet_r34_f2,unet_r34_f3,unet_r34_f4 --tta d4 --tag r34_5
                                                             # ... the same for unet_effv2s (eff_*), unet_pvtv2 (pvt_*)
python src/det/maskrcnn.py prep --only test                  # three-channel composites
python src/det/maskrcnn.py test --name maskrcnn_f0           # ... for every fold
python src/fb/submit.py --rule rules/final.json --ens r34_5,eff_5,pvt_5,maskrcnn_f0+...+maskrcnn_f4 \
       --single "r34_f0_h,...;eff_f0_h,...;pvt_f0_h,...;maskrcnn_f0,..." --out submission/reproduced.csv --ref submission/submission.csv
```

The notebook contains the complete command list.

## Training from scratch

```bash
python src/fb/data.py                     # frames and targets of both splits (27 GB of cache)
python src/det/maskrcnn.py prep
for k in 0 1 2 3 4; do
  common="--fold $k --aug strong --target tsoftc --steps 18000"
  python src/fb/train.py --name unet_r34_f$k    $common
  python src/fb/train.py --name unet_effv2s_f$k $common --enc tu-tf_efficientnetv2_s --lr 3e-4
  python src/fb/train.py --name unet_pvtv2_f$k  $common --enc tu-pvt_v2_b2 --lr 3e-4 --enc_lr_k 0.3
  python src/det/maskrcnn.py train --name maskrcnn_f$k --fold $k
done
for f in unet_r34 unet_effv2s unet_pvtv2 maskrcnn; do
  python src/fb/oof.py assemble --tag $f --runs ${f}_f0,${f}_f1,${f}_f2,${f}_f3,${f}_f4
done
python src/fb/oof.py rule --maps unet_r34,unet_effv2s,unet_pvtv2,maskrcnn --weights 0.2333,0.2333,0.2333,0.3 --out rules/refit.json
```

On one RTX 4090: about 33 minutes per `unet_r34` model, 40 per `unet_effv2s`, 51 per `unet_pvtv2` and 50 per
Mask R-CNN; about 15 GPU-hours in total. The U-Nets use the exact host raster of the annotations (`tsoftc`). The
transformer encoder is fine-tuned at 0.3 x the decoder's learning rate (`--enc_lr_k 0.3`).

## Evaluation

`python src/fb/evaluate.py rules/final.json` scores the out-of-fold maps through the same function that writes the
submission and prints PQ, SQ, RQ, the distributions of IoU and Dice over ground-truth segments, the one-to-many /
many-to-one relations and the agreement between annotators on the frames with three records. Out-of-fold means that every training frame is predicted by the fold model that never saw
it; the split is by file, so all records of a frame stay in the same fold.

| Out-of-fold, 8199 ground-truth segments | |
|---|---|
| matched (IoU > 0.5) | 65.1% |
| mean IoU / Dice over matched segments | 0.686 / 0.810 |
| ground-truth segments related to 2+ predictions (one-to-many) | 2.2% |
| predictions related to 2+ ground-truth segments of one record (many-to-one) | 3.4% |
| predicted masks consisting of more than one piece | 0% |

## Run time

Measured on one RTX 4090 (24 GB), from an empty cache.

| Step | Time |
|---|---|
| Reproduce the submission: preprocessing, 20 networks with test-time augmentation, post-processing (180 frames) | 40 min (13 s per frame), 20 GB of cache |
| Out-of-fold maps of the 707 training frames, their evaluation and the rule fits (section 6 of the notebook) | 38 min, 40 GB more cache |
| Training all 20 networks | about 15 GPU-hours |

## Data and pretrained weights

- Training used only the images and annotations of the competition training set.
- The encoders start from public pretrained weights: ResNet-34 trained on ImageNet (through
  `segmentation-models-pytorch`), EfficientNetV2-S and PVTv2-B2 trained on ImageNet (through `timm`), and
  Mask R-CNN ResNet-50 FPN v2 trained on COCO (through `torchvision`).
- No other images, labels or pseudo-labels were used. Inference uses only the test frames.

## The first final submission (public LB 0.40)

```bash
for k in 0 1 2 3 4; do
  python src/fb/infer.py --runs unet_hard_f$k --tta hflip --tag hard_f${k}_h
  python src/fb/infer.py --runs unet_soft_f$k --tta hflip --tag soft_f${k}_h
done
python src/fb/infer.py --runs unet_hard_f0,unet_hard_f1,unet_hard_f2,unet_hard_f3,unet_hard_f4 --tta d4 --tag hard5
python src/fb/infer.py --runs unet_soft_f0,unet_soft_f1,unet_soft_f2,unet_soft_f3,unet_soft_f4 --tta d4 --tag soft5
python src/fb/submit.py --rule rules/first_final.json --ens hard5,soft5,maskrcnn_f0+...+maskrcnn_f4 \
       --single "hard_f0_h,...;soft_f0_h,...;maskrcnn_f0,..." --out submission/reproduced_first.csv --ref submission/first_final.csv
```

Training: `unet_hard_f$k` with `--target tfull --aug strong --steps 18000`, then `unet_soft_f$k` with
`--target tsoftc --init unet_hard_f$k --steps 4000 --lr 1.5e-4`. Out-of-fold PQ 0.457, public LB 0.40.
