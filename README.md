# Generating Diverse Synthetic Eyeglass Datasets Using GANs

Code for the paper "Generating Diverse Synthetic Eyeglass Datasets Using GANs"
(G. Shubitidze, H. Giedra, R. Pomarnacki, D. Matuzevicius, 2026).

The repository contains:

- `gan/gan_train.py` - StyleGAN2-based generator for 256x512 frontal eyeglass
  frame images (training and generation)
- `metrics/` - the diversity and quality metric suite used in the paper
- `scripts/` - scripts that produce the paper figures
- `configs/` - training settings of the two models reported in the paper

Trained generator weights and the generated datasets are available on
Zenodo (see below).

## Installation

Python 3.10 or newer and a CUDA GPU are recommended.

```
pip install -r requirements.txt
```

Only stock PyTorch is used, no custom CUDA operations are compiled.
Training at the paper settings takes about 3 minutes per epoch on an
RTX 4090 (24 GB).

## Data

The real catalogue images used in the paper are not distributed with this
repository. The code expects frontal frame photographs on a near-uniform
light background, placed in one folder:

```
data/source/images/*.png
```

Images are centre-cropped to 2:1 and resized to 256x512 (bicubic).

## Training

Compute the colour sampling weights once:

```
python metrics/compute_sample_weights.py --real data/source/images --out data/sample_weights.json
```

Train the model without mirror coupling (sharp1 in the paper):

```
python gan/gan_train.py --images data/source/images --output results/sharp1 \
    --end-epoch 1600 --sample-weights data/sample_weights.json
```

Train the model with the learned mirror axis (mirror2 in the paper):

```
python gan/gan_train.py --images data/source/images --output results/mirror2 \
    --end-epoch 1200 --sample-weights data/sample_weights.json --g-mirror-axis
```

All other settings default to the values used in the paper (batch 16,
R1 gamma 5, path length weight 4, colour augmentation off). The full
settings are listed in `configs/`. Each run writes its configuration to
`results/<run>/run_config.json`. The checkpoint with the lowest training
KID is kept as `checkpoints/checkpoint_best.pth`.

## Generation

```
python gan/gan_train.py --generate-only results/mirror2/checkpoints/checkpoint_best.pth \
    --output results/mirror2_eval --g-mirror-axis
```

The default generation settings are the protocol used in the paper:
10,000 images, single latent per image, truncation psi 1.0 for the three
coarse stages and 1.2 for the three fine stages, seed 42. The
`--g-mirror-axis` flag must match the model; omit it for sharp1.

The released weights (`mirror2_generator.pth`, `sharp1_generator.pth`) can be
downloaded from Zenodo into `weights/` and used in the same way:

```
python gan/gan_train.py --generate-only weights/mirror2_generator.pth \
    --output results/mirror2_eval --g-mirror-axis
python gan/gan_train.py --generate-only weights/sharp1_generator.pth \
    --output results/sharp1_eval
```

## Evaluation

Full diversity report (FID, KID, precision, recall, density, coverage,
Vendi, geometry, colour, LPIPS, mirror symmetry and yaw dispersion):

```
python metrics/evaluate_diversity.py --generated results/mirror2_eval/generated/images \
    --real data/source/images --out results/mirror2_eval/diversity
```

Structural artifact checks:

```
python metrics/diagnose_artifacts.py --real data/source/images \
    --generated results/mirror2_eval/generated/images --out results/mirror2_eval/artifacts
```

Realistic novelty rate:

```
python metrics/novelty_rate.py --real data/source/images \
    --generated results/mirror2_eval/generated/images --out results/mirror2_eval/novelty
```

Missing colour bins:

```
python metrics/diagnose_ab_gap.py --real data/source/images \
    --generated results/mirror2_eval/generated/images --out results/mirror2_eval/ab_gap
```

## StyleGAN2-ADA baseline

The paper compares our models with the official StyleGAN2-ADA, trained with
NVIDIA's StyleGAN3 code (https://github.com/NVlabs/stylegan3, commit
c233a919) in its StyleGAN2 configuration. The official code trains on square
images only, so the 256x512 images are padded to 512x512 and the generated
images are cropped back:

```
python baseline/prepare_data.py --images data/source/images --out ../sg2ada_data/padded
python <stylegan3>/dataset_tool.py --source ../sg2ada_data/padded \
    --dest ../sg2ada_data/catalogue_512.zip --resolution 512x512

cd <stylegan3>
python train.py --outdir=../sg2ada_runs --data=../sg2ada_data/catalogue_512.zip \
    --cfg=stylegan2 --gpus=1 --batch=16 --mirror=1 --aug=ada --gamma=10 \
    --metrics=fid50k_full --kimg=3000 --snap=50
```

The paper reports two runs that differ only in `--gamma`: 3.2768 (the
heuristic 0.0002 * 512^2 / 16 of the ADA paper) and 10. The snapshot with the
lowest `fid50k_full` is selected, as in the official procedure. The gamma-10
run was stopped by a rule fixed in advance: stop when the lowest FID has not
improved by more than 2% over three consecutive snapshots, or when the FID
exceeds three times the lowest value. Then:

```
python gen_images.py --network=<snapshot.pkl> --seeds=0-9999 --trunc=1 \
    --noise-mode=const --outdir=../sg2ada_gen/baseline_512
cd <this repository>
python baseline/crop_generated.py --src ../sg2ada_gen/baseline_512 \
    --dst results/baseline/generated/images
```

and evaluate `results/baseline/generated/images` with the commands of the
Evaluation section. Running the official code with current PyTorch and SciPy
versions needs a few compatibility changes (sampler constructor, custom-op
module loading, `scipy.linalg.sqrtm` call); none of them changes the training
or the FID computation.

## Reproducing the paper tables and figures

| Paper item | Command |
|---|---|
| Generation protocol table | `gan_train.py --generate-only` with `--truncation-psi`, `--truncation-psi-fine`, `--pure-frac`, `--truncation-centers`, then `evaluate_diversity.py` |
| Main results table | `evaluate_diversity.py` on each model |
| Noise floor | generate the same checkpoint with `--seed 42` and `--seed 0`, evaluate both |
| Symmetry validation table | `python metrics/probe_symmetry_synthetic.py` |
| Mapping network measurement | `python metrics/probe_mapping_wstd.py` |
| Colour coverage bin check | `python metrics/ab_bin_stability.py --real ... --sets name=path ...` |
| Artifact rates | `diagnose_artifacts.py` |
| Novelty table | `novelty_rate.py`; calibration with `--calibration-check` |
| Baseline comparison table | the StyleGAN2-ADA section above, then `evaluate_diversity.py`, `diagnose_artifacts.py` and `novelty_rate.py` on the cropped images |
| Pareto figure | `python scripts/make_pareto_figure.py --runs label=path ... --baseline-runs label=path` |
| Sample grid | `python scripts/make_report_grid.py --real ... --generated label=path ...` |

Results depend on the GPU, driver and library versions, so values may differ
slightly from the paper. The differences between generation seeds measured
in the paper are of the order of 0.05 FID and 0.02 recall.

## Pretrained models and generated datasets

Zenodo: https://doi.org/10.5281/zenodo.23287084

- `mirror2_generator.pth`, `sharp1_generator.pth` - generator weights (EMA)
- `mirror2_images.zip`, `sharp1_images.zip` - 10,000 generated images each,
  produced with the default generation settings above

## License

Code: MIT license, see `LICENSE`.

## Citation

```
@article{shubitidze2026diverse,
  title   = {Generating Diverse Synthetic Eyeglass Datasets Using {GAN}s},
  author  = {Shubitidze, Giga and Giedra, Henrikas and Pomarnacki, Raimondas
             and Matuzevi{\v{c}}ius, Dalius},
  journal = {Electronics},
  year    = {2026}
}
```
