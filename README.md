# Pneumonia CXR Classification — ResNet-18 vs GoogLeNet

PyTorch implementation of the code plan in: 4-model comparison
(ResNet-18 / GoogLeNet × no-aug / aug) on the Kermany paediatric chest X-ray
dataset, with patient-level splitting, two-stage transfer learning, and
GradCAM++ explanations.

## Layout

```
chest_xray/                 # Kermany dataset (train/, val/, test/)
config.py                   # seeds, paths, device, experiment matrix (Algo 0)
data.py                     # patient-level split, transforms, loaders (Algo 1)
models.py                   # ResNet-18 / GoogLeNet builders (Algo 2)
metrics.py                  # sens/spec/AUROC/MCC/F1 + threshold helpers
engine.py                   # train/eval loops, EarlyStopping, CSV logging
hp_search.py                # coordinate HP search (Algo 3)
train_pipeline.py           # Stage-1 + Stage-2 (Algos 4, 5)
evaluate_test.py            # test eval + threshold sweep + GradCAM++ (Algo 6)
compare.py                  # aggregate report tables (Algo 7)
main.py                     # orchestrator (run this)
outputs/
  results.csv               # all per-run rows (HP, training, eval)
  checkpoints/              # s1_/s2_ best weights per (arch, aug)
  figures/                  # GradCAM and threshold plots, threshold tables
  table_*.csv               # final report tables
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# or: source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

The dataset must already exist at `./chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`.

## Run

```bash
python main.py                 # full pipeline (HP search per arch + train + eval + report)
python main.py --skip-hp       # skip HP sweep, use BASELINE_HP from config.py
python main.py --eval-only     # only run Algorithm 6 against existing checkpoints
python main.py --report-only   # only rebuild report tables from results.csv
python main.py --archs resnet18 --augs 0 1   # restrict matrix
```

## Notes

- Patient leakage is asserted in `data.patient_level_split` — the run will
  abort if patient IDs overlap across train/val/test.
- HP search runs once per architecture on the no-aug split; the best HPs
  picked for each architecture are reused for that architecture's augmented
  variant, so `train_pipeline` produces 4 final Stage-2 models.
- All seeds are fixed (`SEED = 42`) and `cudnn.deterministic = True`.
- GradCAM++ requires the `grad-cam` package; if missing, the test step still
  runs and writes metrics + threshold figures, but skips heatmaps.
