# MCKI-ECG

This repository contains the reference implementation of **MCKI-ECG**, including graph-guided hard-negative mining (GHNM), patient-grouped relation-graph construction, downstream evaluation, component controls, missing-lead robustness, and patient-clustered paired bootstrap analysis.



## Repository layout

```text
mcki_ecg/                  Core model, loss, data, graph, and evaluation modules
scripts/data/              PTB-XL preprocessing
scripts/training/          Pretraining, graph controls, ablations, and adaptation
scripts/evaluation/        Strict LP, external, missing-lead, and sensitivity analyses
scripts/statistics/        Patient-clustered and graph-control paired bootstrap
scripts/figures/           Relation-graph figure and auditable matrix exports
configs/                   Canonical protocol and example comparison configuration
resources/manifests/       PTB-XL split/audit manifests used in the study
resources/hndr_pairs.csv   Prespecified diagnostic pairs for HNDR
tests/                     Lightweight core checks
```

## Terminology

The public code uses one canonical vocabulary:

- **MCKI-ECG**: complete method.
- **GHNM**: graph-guided hard-negative mining loss.
- **Hybrid**: graph built from the equally weighted co-occurrence prior and normalized patient-grouped cross-fitted confusion matrix.
- **Uniform Negatives**: GHNM removed while all other modules remain matched.
- **Degree-Matched Shuffled**: shuffled graph with the same edge budget.
- **Dynamic Lead Masking**, **Local Contrastive Loss**, **Alignment**, and **Lead-Aware Modulation**: component names used in code and manuscript tables.
- **Strict Linear Probing**: encoder parameters and normalization state are frozen; only a new linear classifier is optimized.

Legacy development names such as `stage8`, date-suffixed filenames, `MCKI_Pro`, and `processed_v3` are not part of the public file layout. A few internal compatibility aliases remain solely so that archived checkpoints can be loaded.

## Environment

Python 3.10 or newer is recommended. Install a CUDA-compatible PyTorch build for the target machine first, then install the remaining package dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## PTB-XL preparation

Download and unpack PTB-XL v1.0.3 from PhysioNet. The preparation script reads the official 100 Hz records and creates the five-superclass arrays using folds 1–8/9/10 for train/validation/test:

```bash
python -m scripts.data.prepare_ptbxl \
  --source-root /path/to/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
  --out-dir data/processed
```

Expected signal shape is `12 × 1000` after loading. The class order is fixed to `NORM, MI, STTC, CD, HYP`. The script saves record IDs, patient IDs, `strat_fold`, filenames, labels, positive counts, and patient-overlap checks.

Raw ECG data, external cohorts, predictions, and checkpoints are not committed to Git.

## Reproduce the relation graph and MCKI-ECG training

The canonical configuration is documented in [`configs/mcki_ecg.yaml`](configs/mcki_ecg.yaml). The executable experiment defaults are defined in `mcki_ecg/experiment.py`.

First build the train-only record-grouped reference used for the graph-stability comparison:

```bash
python -m scripts.training.train_crossfit_hybrid \
  --data-dir data/processed \
  --out-dir outputs/record_grouped_reference \
  --variants Hybrid \
  --protocols Linear_Probing \
  --seeds 42,123,1024
```

Then reconstruct the graph with patient-grouped five-fold cross-fitting:

```bash
python -m scripts.training.train_patient_grouped \
  --data-dir data/processed \
  --manifest resources/manifests/ptbxl_train_manifest.csv \
  --record-root outputs/record_grouped_reference/Hybrid \
  --out-dir outputs/patient_grouped_reference
```

Finally use the patient-grouped graph artifacts for the second pretraining stage, matched graph controls, and requested downstream protocols:

```bash
python -m scripts.training.train_fixed_graph_controls \
  --data-dir data/processed \
  --fixed-graph-root outputs/patient_grouped_reference \
  --out-dir outputs/fixed_graph_controls \
  --variants Hybrid,Uniform_Negatives,Degree_Matched_Shuffled \
  --protocols Linear_Probing \
  --seeds 42,123,1024
```

Component variants are run with `scripts.training.train_component_ablations` using the same data, seeds, graph, edge budget, and training schedule.

## Strict linear probing

The checkpoint pattern must contain `{seed}`:

```bash
python -m scripts.evaluation.strict_linear_probe \
  --data-dir data/processed \
  --checkpoint-pattern 'outputs/fixed_graph_controls/Hybrid/seed_{seed}/pretrained_checkpoint.pt' \
  --out-dir outputs/strict_linear_probe
```

The script saves each seed's frozen-encoder head, validation-selected thresholds, validation/test probabilities, targets, metrics, and the three-seed summary.

## Missing-lead robustness

```bash
python -m scripts.evaluation.missing_lead \
  --data-dir data/processed \
  --encoder-checkpoint-pattern 'outputs/fixed_graph_controls/Hybrid/seed_{seed}/pretrained_checkpoint.pt' \
  --head-pattern 'outputs/strict_linear_probe/seed_{seed}/strict_linear_head.pt' \
  --threshold-pattern 'outputs/strict_linear_probe/seed_{seed}/thresholds.npy' \
  --out-dir outputs/missing_lead
```

Random lead masks are deterministic per record and shared across model seeds. The summary reports the mean and sample standard deviation of both absolute AUPRC and the within-seed decrease from Original.

## External evaluation

Georgia and SPH must be prepared as `X_test.npy` and `y_test_mh.npy` with the same input shape and class order:

```bash
python -m scripts.evaluation.external \
  --encoder-checkpoint-pattern 'outputs/fixed_graph_controls/Hybrid/seed_{seed}/pretrained_checkpoint.pt' \
  --head-pattern 'outputs/strict_linear_probe/seed_{seed}/strict_linear_head.pt' \
  --threshold-pattern 'outputs/strict_linear_probe/seed_{seed}/thresholds.npy' \
  --georgia-dir data/external/georgia \
  --sph-dir data/external/sph \
  --out-dir outputs/external
```

No target-cohort training or threshold tuning is performed.

## Patient-clustered paired bootstrap

Define probability/target patterns in a JSON file following [`configs/comparisons.example.json`](configs/comparisons.example.json), then run:

```bash
python -m scripts.statistics.patient_clustered_bootstrap \
  --root . \
  --manifest resources/manifests/ptbxl_test_manifest.csv \
  --pairs resources/hndr_pairs.csv \
  --comparison-config configs/comparisons.example.json \
  --out-dir outputs/bootstrap \
  --n-boot 10000
```

The output reports the seed-mean paired difference, 95% percentile interval, empirical `Pr(Δ ≤ 0)`, and the number of positive seed directions.

