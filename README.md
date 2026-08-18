# DRel-ECG

Reference implementation of **DRel-ECG** (*Diagnostic-Relation-Guided Contrastive Pretraining for Multi-Label ECG Diagnosis*) and its core mechanism, **Graph-Informed Hard Negative Modeling (GHNM)**. The repository includes patient-grouped relation-graph construction, matched controls, component ablations, downstream evaluation, missing-lead robustness, and patient-clustered paired bootstrap analysis.

## Repository layout

```text
drel_ecg/                  Canonical model, loss, data, graph, and evaluation package
mcki_ecg/                  Backward-compatible import wrappers for archived code
scripts/training/          Pretraining, graph controls, ablations, and adaptation
scripts/evaluation/        Strict LP, external, missing-lead, and sensitivity analyses
scripts/statistics/        Patient-clustered and graph-control paired bootstrap
scripts/figures/           Relation-graph figure and auditable matrix exports
configs/                   Canonical protocol and example comparison configuration
resources/manifests/       PTB-XL split and audit manifests used in the study
resources/hndr_pairs.csv   Prespecified diagnostic pairs for HNDR
tests/                     Lightweight core and compatibility checks
```

## Canonical terminology

- **DRel-ECG**: the complete proposed framework.
- **GHNM**: Graph-Informed Hard Negative Modeling, the diagnostic-relation-guided denominator-weighting mechanism.
- **Hybrid**: the graph built from the equally weighted co-occurrence prior and normalized patient-grouped cross-fitted confusion matrix.
- **Uniform Negatives**: GHNM removed while all other modules remain matched.
- **Degree-Matched Shuffled**: a shuffled graph with the same edge budget.
- **Dynamic Lead Masking**, **Local Contrastive Loss**, **Alignment**, and **Lead-Aware Modulation**: component names used in code and manuscript tables.
- **Strict Linear Probing**: encoder parameters and normalization state are frozen; only a new linear classifier is optimized.

The former public name **MCKI-ECG** is retained only in the `mcki_ecg` compatibility namespace and legacy aliases required to load archived scripts and checkpoints. New code, result tables, and artifacts use `DRel-ECG`, `drel_ecg`, and `DRelECGModel`.

## Environment

Python 3.10 or newer is recommended. Install a CUDA-compatible PyTorch build for the target machine first, then install this package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Data contract

DRel-ECG uses the PTB-XL five-superclass task with the fixed class order `NORM, MI, STTC, CD, HYP`. Signals are stored as 100 Hz, 10-second arrays with shape `N × 12 × 1000`. The released manifests document record identity, patient identity, official fold assignment, and split membership.

Place the prepared arrays under `data/processed` using the filenames expected by `drel_ecg.data`. Raw ECG files, external cohorts, predictions, and checkpoints are not committed to Git.

Georgia and SPH evaluation expects cohort-specific `X_test.npy` and `y_test_mh.npy` files with the same signal shape and class order. The evaluation scripts do not perform target-cohort training, model selection, or threshold recalibration.

## Relation graph and DRel-ECG training

The canonical configuration is documented in [`configs/drel_ecg.yaml`](configs/drel_ecg.yaml). Executable defaults are defined in `drel_ecg/experiment.py`.

Build the train-only record-grouped reference used for graph-stability comparison:

```bash
python -m scripts.training.train_crossfit_hybrid \
  --data-dir data/processed \
  --out-dir outputs/record_grouped_reference \
  --variants Hybrid \
  --protocols Linear_Probing \
  --seeds 42,123,1024
```

Reconstruct the relation graph with patient-grouped five-fold cross-fitting:

```bash
python -m scripts.training.train_patient_grouped \
  --data-dir data/processed \
  --manifest resources/manifests/ptbxl_train_manifest.csv \
  --record-root outputs/record_grouped_reference/Hybrid \
  --out-dir outputs/patient_grouped_reference
```

Use the patient-grouped graph artifacts for the second pretraining stage and matched controls:

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

## Strict Linear Probing

The checkpoint pattern must contain `{seed}`:

```bash
python -m scripts.evaluation.strict_linear_probe \
  --data-dir data/processed \
  --checkpoint-pattern 'outputs/fixed_graph_controls/Hybrid/seed_{seed}/pretrained_checkpoint.pt' \
  --out-dir outputs/strict_linear_probe
```

Each seed directory contains the frozen-encoder linear head, validation-selected thresholds, validation/test probabilities, targets, metrics, and the three-seed summary.

## Missing-lead robustness

```bash
python -m scripts.evaluation.missing_lead \
  --data-dir data/processed \
  --encoder-checkpoint-pattern 'outputs/fixed_graph_controls/Hybrid/seed_{seed}/pretrained_checkpoint.pt' \
  --head-pattern 'outputs/strict_linear_probe/seed_{seed}/strict_linear_head.pt' \
  --threshold-pattern 'outputs/strict_linear_probe/seed_{seed}/thresholds.npy' \
  --out-dir outputs/missing_lead
```

Random lead masks are deterministic per record and shared across model seeds. The summary reports the mean and sample standard deviation of both absolute AUPRC and the within-seed decrease from the original input.

## External evaluation

```bash
python -m scripts.evaluation.external \
  --encoder-checkpoint-pattern 'outputs/fixed_graph_controls/Hybrid/seed_{seed}/pretrained_checkpoint.pt' \
  --head-pattern 'outputs/strict_linear_probe/seed_{seed}/strict_linear_head.pt' \
  --threshold-pattern 'outputs/strict_linear_probe/seed_{seed}/thresholds.npy' \
  --georgia-dir data/external/georgia \
  --sph-dir data/external/sph \
  --out-dir outputs/external
```

## Patient-clustered paired bootstrap

Define probability and target patterns following [`configs/comparisons.example.json`](configs/comparisons.example.json), then run:

```bash
python -m scripts.statistics.patient_clustered_bootstrap \
  --root . \
  --manifest resources/manifests/ptbxl_test_manifest.csv \
  --pairs resources/hndr_pairs.csv \
  --comparison-config configs/comparisons.example.json \
  --out-dir outputs/bootstrap \
  --n-boot 10000
```

The output reports the seed-mean paired difference, 95% percentile interval, empirical `Pr(Delta <= 0)`, and the number of seeds with a positive direction.

## Backward compatibility

Archived code can continue to use imports such as:

```python
from mcki_ecg.model import MCKIECGModel
```

The canonical equivalent is:

```python
from drel_ecg.model import DRelECGModel
```

Both names resolve to the same implementation. The legacy backbone identifiers `mcki_ecg_resnet18`, `build_mcki_ecg_backbone`, and `build_MCKI_backbone` are also accepted. Compatibility names must not be used in new result tables or manuscript text.

