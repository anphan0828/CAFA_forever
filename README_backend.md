# Backend README

This repository has two user-facing surfaces:

- the LAFA compute backend, which builds timepoint artifacts and publishes evaluation releases
- the Streamlit frontend, which reads only published releases from `data/releases/`

This document covers the backend. Frontend usage is summarized in [README.md](README.md).

## What The Backend Does

The backend translates staged UniProt/GO inputs into two kinds of outputs:

- canonical timepoint bundles under `<timepoint>/release/`
- published release windows under `data/releases/<T0>_<T1>/`

The current Nextflow implementation supports three top-level modes:

- `timepoint_build`
- `evaluate_window`
- `evaluate_late_predictions`

The default design assumption is `prediction_target_mode = 'window_groundtruth'`. In that mode, `timepoint_build` prepares canonical artifacts only, and `evaluate_window` runs predictors against the window-specific `groundtruth_targets.fasta`.

## Installation

Conda or micromamba is strongly recommended. The current pipeline expects:

- a `democafaenv` environment for `democafa`
- a `cafa5-evaluator` environment for `cafaeval`
- Nextflow available either from a local install or an HPC module

### 1. Clone the repo

```bash
git clone https://github.com/anphan0828/CAFA_forever.git
cd CAFA_forever
```

### 2. Create the `democafa` environment

Replace `/path/to/democafa_package` with your local checkout.

```bash
micromamba create -y -n democafaenv python=3.10 pip
micromamba activate democafaenv
pip install -e /path/to/democafa_package
```

### 3. Create the `cafaeval` environment

```bash
micromamba create -y -n cafa5-evaluator python=3.10 pip
micromamba activate cafa5-evaluator
pip install cafaeval
```

### 4. Make Nextflow available

On HPC this is often a module:

```bash
module load nextflow/23.10.1-f6t76je
```

Or install Nextflow locally following the standard Nextflow instructions.

### 5. Point the workflow at `democafa`

The pipeline currently uses `params.democafa_package` from [nextflow.config](nextflow.config). Override it at runtime if your checkout is elsewhere:

```bash
nextflow run main.nf ... --democafa_package /path/to/democafa_package
```

## Typical Use Cases

### Build a new timepoint bundle

Use this after staging raw snapshot files into a timepoint directory.

```bash
nextflow run main.nf \
  --mode timepoint_build \
  --timepoint_id Oct_2025 \
  --timepoint_root /path/to/time_points/Oct_2025 \
  --output_root /path/to/output \
  --democafa_package /path/to/democafa_package
```

What this produces:

- filtered and propagated training artifacts
- `blast_results.tsv`
- `test_sequences_split/`
- `release/`
- timepoint-global predictions only when `prediction_target_mode != 'window_groundtruth'`

### Evaluate a release window

This is the main user flow in the current implementation.

```bash
nextflow run main.nf \
  --mode evaluate_window \
  --t0_id Jun_2025 \
  --t1_id Oct_2025 \
  --t0_root /path/to/time_points/Jun_2025 \
  --t1_root /path/to/time_points/Oct_2025 \
  --publish_root /path/to/data/releases \
  --output_root /path/to/output \
  --democafa_package /path/to/democafa_package
```

In the default `window_groundtruth` mode this workflow:

- intersects the target universe between `T0` and `T1`
- classifies ground truth into `NK`, `LK`, and `PK`
- materializes `groundtruth_targets.fasta`
- runs enabled methods against that window-specific FASTA
- writes window predictions under `T0/predictions_by_window/<window_id>/`
- publishes a frontend-ready release under `data/releases/<window_id>/`

You can restrict which methods run:

```bash
nextflow run main.nf \
  --mode evaluate_window \
  --enabled_window_methods naive,blast,prott5 \
  ...
```

### Evaluate late predictions

Use this when new predictions have been dropped into `T0/predictions_uneval/` and you want to merge them into an existing release without rerunning the full window.

```bash
nextflow run main.nf \
  --mode evaluate_late_predictions \
  --t0_id Jun_2025 \
  --release_id Jun_2025_Oct_2025 \
  --t0_root /path/to/time_points/Jun_2025 \
  --release_root /path/to/data/releases/Jun_2025_Oct_2025
```

## Smoke Tests

The repo contains a minimal fixture for parser and `-stub-run` validation:

- [tests/fixtures/smoke/README.md](tests/fixtures/smoke/README.md)

Use it to test:

- `timepoint_build`
- `evaluate_window`
- `evaluate_late_predictions`

The fixture is intentionally tiny and should not be used for scientific validation.

## Published Release Contract

Each published release under `data/releases/<release_id>/` is expected to contain:

- `method_names.tsv`
- `method_availability.tsv`
- `groundtruth_NK.tsv`
- `groundtruth_LK.tsv`
- `groundtruth_PK.tsv`
- `groundtruth_PK_known.tsv`
- `groundtruth_terms_of_interest.txt`
- `groundtruth_targets.tsv`
- `groundtruth_targets.fasta`
- `results_NK/`
- `results_LK/`
- `results_PK/`

The frontend reads this published surface only. It does not read raw timepoint directories, Nextflow `work/`, or unpublished staging locations.

## Supplementary Details

### Execution Topology

- `timepoint_build` is mostly sequential until prediction fan-out.
- `evaluate_window` is sequential through ground-truth construction, then fans out by method, then evaluates `NK`, `LK`, and `PK`.
- `evaluate_late_predictions` reevaluates only `predictions_uneval/` and merges the late result tables.

### Artifact Map

Canonical timepoint artifacts live under `<T0>/release/` and may also include:

- `<T0>/test_sequences_split/`
- `<T0>/predictions/` in full-set mode
- `<T0>/predictions_by_window/<window_id>/` in `window_groundtruth` mode

Published release windows live under `data/releases/<T0>_<T1>/`.

### Notes On Current Ops Assumptions

- `democafa` commands are installed from a local checkout via `pip install -e`.
- `cafaeval` is treated as an external executable.
- Predictor containers and cache paths are still configured through local absolute paths in [nextflow.config](nextflow.config) and the prediction modules; those should be cleaned up before public multi-user deployment.

For the full process-oriented backend contract, see [lafa-compute-backend-process-spec.md](lafa-compute-backend-process-spec.md).
