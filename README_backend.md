# Backend README

This repository has two user-facing surfaces:

- the LAFA compute backend, which builds timepoint artifacts and publishes evaluation releases
- the Streamlit frontend, which reads only published releases from `data/releases/`

This document covers the backend. Frontend usage is summarized in [README.md](README.md).

The September 26 implementation adds [timepoint provenance and window policy checks](docs/timepoint_provenance.md). The completed June test's [output review](docs/jun_2026_output_review.md) records coverage, resource use and the PomBase mapping correction; operational success is separate from official scientific approval.

## What The Backend Does

The backend translates staged UniProt/GO inputs into two kinds of outputs:

- canonical timepoint bundles under `<timepoint>/release/`
- published release windows under `data/releases/<T0>_<T1>/`

The current Nextflow implementation supports four top-level modes:

- `model_organism_intake` (dated model-organism GAF acquisition)
- `timepoint_build`
- `evaluate_window`
- `evaluate_late_predictions`

When full timepoint-level prediction is available, `evaluate_window` can use
the existing files in `T0/predictions/`. When GPU compute is constrained, pass
`--window_prediction_mode generate` so `evaluate_window` runs predictors against
the smaller window-specific `groundtruth_targets.fasta`.

The default in [nextflow.config](nextflow.config) is currently
`prediction_target_mode = 'window_groundtruth'`. In that mode,
`timepoint_build` creates canonical timepoint data and an empty `predictions/`
directory, then `evaluate_window` owns the real prediction fan-out after the
window target FASTA exists.

## Current Nextflow State

### `model_organism_intake`

Optional standalone utility. Normal `timepoint_build` runs include this retrieval
automatically alongside GOA and SwissProt. This utility downloads a dated GO
model-organism snapshot through democafa, rejects partial
organism coverage, and preserves raw GAFs, the combined GAF, matching ontology,
and provenance in a separate unapproved intake bundle. Normal timepoint builds
now map and merge organism terms into `train_terms.tsv`. See [usage, review, and three-source
integration gaps](docs/model_organism_intake.md).

### `timepoint_build`

The timepoint workflow now:

- downloads UniProt-GOA, matched SwissProt DAT/FASTA, and model-organism GAFs
  with their matching GO ontology in parallel by default
- uses current sources unless `--resource_release YYYY-MM-DD` selects a dated build
- waits for all three acquisitions before preparation and retains raw files/manifests
  under `<timepoint>/resources/`
- supports `--timepoint_input_mode staged` for the legacy four-file offline boundary
- supports `--timepoint_input_mode acquired` to build from a dated, manifest-bearing
  `resources/` tree without downloading again
- filters GOA to SwissProt as `goa_uniprot_sprot.gaf.gz`
- retrieves experimental terms from filtered GOA, SwissProt DAT, and the combined
  model-organism GAF using the same DAT snapshot for identifier mapping
- validates sequence/ontology coverage and merges on `(EntryID, term, aspect)`,
  retaining the earliest valid date and freezing source lineage and mapping
  audits under `release/annotation_provenance/`
- creates `train_sequences.fasta`, `train_taxonomy.tsv`, and a full
  `test_sequences.fasta`
- propagates terms and computes `IA.tsv`
- splits the full test FASTA into `test_sequences_split/`
- freezes the canonical bundle under `<timepoint>/release/`
- copies the September 2025 evaluation ontology to
  `go-basic_frozen.obo` in both the timepoint root and frozen release; override
  its source with `--frozen_graph /path/to/go-basic.obo`

When `--prediction_target_mode test_sequences` or another non-window value is
used, only the timepoint-compatible built-in predictors are run from this
workflow: `naive`, `goa_nonexp`, `blast`, and `prott5`. `transfew`, `funbind`,
and `deepgoplus` are treated as window-scoped methods because their useful
query input is `groundtruth_targets.fasta`.

### `evaluate_window`

The window workflow now supports three prediction modes:

- default omitted mode: reuse all `*_predictions.tsv` files in `T0/predictions/`
- `--window_prediction_mode generate`: build `groundtruth_targets.fasta`, run
  the enabled methods against that FASTA, then copy predictions to
  `T0/predictions_by_window/<release_id>/`
- `--window_prediction_mode existing`: rebuild/validate ground truth and use
  the exact expected files from `--window_predictions_dir`

When T0 is an already published legacy bundle without annotation provenance,
pass `--trusted_published_t0 true`. The workflow binds that trust to a ready
release in `--published_release_catalog` (or `<publish_root>/catalog.json`),
checks the frozen T0 evaluation artifacts, fully validates the new T1 manifest,
and records the accepted annotation-source expansion in the window release.

It also supports `--query_fasta`/`--queryFasta` to bypass the T0/T1 FASTA
intersection when an explicit query FASTA has already been prepared. That path
still runs ground-truth classification and release packaging.

Window generation can fan out across:

- CPU/container methods: `naive`, `goa_nonexp`
- GPU methods: `blast`, `prott5`, `transfew`, `deepgoplus`
- batched GPU method: `funbind`, split from `groundtruth_targets.fasta` and
  merged by [scripts/merge_funbind_batches.py](scripts/merge_funbind_batches.py)

## Installation

Conda or micromamba is strongly recommended. The current pipeline expects:

- a `democafaenv` environment for `democafa`
- a `cafa5-evaluator` environment for `cafaeval`
- Nextflow available either from a local install or an HPC module

These names are defaults, not hard requirements. The workflow reads them from [nextflow.config](nextflow.config):

- `params.democafa_env`
- `params.cafaeval_env`

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

The pipeline currently uses these runtime dependency params from [nextflow.config](nextflow.config):

- `params.democafa_package`
- `params.democafa_env`
- `params.cafaeval_env`

Override them at runtime if your local setup uses different names or locations:

```bash
nextflow run main.nf ... \
  --democafa_package /path/to/democafa_package \
  --democafa_env my-democafa-env \
  --cafaeval_env my-cafaeval-env
```

### 6. Verify the required tools are installed into those environments

The workflow assumes:

- `python -m democafa...` commands work after `micromamba activate $democafa_env`
- `cafaeval` is on `PATH` after `micromamba activate $cafaeval_env`

Minimal checks:

```bash
micromamba activate democafaenv
python -c "import democafa; print('democafa ok')"

micromamba activate cafa5-evaluator
cafaeval --help
```

If you use different environment names, substitute the values you pass through `--democafa_env` and `--cafaeval_env`.

## Typical Use Cases

### Build a new timepoint bundle

Normal builds download all three sources automatically. Use current resources:

```bash
nextflow run main.nf \
  --mode timepoint_build \
  --timepoint_id Sep_2026 \
  --timepoint_root /path/to/time_points/Sep_2026 \
  --democafa_package /path/to/democafa_package
```

Add `--resource_release YYYY-MM-DD` for a historical cutoff. Use
`--go_release YYYY-MM-DD` when the exact GO/model-organism release is earlier
than that cutoff; GOA/SwissProt still select snapshots available by the cutoff.
See the [fresh June test launcher](docs/jun_2026_test_run.md). Backfills must specify
the date; it is not inferred from the timepoint ID. Override the package's default
organism catalog with `--model_organism_config /path/to/catalog.yaml`.

To reuse legacy staged files without downloading, explicitly pass
`--timepoint_input_mode staged`; the directory must contain `go-basic.obo`,
`goa_uniprot_all.gaf.gz`, `uniprot_sprot.fasta.gz`, and `uniprot_sprot.dat.gz`.
For staged three-source builds, also supply `--model_organism_bundle` and its
`--model_organism_bundle_source` subdirectory (default `current`). Otherwise the
merge policy records a legacy two-source build.
See [retrieval and merge policies](docs/model_organism_intake.md).

To build from already acquired, manifest-bearing resources, retain this layout:

```text
<timepoint_root>/resources/goa/<cutoff>/
<timepoint_root>/resources/swissprot/<cutoff>/
<timepoint_root>/resources/model_organisms/<go-release>/
```

Then use `--timepoint_input_mode acquired` with explicit dated
`--resource_release` and `--go_release`. This mode verifies and retains the
download manifests but does not run any downloader. The acquisition catalog at
`resources/model_organisms/catalog.yaml` remains source provenance. Identifier
normalization uses `--model_organism_mapping_config`, defaulting to the repository
catalog configured by `--model_organism_config`.

For the prepared June candidate:

```bash
nextflow run main.nf \
  --mode timepoint_build \
  --timepoint_id Jun_2026 \
  --timepoint_root new_data/time_points/Jun_2026 \
  --timepoint_input_mode acquired \
  --resource_release 2026-06-01 \
  --go_release 2026-05-19 \
  --model_organism_mapping_config configs/model_organism_annotations.yaml \
  --prediction_target_mode window_groundtruth \
  -profile slurm
```

What this produces:

- filtered and propagated training artifacts
- `test_sequences_split/`
- `release/`
- timepoint-global predictions only when `prediction_target_mode != 'window_groundtruth'`

If you want the later `evaluate_window` run to use `T0/predictions/`, build or
stage those prediction files before evaluation. For the current built-in
timepoint prediction branch (`Naive`, `GOA Non-exp`, `BLAST`, and `ProtT5`),
pass a non-window target mode:

```bash
nextflow run main.nf \
  --mode timepoint_build \
  --timepoint_id Oct_2025 \
  --timepoint_root /path/to/time_points/Oct_2025 \
  --prediction_target_mode test_sequences \
  --democafa_package /path/to/democafa_package
```

### Evaluate a release window with existing timepoint predictions

Omit `--window_prediction_mode` when `T0/predictions/` already contains
timepoint-level prediction files from a previous timepoint build. The workflow
still builds ground truth between `T0` and `T1`, then runs `cafaeval` with the
existing predictions in `T0/predictions/`. It errors early if that directory is
missing or contains no `*_predictions.tsv` files.

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

This mode is appropriate when predictions were generated against the full
timepoint test set and are already available under `T0/predictions/`.

### Generate predictions for one ground-truth window

Use `--window_prediction_mode generate` when full timepoint-level GPU inference
is not available. The workflow first builds the ground truth between `T0` and
`T1`, materializes `groundtruth_targets.fasta`, then runs the enabled methods
against that window-specific target FASTA only.

```bash
nextflow run main.nf \
  --mode evaluate_window \
  --t0_id Jun_2025 \
  --t1_id Oct_2025 \
  --t0_root /path/to/time_points/Jun_2025 \
  --t1_root /path/to/time_points/Oct_2025 \
  --publish_root /path/to/data/releases \
  --window_prediction_mode generate \
  --enabled_window_methods naive,blast,prott5 \
  --democafa_package /path/to/democafa_package
```

On the HPC cluster, add the Slurm profile:

```bash
nextflow run main.nf \
  --mode evaluate_window \
  --t0_id Jun_2025 \
  --t1_id Oct_2025 \
  --t0_root /path/to/time_points/Jun_2025 \
  --t1_root /path/to/time_points/Oct_2025 \
  --publish_root /path/to/data/releases \
  --window_prediction_mode generate \
  --enabled_window_methods naive,blast,prott5 \
  -profile slurm
```

In explicit window-prediction mode this workflow:

- intersects the target universe between `T0` and `T1`
- classifies ground truth into `NK`, `LK`, and `PK`
- supplies T1's validated `release/go-basic_frozen.obo` as the ontology that
  methods could access before the annotation-accumulation period
- materializes `groundtruth_targets.fasta`
- runs enabled methods against that window-specific FASTA
- writes window predictions under `T0/predictions_by_window/<window_id>/`
- publishes a frontend-ready release under `data/releases/<window_id>/`

### Reuse existing predictions for one ground-truth window

Use `--window_prediction_mode existing` when the window-specific prediction
files have already been generated. The workflow rebuilds/validates the same
ground-truth window, copies the supplied prediction files into the working
prediction bundle, and evaluates those predictions against the window-specific
ground truth.

```bash
nextflow run main.nf \
  --mode evaluate_window \
  --t0_id Jun_2025 \
  --t1_id Oct_2025 \
  --t0_root /path/to/time_points/Jun_2025 \
  --t1_root /path/to/time_points/Oct_2025 \
  --publish_root /path/to/data/releases \
  --window_prediction_mode existing \
  --window_predictions_dir /path/to/time_points/Jun_2025/predictions_by_window/Jun_2025_Oct_2025 \
  --enabled_window_methods naive,blast,prott5 \
  --democafa_package /path/to/democafa_package
```

For `existing` mode, `--enabled_window_methods` should match the prediction
files present in `--window_predictions_dir`. For `generate` mode, it controls
which method processes are launched. In both explicit window modes, evaluation
is restricted to targets in the generated `groundtruth_targets.fasta`.

### Recover from failed prediction jobs

The Slurm profile submits predictor tasks through Nextflow, not through `sbatch`
inside a process. Some method failures are intentionally tolerated so long
running jobs can continue:

- `PREDICT_NAIVE`, `PREDICT_GOA_NONEXP`, `PREDICT_BLAST`,
  `PREDICT_PROTT5`, and `PREDICT_TRANSFEW` use `errorStrategy = 'ignore'`
  in `evaluate_window`
- `PREDICT_FUNBIND_BATCH` retries twice, then ignores the failed batch
- `PREDICT_DEEPGOPLUS` retries up to three times and then fails

This does not mean a release can always publish successfully. Before assembly,
`evaluate_window` checks that every file implied by `--enabled_window_methods`
exists. A failed `blast` run, for example, still causes assembly to abort with a
missing `blast_predictions.tsv` error if `blast` is enabled.

The practical recovery path is:

1. Inspect the failed method work directory in the Nextflow log.
2. Rerun only the failed method if possible by narrowing
   `--enabled_window_methods`.
3. Once the expected prediction files are present in a stable directory, rerun
   evaluation with `--window_prediction_mode existing` and
   `--window_predictions_dir`.

Treat `-resume` as best effort for this backend. It is useful for deterministic
data-prep steps, but it has been unreliable after ignored or retried Slurm
predictor failures because missing outputs, dynamic FunBind batch grouping, and
side-effect publication directories do not always recreate the same cacheable
task graph. For production recovery, prefer the explicit `existing` prediction
mode over relying on `-resume` to repair a partially completed window run.

### Evaluate late predictions

Use this to evaluate registered, window-specific predictions against an existing
release. It produces a validated candidate by default and does not write to T0.
See [late-prediction inputs, provenance and publication](docs/late_predictions.md).

```bash
nextflow run main.nf \
  --mode evaluate_late_predictions \
  --t0_id Mar_2026 --release_id Mar_2026_Jun_2026 --methods protea \
  --release_root /path/to/published/Mar_2026_Jun_2026 \
  --late_predictions_dir /path/to/method_runs/session/Mar_2026_Jun_2026 \
  --late_reference_dir /path/to/archived/window_release \
  --late_evaluation_context /path/to/evaluation_context.json \
  --late_publish false
```

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
- `evaluate_window` is sequential through ground-truth construction. In `generate` mode it then fans out by method; otherwise it reuses existing prediction files before evaluating `NK`, `LK`, and `PK`.
- `evaluate_late_predictions` evaluates selected window-specific methods, preserves existing results, and produces a validated candidate; publication is explicit.

### Artifact Map

Canonical timepoint artifacts live under `<T0>/release/` and may also include:

- `<T0>/test_sequences_split/`
- `<T0>/predictions/` in full-set mode
- `<T0>/predictions_by_window/<window_id>/` in `window_groundtruth` mode

Published release windows live under `data/releases/<T0>_<T1>/`.

### Notes On Current Ops Assumptions

- `democafa` dependencies are provisioned once in its environment; Nextflow tasks
  import the configured source checkout through `PYTHONPATH` and do not run
  concurrent package installations.
- `cafaeval` is treated as an external executable.
- Predictor containers and cache paths are still configured through local absolute paths in [nextflow.config](nextflow.config) and the prediction modules; those should be cleaned up before public multi-user deployment.

### Container Method Onboarding

Before a new containerized method is added to Nextflow, use
[check_container_image.md](check_container_image.md) as the engineer-facing
acceptance checklist and [container_info_sheet.md](container_info_sheet.md) as
the questionnaire for method developers. The intended handoff is a working
container, tested commands, resource estimates, method-card text, and a final
`<method>_predictions.tsv` contract before any Nextflow module is written.

### Caveats And Open Decisions

- Real validation is still incomplete. Stub smoke tests exercise graph shape,
  but they do not prove scientific correctness, container bindings, GPU
  availability, or wall-clock viability on Slurm.
- The implementation default for `evaluate_window` is currently to reuse
  `T0/predictions/` when `--window_prediction_mode` is omitted. This differs
  from the newer constrained-compute design where window-scoped prediction is
  the default mental model. Decide whether the code default should switch to
  `generate`, or whether operators should always pass the desired mode
  explicitly.
- `funbind_allow_missing_batches = true` lets a window evaluate a partial
  FunBind prediction if some batches fail. Decide whether production releases
  should allow that, or require all batches to finish.
- `timepoint_build` now requires `uniprot_sprot.dat.gz` and merges DAT-derived
  annotations into `train_terms.tsv`. Existing staged historical timepoints may
  need this file added before rebuilding, and the process spec should be kept in
  sync if this DAT requirement is intentional.
- The workflow has compatibility aliases for snake_case and camelCase
  parameters. Decide which names are canonical before this becomes a public
  interface.
- `--query_fasta` bypasses the normal T0/T1 target intersection. Use it only
  when the provided FASTA is known to be compatible with both timepoints; a
  validation process should be added before treating this as production-safe.
