# CAFA Forever

CAFA Forever has two main components:

- a Streamlit frontend for browsing published LAFA evaluation releases
- a Nextflow-based backend for building timepoints and publishing release windows

The frontend and backend share the same repository, but they do not read the same directories at runtime.

## Frontend Overview

The website reads only validated, published release windows under `data/releases/`, optionally filtered by `data/releases/catalog.json`.

Important implications:

- the website does not read raw timepoint build directories
- the website does not read Nextflow `work/` directories
- the website does not read unpublished staging outputs

Each frontend-visible release corresponds to a release window such as `Jun_2025_Oct_2025`, not to a single raw timepoint snapshot. The app discovers available releases from `data/releases/`, derives the available time points from those release ids, and uses the published files in each release directory for plotting and tables.

## Run The Frontend Locally

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then launch Streamlit from the repo root:

```bash
streamlit run app/streamlit_plot.py
```

The frontend expects the release contract under `data/releases/` to already exist.

## Docker / Website Deployment

The Docker image runs the Streamlit frontend on port `8501`. In the current deployment model:

- the container serves Streamlit on `8501`
- Nginx on the host terminates HTTP/HTTPS
- Nginx proxies requests to `127.0.0.1:8501`

With Docker Compose:

```bash
docker compose up -d --build
```

The current [docker-compose.yml](docker-compose.yml) binds the app to `127.0.0.1:8501:8501`, which is appropriate when Nginx is the public entrypoint.

## What The Frontend Displays

For each published release window, the app uses:

- `method_names.tsv`
- `method_availability.tsv` when present
- `groundtruth_NK.tsv`
- `groundtruth_LK.tsv`
- `groundtruth_PK.tsv`
- `results_NK/evaluation_best_f_micro_w.tsv`
- `results_NK/evaluation_all.tsv`
- `results_LK/evaluation_best_f_micro_w.tsv`
- `results_LK/evaluation_all.tsv`
- `results_PK/evaluation_best_f_micro_w.tsv`
- `results_PK/evaluation_all.tsv`

The sidebar lets users choose one or two release windows for comparison. From those windows, the app derives the available time points shown in the release sliders.

## Backend Pointer

Backend setup and Nextflow usage are documented in [README_backend.md](README_backend.md).

In short, the backend is responsible for:

- building canonical timepoint artifacts
- generating or reusing predictions
- evaluating `NK`, `LK`, and `PK`
- publishing frontend-ready release windows into `data/releases/`

## Repository Layout

Key paths:

- [app/streamlit_app.py](app/streamlit_app.py): Streamlit application
- [app/config.py](app/config.py): frontend release discovery and plotting config
- [main.nf](main.nf): backend Nextflow entrypoint
- [workflows/](workflows): top-level backend workflows
- [modules/local/](modules/local): reusable Nextflow processes
- [data/releases/](data/releases): published frontend-consumable releases

## Notes

The precision-recall curves are displayed with a monotonic precision option, using a cumulative maximum over precision values. This makes curves easier to compare when small evaluation sets or threshold noise would otherwise introduce non-monotonic segments.
