#!/usr/bin/env python3
"""
Data generation script for LAFA frontend.

Transforms TSV data files into JSON format for the React frontend.
Reads from data/releases/ and outputs to frontend/public/data/.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = FRONTEND_DIR.parent
DATA_ROOT = REPO_ROOT / "data"
RELEASES_DIR = DATA_ROOT / "releases"
OUTPUT_DIR = FRONTEND_DIR / "public" / "data"

# Import config from app module
sys.path.insert(0, str(REPO_ROOT / "app"))
try:
    from config import (
        DATA_DATES,
        METHOD_HELP_MSG,
        METHOD_DOCKER_URLS,
        BASELINE_METHOD_LABELS,
        GO_ASPECTS,
    )
except ImportError:
    print("Warning: Could not import from app/config.py, using defaults")
    DATA_DATES = {}
    METHOD_HELP_MSG = {}
    METHOD_DOCKER_URLS = {}
    BASELINE_METHOD_LABELS = set()
    GO_ASPECTS = {
        "biological_process": "Biological Process",
        "molecular_function": "Molecular Function",
        "cellular_component": "Cellular Component",
    }

SUBSETS = ["NK", "LK", "PK"]
ASPECT_SHORT = {"biological_process": "P", "molecular_function": "F", "cellular_component": "C"}
ASPECT_LONG = {v: k for k, v in ASPECT_SHORT.items()}


def parse_tsv(path: Path) -> list[dict]:
    """Parse a TSV file into a list of dicts."""
    if not path.exists():
        print(f"  Warning: File not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    if not lines:
        return []

    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        row = {}
        for i, header in enumerate(headers):
            row[header] = values[i] if i < len(values) else ""
        rows.append(row)
    return rows


def split_release_id(release_id: str) -> tuple[str, str]:
    """Split a release folder name into start and end time points."""
    parts = str(release_id).split("_")
    if len(parts) != 4:
        raise ValueError(f"Release id must follow Mon_YYYY_Mon_YYYY: {release_id}")
    return "_".join(parts[:2]), "_".join(parts[2:])


def parse_timepoint_label(timepoint_label: str) -> datetime:
    """Parse a Mon_YYYY label into a sortable datetime."""
    return datetime.strptime(str(timepoint_label), "%b_%Y")


def generate_catalog() -> dict:
    """Generate catalog.json from existing catalog and release directories."""
    catalog_path = RELEASES_DIR / "catalog.json"

    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            raw_catalog = json.load(f)
        raw_releases = raw_catalog.get("releases", raw_catalog)
    else:
        # Discover releases from directories
        raw_releases = []
        for item in sorted(RELEASES_DIR.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                raw_releases.append({
                    "release_id": item.name,
                    "path": str(item.relative_to(REPO_ROOT)),
                    "status": "ready"
                })

    releases = []
    timepoints = set()

    for entry in raw_releases:
        release_id = entry.get("release_id") or entry.get("id") or entry.get("name")
        status = entry.get("status", "ready")

        if not release_id:
            continue

        try:
            start_tp, end_tp = split_release_id(release_id)
            timepoints.add(start_tp)
            timepoints.add(end_tp)

            releases.append({
                "id": release_id,
                "startTimepoint": start_tp,
                "endTimepoint": end_tp,
                "status": status
            })
        except ValueError as e:
            print(f"  Warning: {e}")
            continue

    # Sort releases by timepoints
    releases.sort(key=lambda r: (parse_timepoint_label(r["startTimepoint"]),
                                  parse_timepoint_label(r["endTimepoint"])))

    # Sort timepoints
    sorted_timepoints = sorted(timepoints, key=parse_timepoint_label)

    return {
        "releases": releases,
        "timepoints": sorted_timepoints,
        "generatedAt": datetime.now(timezone.utc).isoformat()
    }


def generate_methods_config() -> dict:
    """Generate methods.json with help text, URLs, and baseline flags."""
    methods = {}

    # Combine all known methods from various sources
    all_method_names = set(METHOD_HELP_MSG.keys()) | set(METHOD_DOCKER_URLS.keys()) | BASELINE_METHOD_LABELS

    for name in all_method_names:
        methods[name] = {
            "label": name,
            "description": METHOD_HELP_MSG.get(name, ""),
            "dockerUrl": METHOD_DOCKER_URLS.get(name, ""),
            "isBaseline": name in BASELINE_METHOD_LABELS
        }

    return {
        "methods": methods,
        "aspects": GO_ASPECTS,
        "subsets": {
            "NK": "No Knowledge - targets with no prior experimental annotations",
            "LK": "Limited Knowledge - targets with some prior experimental annotations",
            "PK": "Prior Knowledge - targets with prior experimental annotations"
        }
    }


def count_groundtruth_targets(release_dir: Path) -> dict:
    """Count targets by subset and aspect from groundtruth files."""
    counts = {}

    for subset in SUBSETS:
        gt_path = release_dir / f"groundtruth_{subset}.tsv"
        rows = parse_tsv(gt_path)

        if not rows:
            counts[subset] = {"total": 0, "byAspect": {}}
            continue

        # Count unique EntryIDs
        entry_ids = set(row.get("EntryID", "") for row in rows)
        total = len(entry_ids)

        # Count by aspect
        aspect_counts = defaultdict(set)
        for row in rows:
            entry_id = row.get("EntryID", "")
            aspect_code = row.get("aspect", "")
            if entry_id and aspect_code:
                aspect_name = ASPECT_LONG.get(aspect_code, aspect_code)
                aspect_counts[aspect_name].add(entry_id)

        counts[subset] = {
            "total": total,
            "byAspect": {
                aspect: len(ids) for aspect, ids in aspect_counts.items()
            }
        }

    return counts


def generate_release_meta(release_id: str, release_dir: Path) -> dict:
    """Generate meta.json for a release."""
    start_tp, end_tp = split_release_id(release_id)

    start_dates = DATA_DATES.get(start_tp, {})
    end_dates = DATA_DATES.get(end_tp, {})

    return {
        "releaseId": release_id,
        "startTimepoint": start_tp,
        "endTimepoint": end_tp,
        "dates": {
            "goaStart": start_dates.get("goa", "N/A"),
            "goaEnd": end_dates.get("goa", "N/A"),
            "uniprotStart": start_dates.get("uniprot", "N/A"),
            "uniprotEnd": end_dates.get("uniprot", "N/A")
        },
        "targetCounts": count_groundtruth_targets(release_dir)
    }


def generate_release_methods(release_dir: Path) -> dict:
    """Generate methods.json for a release from method_names and method_availability."""
    methods = {}

    # Read method names
    names_rows = parse_tsv(release_dir / "method_names.tsv")
    for row in names_rows:
        filename = row.get("filename", "")
        label = row.get("label", "")
        group = row.get("group", "")
        if label:
            methods[label] = {
                "filename": filename,
                "label": label,
                "group": group,
                "availability": {}
            }

    # Read availability
    avail_rows = parse_tsv(release_dir / "method_availability.tsv")
    for row in avail_rows:
        method_name = row.get("method", "")
        if method_name in methods:
            for subset in SUBSETS:
                methods[method_name]["availability"][subset] = row.get(subset, "").lower() == "true"

    return {"methods": methods}


def generate_best_metrics(release_dir: Path) -> dict:
    """Generate best.json with best metrics by subset and aspect."""
    best = {}

    for subset in SUBSETS:
        best_path = release_dir / f"results_{subset}" / "evaluation_best_f_micro_w.tsv"
        rows = parse_tsv(best_path)

        if not rows:
            continue

        for row in rows:
            filename = row.get("filename", "")
            aspect = row.get("ns", "")  # namespace

            # Extract method label from filename
            method_label = filename.replace("_predictions.tsv", "").replace("_", " ").title()

            # Map back to actual method name using method_names.tsv
            names_rows = parse_tsv(release_dir / "method_names.tsv")
            for name_row in names_rows:
                if name_row.get("filename", "") == filename:
                    method_label = name_row.get("label", method_label)
                    break

            key = f"{subset}_{aspect}"

            try:
                entry = {
                    "method": method_label,
                    "subset": subset,
                    "aspect": aspect,
                    "precision": float(row.get("pr_micro_w", 0)),
                    "recall": float(row.get("rc_micro_w", 0)),
                    "fmax": float(row.get("f_micro_w", 0)),
                    "coverage": float(row.get("cov_w", 0)),
                    "threshold": float(row.get("tau", 0)),
                    "n": float(row.get("n_w", row.get("n", 0)))
                }

                if key not in best:
                    best[key] = []
                best[key].append(entry)
            except (ValueError, TypeError) as e:
                print(f"  Warning: Could not parse metrics for {filename}/{aspect}: {e}")

    # Sort each list by fmax descending
    for key in best:
        best[key].sort(key=lambda x: x["fmax"], reverse=True)

    return best


def generate_curves(release_dir: Path) -> dict:
    """Generate curves.json with PR curve points."""
    curves = {}

    for subset in SUBSETS:
        all_path = release_dir / f"results_{subset}" / "evaluation_all.tsv"
        rows = parse_tsv(all_path)

        if not rows:
            continue

        # Load method name mappings
        name_map = {}
        names_rows = parse_tsv(release_dir / "method_names.tsv")
        for name_row in names_rows:
            name_map[name_row.get("filename", "")] = name_row.get("label", "")

        # Group by method and aspect
        grouped = defaultdict(list)
        for row in rows:
            filename = row.get("filename", "")
            aspect = row.get("ns", "")
            method = name_map.get(filename, filename.replace("_predictions.tsv", ""))

            try:
                point = {
                    "tau": float(row.get("tau", 0)),
                    "precision": float(row.get("pr_micro_w", 0)),
                    "recall": float(row.get("rc_micro_w", 0))
                }
                key = f"{subset}_{aspect}_{method}"
                grouped[key].append(point)
            except (ValueError, TypeError):
                continue

        # Sort points by tau
        for key, points in grouped.items():
            points.sort(key=lambda p: p["tau"])
            curves[key] = points

    return curves


def process_release(release_id: str, release_dir: Path, output_dir: Path):
    """Process a single release and generate all JSON files."""
    release_output = output_dir / "releases" / release_id
    release_output.mkdir(parents=True, exist_ok=True)

    print(f"  Generating meta.json...")
    meta = generate_release_meta(release_id, release_dir)
    with open(release_output / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Generating methods.json...")
    methods = generate_release_methods(release_dir)
    with open(release_output / "methods.json", "w", encoding="utf-8") as f:
        json.dump(methods, f, indent=2)

    print(f"  Generating best.json...")
    best = generate_best_metrics(release_dir)
    with open(release_output / "best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"  Generating curves.json...")
    curves = generate_curves(release_dir)
    with open(release_output / "curves.json", "w", encoding="utf-8") as f:
        json.dump(curves, f)  # No indent for size efficiency


def main():
    """Main entry point."""
    print(f"LAFA Frontend Data Generator")
    print(f"=" * 40)
    print(f"Data source: {RELEASES_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate catalog
    print("Generating catalog.json...")
    catalog = generate_catalog()
    with open(OUTPUT_DIR / "catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"  Found {len(catalog['releases'])} releases, {len(catalog['timepoints'])} timepoints")

    # Generate methods config
    print("\nGenerating methods.json...")
    methods_config = generate_methods_config()
    with open(OUTPUT_DIR / "methods.json", "w", encoding="utf-8") as f:
        json.dump(methods_config, f, indent=2)
    print(f"  Configured {len(methods_config['methods'])} methods")

    # Process each release
    print("\nProcessing releases...")
    for release in catalog["releases"]:
        release_id = release["id"]
        release_dir = RELEASES_DIR / release_id

        if not release_dir.exists():
            print(f"\n  Skipping {release_id}: directory not found")
            continue

        print(f"\n[{release_id}]")
        process_release(release_id, release_dir, OUTPUT_DIR)

    print("\n" + "=" * 40)
    print("Data generation complete!")

    # Print summary
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.json"))
    print(f"Total output size: {total_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
