process GENERATE_METHOD_NAMES {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(predictionsDir), val(methodNamesFile)

    output:
    tuple val(meta), path('method_names.tsv')

    script:
    """
    python3 - "${predictionsDir}" "${methodNamesFile}" <<'PY'
import csv
import pathlib
import sys

pred_dir = pathlib.Path(sys.argv[1])
method_names_path = sys.argv[2]
prediction_filenames = sorted(path.name for path in pred_dir.glob("*.tsv"))
labels = {
    "blast_predictions.tsv": "BLAST",
    "deepgoplus_predictions.tsv": "DeepGOPlus",
    "funbind_predictions.tsv": "FunBind",
    "goa_nonexp_predictions.tsv": "GOA Non-exp",
    "naive_predictions.tsv": "Naive",
    "prott5_predictions.tsv": "ProtT5",
    "transfew_predictions.tsv": "TransFew",
}

if method_names_path and pathlib.Path(method_names_path).is_file():
    with open(method_names_path) as handle:
        reader = csv.DictReader(handle, delimiter="\\t")
        rows = [row for row in reader if row.get("filename") in prediction_filenames]

    with open("method_names.tsv", "w") as out:
        fieldnames = ["filename", "label", "group"]
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\\t", lineterminator="\\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "filename": row.get("filename", ""),
                "label": row.get("label", ""),
                "group": row.get("group", row.get("label", "")),
            })
else:
    with open("method_names.tsv", "w") as out:
        out.write("filename\\tlabel\\tgroup\\n")
        for filename in prediction_filenames:
            label = labels.get(filename, pathlib.Path(filename).stem.replace("_", " ").title())
            out.write(f"{filename}\\t{label}\\t{label}\\n")
PY
    """
}

process GENERATE_METHOD_AVAILABILITY {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(methodNamesTsv), path(nkDir), path(lkDir), path(pkDir)

    output:
    tuple val(meta), path('method_availability.tsv')

    script:
    """
    python3 - "${methodNamesTsv}" "${nkDir}/evaluation_best_f_micro_w.tsv" "${lkDir}/evaluation_best_f_micro_w.tsv" "${pkDir}/evaluation_best_f_micro_w.tsv" <<'PY'
import csv
import sys

method_names = {}
with open(sys.argv[1]) as handle:
    for row in csv.DictReader(handle, delimiter="\\t"):
        method_names[row["filename"]] = row["label"]

def methods_present(path):
    present = set()
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\\t"):
            present.add(row["filename"])
    return present

nk = methods_present(sys.argv[2])
lk = methods_present(sys.argv[3])
pk = methods_present(sys.argv[4])

with open("method_availability.tsv", "w") as out:
    out.write("method\\tNK\\tLK\\tPK\\n")
    for filename, label in sorted(method_names.items(), key=lambda kv: kv[1]):
        out.write(f"{label}\\t{'true' if filename in nk else 'false'}\\t{'true' if filename in lk else 'false'}\\t{'true' if filename in pk else 'false'}\\n")
PY
    """
}

process PACKAGE_RELEASE_DIRECTORY {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(nkTsv), path(lkTsv), path(pkTsv), path(pkKnownTsv), path(toiFile), path(targetsTsv), path(targetsFasta), path(nkDir), path(lkDir), path(pkDir), path(methodNamesTsv), path(methodAvailTsv)

    output:
    tuple val(meta), path("${meta.release_id}")

    script:
    """
    mkdir -p "${meta.release_id}/results_NK" "${meta.release_id}/results_LK" "${meta.release_id}/results_PK"
    cp "${methodNamesTsv}" "${meta.release_id}/method_names.tsv"
    cp "${methodAvailTsv}" "${meta.release_id}/method_availability.tsv"
    cp "${nkTsv}" "${meta.release_id}/groundtruth_NK.tsv"
    cp "${lkTsv}" "${meta.release_id}/groundtruth_LK.tsv"
    cp "${pkTsv}" "${meta.release_id}/groundtruth_PK.tsv"
    cp "${pkKnownTsv}" "${meta.release_id}/groundtruth_PK_known.tsv"
    cp "${toiFile}" "${meta.release_id}/groundtruth_terms_of_interest.txt"
    cp "${targetsTsv}" "${meta.release_id}/groundtruth_targets.tsv"
    cp "${targetsFasta}" "${meta.release_id}/groundtruth_targets.fasta"
    cp -r "${nkDir}/." "${meta.release_id}/results_NK/"
    cp -r "${lkDir}/." "${meta.release_id}/results_LK/"
    cp -r "${pkDir}/." "${meta.release_id}/results_PK/"
    """
}

process VALIDATE_RELEASE_DIRECTORY {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(releaseDir)

    output:
    tuple val(meta), path(releaseDir)

    script:
    """
    python3 - "${releaseDir}" <<'PY'
import csv
import pathlib
import sys

release_dir = pathlib.Path(sys.argv[1])
required = [
    "method_names.tsv",
    "groundtruth_NK.tsv",
    "groundtruth_LK.tsv",
    "groundtruth_PK.tsv",
    "groundtruth_PK_known.tsv",
    "groundtruth_terms_of_interest.txt",
    "groundtruth_targets.tsv",
    "groundtruth_targets.fasta",
    "results_NK/evaluation_best_f_micro_w.tsv",
    "results_NK/evaluation_all.tsv",
    "results_LK/evaluation_best_f_micro_w.tsv",
    "results_LK/evaluation_all.tsv",
    "results_PK/evaluation_best_f_micro_w.tsv",
    "results_PK/evaluation_all.tsv",
]
for rel in required:
    path = release_dir / rel
    if not path.exists():
        raise SystemExit(f"Missing required release artifact: {path}")

for gt_name in ("groundtruth_NK.tsv", "groundtruth_LK.tsv", "groundtruth_PK.tsv"):
    with (release_dir / gt_name).open() as handle:
        reader = csv.DictReader(handle, delimiter="\\t")
        expected = {"EntryID", "term", "aspect"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{gt_name} missing columns: {sorted(missing)}")
        for row in reader:
            if row["aspect"] not in {"P", "F", "C"}:
                raise SystemExit(f"{gt_name} has invalid aspect: {row['aspect']}")

for subset in ("NK", "LK", "PK"):
    for table, required_cols in {
        "evaluation_best_f_micro_w.tsv": {"filename", "ns", "tau", "n", "pr_micro_w", "rc_micro_w", "f_micro_w", "cov_w"},
        "evaluation_all.tsv": {"filename", "ns", "tau", "cov", "rc_micro_w", "pr_micro_w", "f_micro_w"},
    }.items():
        with (release_dir / f"results_{subset}" / table).open() as handle:
            reader = csv.DictReader(handle, delimiter="\\t")
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"results_{subset}/{table} missing columns: {sorted(missing)}")
PY
    """
}

process PUBLISH_RELEASE_DIRECTORY {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(releaseDir)

    output:
    tuple val(meta), path(releaseDir)

    script:
    """
    publish_root="${file(params.publish_root)}"
    target_dir="\${publish_root}/${meta.release_id}"
    staging_dir="\${publish_root}/.${meta.release_id}.staging"

    mkdir -p "\${publish_root}"
    rm -rf "\${staging_dir}"
    mkdir -p "\${staging_dir}"
    cp -r "${releaseDir}/." "\${staging_dir}/"
    rm -rf "\${target_dir}"
    mv "\${staging_dir}" "\${target_dir}"
    test -d "\${target_dir}"
    """

    stub:
    """
    publish_root="${file(params.publish_root)}"
    target_dir="\${publish_root}/${meta.release_id}"
    staging_dir="\${publish_root}/.${meta.release_id}.staging"

    mkdir -p "\${publish_root}"
    rm -rf "\${staging_dir}"
    mkdir -p "\${staging_dir}"
    cp -r "${releaseDir}/." "\${staging_dir}/"
    rm -rf "\${target_dir}"
    mv "\${staging_dir}" "\${target_dir}"
    test -d "\${target_dir}"
    """
}

process BUILD_RELEASE_CATALOG {
    label 'packaging'
    tag "${meta.release_id}"
    publishDir "${params.publish_root}", mode: 'copy'

    input:
    tuple val(meta), path(releaseDir)

    output:
    tuple val(meta), path('catalog.json')

    script:
    """
    python3 - "${file(params.workspace_root)}" "${file(params.publish_root)}" <<'PY'
import importlib.util
import json
import pathlib
import sys
from datetime import datetime, timezone

repo_root = pathlib.Path(sys.argv[1]).resolve()
publish_dir = pathlib.Path(sys.argv[2]).resolve()
publish_dir.mkdir(parents=True, exist_ok=True)

module_path = repo_root / "scripts" / "publish_release_windows.py"
spec = importlib.util.spec_from_file_location("publish_release_windows", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

release_dirs = sorted(
    path for path in publish_dir.iterdir()
    if path.is_dir() and not path.name.startswith(".")
)
results = [module.validate_release_window(path) for path in release_dirs]
catalog = {
    "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "validator": "scripts/publish_release_windows.py",
    "releases": module.build_catalog_entries(results, publish_dir, repo_root=repo_root),
}

with open("catalog.json", "w", encoding="utf-8") as handle:
    json.dump(catalog, handle, indent=2)
    handle.write("\\n")
PY
    """

    stub:
    """
    python3 - "${file(params.workspace_root)}" "${file(params.publish_root)}" "${releaseDir}" <<'PY'
import json
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
publish_dir = pathlib.Path(sys.argv[2]).resolve()
release_dir = pathlib.Path(sys.argv[3]).resolve()
publish_dir.mkdir(parents=True, exist_ok=True)

release_ids = {
    path.name
    for path in publish_dir.iterdir()
    if path.is_dir() and not path.name.startswith(".")
}
release_ids.add(release_dir.name)

catalog = {
    "generated_at": "1970-01-01T00:00:00+00:00",
    "validator": "scripts/publish_release_windows.py",
    "releases": [
        {
            "release_id": release_id,
            "path": (
                (publish_dir / release_id).resolve().relative_to(repo_root).as_posix()
                if (publish_dir / release_id).resolve().is_relative_to(repo_root)
                else str((publish_dir / release_id).resolve())
            ),
            "status": "ready",
        }
        for release_id in sorted(release_ids)
    ],
}

with open("catalog.json", "w", encoding="utf-8") as handle:
    json.dump(catalog, handle, indent=2)
    handle.write("\\n")
PY
    """
}
