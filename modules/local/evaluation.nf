process INTERSECT_TARGETS {
    label 'cpu_small'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile)

    output:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path('diff_test_sequences_common.fasta')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate democafaenv
    pip install "${params.democafa_package}"

    python3 -m democafa.datacollection.compare_fasta \
      "${t0Dir}/test_sequences.fasta" \
      "${t1Dir}/test_sequences.fasta" \
      diff_test_sequences.fasta

    test -f diff_test_sequences_common.fasta
    """

    stub:
    """
    touch diff_test_sequences_common.fasta
    """
}

process CLASSIFY_GROUNDTRUTH {
    label 'cpu_medium'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path(commonFasta)

    output:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path(commonFasta), path('groundtruth_NK.tsv'), path('groundtruth_LK.tsv'), path('groundtruth_PK.tsv'), path('groundtruth_PK_known.tsv'), path('groundtruth_terms_of_interest.txt'), path('groundtruth_targets.tsv')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate democafaenv
    pip install "${params.democafa_package}"

    python3 -m democafa.groundtruth.classify_ground_truth \
      --annot_known "${t0Dir}/train_terms_propagated.tsv" \
      --annot2 "${t1Dir}/train_terms_propagated.tsv" \
      --query_file "${commonFasta}" \
      --graph "${t0Dir}/go-basic.obo" \
      --graph2 "${t1Dir}/go-basic.obo" \
      --out_prefix groundtruth.tsv
    test -f groundtruth_targets.tsv
    test -f groundtruth_terms_of_interest.txt
    """

    stub:
    """
    printf 'EntryID\tterm\taspect\n' > groundtruth_NK.tsv
    printf 'EntryID\tterm\taspect\n' > groundtruth_LK.tsv
    printf 'EntryID\tterm\taspect\n' > groundtruth_PK.tsv
    printf 'EntryID\tterm\taspect\n' > groundtruth_PK_known.tsv
    printf 'GO:0000001\n' > groundtruth_terms_of_interest.txt
    printf 'EntryID\n' > groundtruth_targets.tsv
    """
}

process EXTRACT_GROUNDTRUTH_TARGETS {
    label 'cpu_small'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path(commonFasta), path(nkTsv), path(lkTsv), path(pkTsv), path(pkKnownTsv), path(toiFile), path(targetsTsv)

    output:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path(commonFasta), path(nkTsv), path(lkTsv), path(pkTsv), path(pkKnownTsv), path(toiFile), path(targetsTsv), path('groundtruth_targets.fasta')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate democafaenv
    pip install "${params.democafa_package}"

    python3 -m democafa.datacollection.retrieve_sequences \
      --fasta "${commonFasta}" \
      --input "${targetsTsv}" \
      --out_fasta groundtruth_targets.fasta
    """

    stub:
    """
    touch groundtruth_targets.fasta
    """
}

process PREPARE_WINDOW_RELEASE {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path(commonFasta), path(nkTsv), path(lkTsv), path(pkTsv), path(pkKnownTsv), path(toiFile), path(targetsTsv), path(targetsFasta)

    output:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path(commonFasta), path(nkTsv), path(lkTsv), path(pkTsv), path(pkKnownTsv), path(toiFile), path(targetsTsv), path(targetsFasta), path('window_release')

    script:
    """
    mkdir -p window_release
    cp "${t0Dir}/release/go-basic.obo" window_release/go-basic.obo
    cp "${t0Dir}/release/goa_uniprot_sprot.gaf.gz" window_release/goa_uniprot_sprot.gaf.gz
    cp "${t0Dir}/release/train_terms.tsv" window_release/train_terms.tsv
    cp "${t0Dir}/release/train_sequences.fasta" window_release/train_sequences.fasta
    cp "${t0Dir}/release/train_terms_propagated.tsv" window_release/train_terms_propagated.tsv
    cp "${t0Dir}/release/train_taxonomy.tsv" window_release/train_taxonomy.tsv
    cp "${t0Dir}/release/blast_results.tsv" window_release/blast_results.tsv
    cp "${t0Dir}/release/IA.tsv" window_release/IA.tsv
    cp "${targetsFasta}" window_release/groundtruth_targets.fasta
    """

    stub:
    """
    mkdir -p window_release
    touch window_release/go-basic.obo
    touch window_release/goa_uniprot_sprot.gaf.gz
    touch window_release/train_terms.tsv
    touch window_release/train_sequences.fasta
    touch window_release/train_terms_propagated.tsv
    touch window_release/train_taxonomy.tsv
    touch window_release/blast_results.tsv
    touch window_release/IA.tsv
    touch window_release/groundtruth_targets.fasta
    """
}

process SPLIT_WINDOW_TARGETS {
    label 'cpu_small'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path(commonFasta), path(nkTsv), path(lkTsv), path(pkTsv), path(pkKnownTsv), path(toiFile), path(targetsTsv), path(targetsFasta), path(windowRelease)

    output:
    tuple val(meta), path(t0Dir), path(t1Dir), val(methodNamesFile), path(commonFasta), path(nkTsv), path(lkTsv), path(pkTsv), path(pkKnownTsv), path(toiFile), path(targetsTsv), path(targetsFasta), path('window_release_ready')

    script:
    """
    cp -r "${windowRelease}" window_release_ready
    mkdir -p window_release_ready/groundtruth_targets_split
    python3 - "window_release_ready/groundtruth_targets.fasta" "${params.split_size}" <<'PY'
import pathlib
import sys

fasta_path = pathlib.Path(sys.argv[1])
chunk_size = int(sys.argv[2])
out_dir = pathlib.Path("window_release_ready/groundtruth_targets_split")

records = []
header = None
seq = []
with fasta_path.open() as handle:
    for line in handle:
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header = line.rstrip()
            seq = []
        else:
            seq.append(line)
    if header is not None:
        records.append((header, "".join(seq)))

for idx in range(0, len(records), chunk_size):
    chunk = records[idx:idx + chunk_size]
    part = out_dir / f"part_{idx // chunk_size + 1:03d}.fasta"
    with part.open("w") as out:
        for hdr, sequence in chunk:
            out.write(f"{hdr}\\n{sequence}")
PY
    """

    stub:
    """
    cp -r "${windowRelease}" window_release_ready
    mkdir -p window_release_ready/groundtruth_targets_split
    touch window_release_ready/groundtruth_targets_split/part_001.fasta
    """
}

process EVALUATE_NK {
    label 'cpu_medium'
    tag "${meta.release_id}:NK"

    input:
    tuple val(meta), path(t0Dir), path(predictionsDir), path(groundtruthTsv), path(toiFile)

    output:
    tuple val(meta), path('results_NK')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate cafa5-evaluator

    cafaeval "${t0Dir}/release/go-basic.obo" "${predictionsDir}" "${groundtruthTsv}" \
      -ia "${t0Dir}/release/IA.tsv" \
      -out_dir results_NK/ \
      -toi "${toiFile}" \
      -prop fill -norm cafa -threads ${task.cpus} -no_orphans
    """

    stub:
    """
    mkdir -p results_NK
    printf 'filename\tns\ttau\tn\tpr_micro_w\trc_micro_w\tf_micro_w\tcov_w\n' > results_NK/evaluation_best_f_micro_w.tsv
    printf 'filename\tns\ttau\tcov\trc_micro_w\tpr_micro_w\tf_micro_w\n' > results_NK/evaluation_all.tsv
    """
}

process EVALUATE_LK {
    label 'cpu_medium'
    tag "${meta.release_id}:LK"

    input:
    tuple val(meta), path(t0Dir), path(predictionsDir), path(groundtruthTsv), path(toiFile)

    output:
    tuple val(meta), path('results_LK')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate cafa5-evaluator

    cafaeval "${t0Dir}/release/go-basic.obo" "${predictionsDir}" "${groundtruthTsv}" \
      -ia "${t0Dir}/release/IA.tsv" \
      -out_dir results_LK/ \
      -toi "${toiFile}" \
      -prop fill -norm cafa -threads ${task.cpus} -no_orphans
    """

    stub:
    """
    mkdir -p results_LK
    printf 'filename\tns\ttau\tn\tpr_micro_w\trc_micro_w\tf_micro_w\tcov_w\n' > results_LK/evaluation_best_f_micro_w.tsv
    printf 'filename\tns\ttau\tcov\trc_micro_w\tpr_micro_w\tf_micro_w\n' > results_LK/evaluation_all.tsv
    """
}

process EVALUATE_PK {
    label 'cpu_large'
    tag "${meta.release_id}:PK"

    input:
    tuple val(meta), path(t0Dir), path(predictionsDir), path(groundtruthTsv), path(pkKnownTsv), path(toiFile)

    output:
    tuple val(meta), path('results_PK')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate cafa5-evaluator

    cafaeval "${t0Dir}/release/go-basic.obo" "${predictionsDir}" "${groundtruthTsv}" \
      -ia "${t0Dir}/release/IA.tsv" \
      -out_dir results_PK/ \
      -toi "${toiFile}" \
      -known "${pkKnownTsv}" \
      -prop fill -norm cafa -threads ${task.cpus} -no_orphans
    """

    stub:
    """
    mkdir -p results_PK
    printf 'filename\tns\ttau\tn\tpr_micro_w\trc_micro_w\tf_micro_w\tcov_w\n' > results_PK/evaluation_best_f_micro_w.tsv
    printf 'filename\tns\ttau\tcov\trc_micro_w\tpr_micro_w\tf_micro_w\n' > results_PK/evaluation_all.tsv
    """
}

process EVALUATE_LATE_NK {
    label 'cpu_medium'
    tag "${meta.release_id}:NK:late"

    input:
    tuple val(meta), path(t0Dir), path(releaseDir), val(methodNamesFile)

    output:
    tuple val(meta), path('results_uneval_NK')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate cafa5-evaluator

    cafaeval "${t0Dir}/release/go-basic.obo" "${t0Dir}/predictions_uneval" "${releaseDir}/groundtruth_NK.tsv" \
      -ia "${t0Dir}/release/IA.tsv" \
      -out_dir results_uneval_NK/ \
      -toi "${releaseDir}/groundtruth_terms_of_interest.txt" \
      -prop fill -norm cafa -threads ${task.cpus} -no_orphans
    """

    stub:
    """
    mkdir -p results_uneval_NK
    printf 'filename\tns\ttau\tn\tpr_micro_w\trc_micro_w\tf_micro_w\tcov_w\n' > results_uneval_NK/evaluation_best_f_micro_w.tsv
    printf 'filename\tns\ttau\tcov\trc_micro_w\tpr_micro_w\tf_micro_w\n' > results_uneval_NK/evaluation_all.tsv
    """
}

process EVALUATE_LATE_LK {
    label 'cpu_medium'
    tag "${meta.release_id}:LK:late"

    input:
    tuple val(meta), path(t0Dir), path(releaseDir), val(methodNamesFile)

    output:
    tuple val(meta), path('results_uneval_LK')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate cafa5-evaluator

    cafaeval "${t0Dir}/release/go-basic.obo" "${t0Dir}/predictions_uneval" "${releaseDir}/groundtruth_LK.tsv" \
      -ia "${t0Dir}/release/IA.tsv" \
      -out_dir results_uneval_LK/ \
      -toi "${releaseDir}/groundtruth_terms_of_interest.txt" \
      -prop fill -norm cafa -threads ${task.cpus} -no_orphans
    """

    stub:
    """
    mkdir -p results_uneval_LK
    printf 'filename\tns\ttau\tn\tpr_micro_w\trc_micro_w\tf_micro_w\tcov_w\n' > results_uneval_LK/evaluation_best_f_micro_w.tsv
    printf 'filename\tns\ttau\tcov\trc_micro_w\tpr_micro_w\tf_micro_w\n' > results_uneval_LK/evaluation_all.tsv
    """
}

process EVALUATE_LATE_PK {
    label 'cpu_large'
    tag "${meta.release_id}:PK:late"

    input:
    tuple val(meta), path(t0Dir), path(releaseDir), val(methodNamesFile)

    output:
    tuple val(meta), path('results_uneval_PK')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate cafa5-evaluator

    cafaeval "${t0Dir}/release/go-basic.obo" "${t0Dir}/predictions_uneval" "${releaseDir}/groundtruth_PK.tsv" \
      -ia "${t0Dir}/release/IA.tsv" \
      -out_dir results_uneval_PK/ \
      -toi "${releaseDir}/groundtruth_terms_of_interest.txt" \
      -known "${releaseDir}/groundtruth_PK_known.tsv" \
      -prop fill -norm cafa -threads ${task.cpus} -no_orphans
    """

    stub:
    """
    mkdir -p results_uneval_PK
    printf 'filename\tns\ttau\tn\tpr_micro_w\trc_micro_w\tf_micro_w\tcov_w\n' > results_uneval_PK/evaluation_best_f_micro_w.tsv
    printf 'filename\tns\ttau\tcov\trc_micro_w\tpr_micro_w\tf_micro_w\n' > results_uneval_PK/evaluation_all.tsv
    """
}

process MERGE_LATE_RESULTS {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(nkDir), path(lkDir), path(pkDir)

    output:
    tuple val(meta), path("${meta.release_id}")

    script:
    """
    mkdir -p "${meta.release_id}/results_NK" "${meta.release_id}/results_LK" "${meta.release_id}/results_PK"

    for subset in NK LK PK; do
      src_dir="results_uneval_\${subset}"
      dest_dir="${meta.release_id}/results_\${subset}"
      for file in "\${src_dir}"/*; do
        filename=\$(basename "\${file}")
        sed '1d' "\${file}" > "\${dest_dir}/\${filename}"
      done
    done
    """

    stub:
    """
    mkdir -p "${meta.release_id}/results_NK" "${meta.release_id}/results_LK" "${meta.release_id}/results_PK"
    for subset in NK LK PK; do
      src_dir="results_uneval_\${subset}"
      dest_dir="${meta.release_id}/results_\${subset}"
      for file in "\${src_dir}"/*; do
        filename=\$(basename "\${file}")
        sed '1d' "\${file}" > "\${dest_dir}/\${filename}"
      done
    done
    """
}

process FINALIZE_LATE_PREDICTIONS {
    label 'packaging'
    tag "${meta.t0_id}"
    publishDir "${params.output_root}", mode: 'copy'

    input:
    tuple val(meta), path(t0Dir), path(releaseDir), val(methodNamesFile)

    output:
    tuple val(meta), path("${meta.t0_id}")

    script:
    """
    cp -r "${t0Dir}" "${meta.t0_id}"
    mkdir -p "${meta.t0_id}/predictions"
    if compgen -G "${meta.t0_id}/predictions_uneval/*" > /dev/null; then
      mv "${meta.t0_id}"/predictions_uneval/* "${meta.t0_id}/predictions/" || true
    fi
    """

    stub:
    """
    cp -r "${t0Dir}" "${meta.t0_id}"
    mkdir -p "${meta.t0_id}/predictions"
    if compgen -G "${meta.t0_id}/predictions_uneval/*" > /dev/null; then
      mv "${meta.t0_id}"/predictions_uneval/* "${meta.t0_id}/predictions/" || true
    fi
    """
}
