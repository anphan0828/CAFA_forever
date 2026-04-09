process PREDICT_NAIVE {
    label 'cpu_medium'
    tag "${meta.release_id ?: meta.timepoint_id}"

    input:
    tuple val(meta), path(releaseDir), val(queryFastaName)

    output:
    tuple val(meta), path('naive_predictions.tsv')

    script:
    """
    module load singularity || true
    export SINGULARITY_CACHEDIR=${params.workspace_root}/.singularity_cache
    export SINGULARITY_TMPDIR=${params.workspace_root}/.singularity_tmp
    mkdir -p "\$SINGULARITY_CACHEDIR" "\$SINGULARITY_TMPDIR" output

    singularity exec --pwd /app \
      --bind "${releaseDir}:/app/data" \
      --bind "\$PWD/output:/app/output" \
      "${params.workspace_root}/test_naive_latest.sif" \
      python3 naive.py \
      --annot_file /app/data/train_terms.tsv \
      --query_file "/app/data/${queryFastaName}" \
      --graph /app/data/go-basic.obo \
      --output_baseline /app/output/naive_predictions.tsv \
      --n_terms 1500

    cp output/naive_predictions.tsv .
    """

    stub:
    """
    touch naive_predictions.tsv
    """
}

process PREDICT_GOA_NONEXP {
    label 'cpu_medium'
    tag "${meta.release_id ?: meta.timepoint_id}"

    input:
    tuple val(meta), path(releaseDir), val(queryFastaName)

    output:
    tuple val(meta), path('goa_nonexp_predictions.tsv')

    script:
    """
    module load singularity || true
    export SINGULARITY_CACHEDIR=${params.workspace_root}/.singularity_cache
    export SINGULARITY_TMPDIR=${params.workspace_root}/.singularity_tmp
    export APPTAINER_TMPDIR="\$SINGULARITY_TMPDIR"
    mkdir -p "\$SINGULARITY_CACHEDIR" "\$SINGULARITY_TMPDIR" output

    singularity exec --pwd /app \
      --bind "${releaseDir}:/app/data" \
      --bind "\$PWD/output:/app/output" \
      "${params.workspace_root}/test_goa_nonexp_latest.sif" \
      python3 goa_nonexp.py \
      --annot_file /app/data/goa_uniprot_sprot.gaf.gz \
      --selected_go 'Computational,Phylogenetical,Electronic,ND,NAS' \
      --query_file "/app/data/${queryFastaName}" \
      --graph /app/data/go-basic.obo \
      --output_baseline /app/output/goa_nonexp_predictions.tsv

    cp output/goa_nonexp_predictions.tsv .
    """

    stub:
    """
    touch goa_nonexp_predictions.tsv
    """
}

process PREDICT_BLAST {
    label 'gpu_a100'
    tag "${meta.release_id ?: meta.timepoint_id}"

    input:
    tuple val(meta), path(releaseDir), val(queryFastaName)

    output:
    tuple val(meta), path('blast_predictions.tsv')

    script:
    """
    module load singularity || true
    export SINGULARITY_CACHEDIR=${params.workspace_root}/.singularity_cache
    export SINGULARITY_TMPDIR=${params.workspace_root}/.singularity_tmp
    mkdir -p "\$SINGULARITY_CACHEDIR" "\$SINGULARITY_TMPDIR" output

    nvidia-cuda-mps-server || true

    singularity exec --nv --pwd /app \
      --bind "${releaseDir}:/app/data" \
      --bind "\$PWD/output:/app/output" \
      "${params.workspace_root}/test_blast_latest_gpu.sif" \
      python3 blast_main.py \
      --annot_file /app/data/train_terms_propagated.tsv \
      --query_file "/app/data/${queryFastaName}" \
      --graph /app/data/go-basic.obo \
      --blast_results /app/data/blast_results.tsv \
      --train_sequences /app/data/train_sequences.fasta \
      --train_taxonomy /app/data/train_taxonomy.tsv \
      --output_baseline /app/output/blast_predictions.tsv

    cp output/blast_predictions.tsv .
    """

    stub:
    """
    touch blast_predictions.tsv
    """
}

process PREDICT_PROTT5 {
    label 'gpu_a100'
    tag "${meta.release_id ?: meta.timepoint_id}"

    input:
    tuple val(meta), path(releaseDir), val(queryFastaName)

    output:
    tuple val(meta), path('prott5_predictions.tsv')

    script:
    """
    module load singularity || true
    export SINGULARITY_CACHEDIR=\${TMPDIR:-${params.workspace_root}/.singularity_tmp}
    export SINGULARITY_TMPDIR=\${TMPDIR:-${params.workspace_root}/.singularity_tmp}
    mkdir -p output

    nvidia-cuda-mps-server || true

    singularity exec --nv --pwd /app \
      --bind "${releaseDir}:/app/data" \
      --bind "\$PWD/output:/app/output" \
      "${params.workspace_root}/test_prott5_latest_gpu.sif" \
      python3 prott5_main.py \
      --annot_file /app/data/train_terms.tsv \
      --query_file "/app/data/${queryFastaName}" \
      --graph /app/data/go-basic.obo \
      --train_sequences /app/data/train_sequences.fasta \
      --output_baseline /app/output/prott5_predictions.tsv \
      --n_terms 1500

    cp output/prott5_predictions.tsv .
    """

    stub:
    """
    touch prott5_predictions.tsv
    """
}

process PREDICT_TRANSFEW {
    label 'gpu_a100'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(releaseDir), val(queryFastaName)

    output:
    tuple val(meta), path('transfew_predictions.tsv')

    script:
    """
    module load singularity || true
    export SINGULARITY_CACHEDIR=\${TMPDIR:-${params.workspace_root}/.singularity_tmp}
    export SINGULARITY_TMPDIR=\${TMPDIR:-${params.workspace_root}/.singularity_tmp}
    mkdir -p output

    for ontology in mf bp cc; do
      XDG_CACHE_HOME=/root/.cache/ singularity run --nv \
        --pwd /workspace \
        --bind "${releaseDir}:/root/data" \
        --bind "\$PWD/output:/root/output" \
        --bind "${params.workspace_root}/checkpoints_TransFew:/root/.cache/torch/hub/checkpoints" \
        "${params.workspace_root}/transfew_predictor_latest.sif" \
        --fasta-path "/root/data/${queryFastaName}" \
        --working-dir /root/output \
        --ontology "\${ontology}" \
        --output "transfew_predictions_\${ontology}.tsv.gz"

      gunzip "output/transfew_predictions_\${ontology}.tsv.gz"
      awk -F'\\t' 'BEGIN{OFS="\\t"} {split(\$1,a,"|"); print a[2], \$2, \$3}' \
        "output/transfew_predictions_\${ontology}.tsv" > "output/transfew_predictions2_\${ontology}.tsv"
    done

    cat output/transfew_predictions2_mf.tsv output/transfew_predictions2_bp.tsv output/transfew_predictions2_cc.tsv > transfew_predictions.tsv
    """

    stub:
    """
    touch transfew_predictions.tsv
    """
}

process PREDICT_FUNBIND {
    label 'gpu_scavenger'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(releaseDir), val(splitDirName)

    output:
    tuple val(meta), path('funbind_predictions.tsv')

    script:
    """
    module load apptainer || true
    export SINGULARITY_CACHEDIR=\${TMPDIR:-${params.workspace_root}/.singularity_tmp}
    export SINGULARITY_TMPDIR=\${TMPDIR:-${params.workspace_root}/.singularity_tmp}
    mkdir -p output

    nvidia-cuda-mps-server || true

    for fasta in "${releaseDir}/${splitDirName}"/part_*.fasta; do
      part_name=\$(basename "\${fasta}" .fasta)
      fasta_name=\$(basename "\${fasta}")
      apptainer run --nv --pwd /software/FunBind --writable-tmpfs \
        --bind "${releaseDir}:/root/data" \
        --bind "\$PWD/output:/root/output" \
        --bind "${params.workspace_root}/cache/pretrained_huggingface_models/cache_folder:/root/.cache/huggingface/hub" \
        --bind "${params.workspace_root}/checkpoints_FunBind:/root/.cache/checkpoints" \
        "${params.workspace_root}/test_funbind_latest.sif" \
        --data-path /root/.cache/checkpoints \
        --sequence-path "/root/data/${splitDirName}/\${fasta_name}" \
        --output "/root/output/\${part_name}"
    done

    find output -type f \\( -path "*/BP/Sequence_BP.tsv" -o -path "*/MF/Sequence_MF.tsv" -o -path "*/CC/Sequence_CC.tsv" \\) \
      | sort | xargs cat > funbind_predictions.tsv
    """

    stub:
    """
    touch funbind_predictions.tsv
    """
}

process PREDICT_DEEPGOPLUS {
    label 'gpu_a100'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(releaseDir), val(queryFastaName)

    output:
    tuple val(meta), path('deepgoplus_predictions.tsv')

    script:
    """
    module load apptainer || true
    export SINGULARITY_CACHEDIR=\${TMPDIR:-${params.workspace_root}/.singularity_tmp}
    export SINGULARITY_TMPDIR=\${TMPDIR:-${params.workspace_root}/.singularity_tmp}

    apptainer run --nv --pwd /deepgoplus --writable-tmpfs \
      --bind "\$PWD:/output" \
      --bind "${releaseDir}:/ext_data" \
      "${params.workspace_root}/deepgoplus_latest.sif" \
      -if /ext_data/${queryFastaName} \
      -of /output/deepgoplus_predictions.tsv

    awk -F'\\t' 'BEGIN{OFS="\\t"} {split(\$1,a,"|"); print a[2], \$2, \$3}' \
      deepgoplus_predictions.tsv > deepgoplus_predictions2.tsv
    mv deepgoplus_predictions2.tsv deepgoplus_predictions.tsv
    """

    stub:
    """
    touch deepgoplus_predictions.tsv
    """
}

process ASSEMBLE_TIMEPOINT_PREDICTIONS {
    label 'packaging'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(naivePred), path(goaPred), path(blastPred), path(prott5Pred)

    output:
    tuple val(meta), path('predictions')

    script:
    """
    mkdir -p predictions
    cp "${naivePred}" predictions/
    cp "${goaPred}" predictions/
    cp "${blastPred}" predictions/
    cp "${prott5Pred}" predictions/
    """

    stub:
    """
    mkdir -p predictions
    touch predictions/naive_predictions.tsv
    touch predictions/goa_nonexp_predictions.tsv
    touch predictions/blast_predictions.tsv
    touch predictions/prott5_predictions.tsv
    """
}

process INIT_EMPTY_PREDICTIONS_DIR {
    label 'packaging'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(releaseDir)

    output:
    tuple val(meta), path('predictions')

    script:
    """
    mkdir -p predictions
    """

    stub:
    """
    mkdir -p predictions
    """
}

process ASSEMBLE_WINDOW_PREDICTIONS {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(predictionFiles)

    output:
    tuple val(meta), path('predictions_window')

    script:
    """
    mkdir -p predictions_window
    cp ${predictionFiles} predictions_window/
    """

    stub:
    """
    mkdir -p predictions_window
    for file in ${predictionFiles}; do
      touch "predictions_window/\$(basename "\$file")"
    done
    """
}

process PACKAGE_WINDOW_PREDICTIONS {
    label 'packaging'
    tag "${meta.release_id}"

    input:
    tuple val(meta), path(t0Dir), path(predictionsDir)

    output:
    tuple val(meta), path('published_window_predictions.txt')

    script:
    """
    target_dir="${t0Dir}/${params.window_prediction_subdir}/${meta.release_id}"
    mkdir -p "\${target_dir}"
    cp -r "${predictionsDir}/." "\${target_dir}/"
    printf '%s\\n' "\${target_dir}" > published_window_predictions.txt
    """

    stub:
    """
    target_dir="${t0Dir}/${params.window_prediction_subdir}/${meta.release_id}"
    mkdir -p "\${target_dir}"
    cp -r "${predictionsDir}/." "\${target_dir}/"
    printf '%s\\n' "\${target_dir}" > published_window_predictions.txt
    """
}

process PACKAGE_TIMEPOINT_DIRECTORY {
    label 'packaging'
    tag "${meta.timepoint_id}"
    publishDir "${params.output_root}", mode: 'copy'

    input:
    tuple val(meta), path(goBasic), path(filteredGaf), path(trainTerms), path(trainSequences), path(trainTaxonomy), path(testSequences), path(trainTermsPropagated), path(iaTsv), path(blastResults), path(splitDir), path(releaseDir), path(predictionsDir)

    output:
    tuple val(meta), path("${meta.timepoint_id}")

    script:
    """
    mkdir -p "${meta.timepoint_id}"
    cp "${goBasic}" "${meta.timepoint_id}/go-basic.obo"
    cp "${filteredGaf}" "${meta.timepoint_id}/goa_uniprot_sprot.gaf.gz"
    cp "${trainTerms}" "${meta.timepoint_id}/train_terms.tsv"
    cp "${trainSequences}" "${meta.timepoint_id}/train_sequences.fasta"
    cp "${trainTaxonomy}" "${meta.timepoint_id}/train_taxonomy.tsv"
    cp "${testSequences}" "${meta.timepoint_id}/test_sequences.fasta"
    cp "${trainTermsPropagated}" "${meta.timepoint_id}/train_terms_propagated.tsv"
    cp "${iaTsv}" "${meta.timepoint_id}/IA.tsv"
    cp "${blastResults}" "${meta.timepoint_id}/blast_results.tsv"
    cp -r "${splitDir}" "${meta.timepoint_id}/test_sequences_split"
    cp -r "${releaseDir}" "${meta.timepoint_id}/release"
    cp -r "${predictionsDir}" "${meta.timepoint_id}/predictions"
    """

    stub:
    """
    mkdir -p "${meta.timepoint_id}"
    touch "${meta.timepoint_id}/go-basic.obo"
    touch "${meta.timepoint_id}/goa_uniprot_sprot.gaf.gz"
    touch "${meta.timepoint_id}/train_terms.tsv"
    touch "${meta.timepoint_id}/train_sequences.fasta"
    touch "${meta.timepoint_id}/train_taxonomy.tsv"
    touch "${meta.timepoint_id}/test_sequences.fasta"
    touch "${meta.timepoint_id}/train_terms_propagated.tsv"
    touch "${meta.timepoint_id}/IA.tsv"
    touch "${meta.timepoint_id}/blast_results.tsv"
    cp -r "${splitDir}" "${meta.timepoint_id}/test_sequences_split"
    cp -r "${releaseDir}" "${meta.timepoint_id}/release"
    cp -r "${predictionsDir}" "${meta.timepoint_id}/predictions"
    """
}
