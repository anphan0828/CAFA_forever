process VALIDATE_TIMEPOINT_INPUTS {
    label 'packaging'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(goBasic), path(goaAll), path(swissProt)

    output:
    tuple val(meta), path(goBasic), path(goaAll), path(swissProt)

    script:
    """
    test -f "${goBasic}"
    test -f "${goaAll}"
    test -f "${swissProt}"
    """

    stub:
    """
    test -f "${goBasic}"
    test -f "${goaAll}"
    test -f "${swissProt}"
    """
}

process FILTER_GAF_TO_SWISSPROT {
    label 'cpu_large'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(goBasic), path(goaAll), path(swissProt)

    output:
    tuple val(meta), path(goBasic), path('goa_uniprot_sprot.gaf.gz'), path(swissProt)

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate "${params.democafa_env}"
    pip install "${params.democafa_package}"

    python3 -m democafa.datacollection.filter_gaf \
      -a "${goaAll}" \
      -q "${swissProt}" \
      -o goa_uniprot_sprot.gaf.gz
    """

    stub:
    """
    touch goa_uniprot_sprot.gaf.gz
    """
}

process RETRIEVE_EXPERIMENTAL_TERMS {
    label 'cpu_medium'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(goBasic), path(filteredGaf), path(swissProt)

    output:
    tuple val(meta), path(goBasic), path(filteredGaf), path(swissProt), path('train_terms.tsv')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate "${params.democafa_env}"
    pip install "${params.democafa_package}"

    python3 -m democafa.datacollection.retrieve_terms \
      --annot "${filteredGaf}" \
      -sgc 'Experimental,IC,TAS' \
      -g "${goBasic}" \
      --tsv train_terms.tsv
    """

    stub:
    """
    touch train_terms.tsv
    """
}

process CREATE_TRAINING_SET {
    label 'cpu_medium'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(trainTerms), path(swissProt)

    output:
    tuple val(meta), path(trainTerms), path('train_sequences.fasta'), path('train_taxonomy.tsv')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate "${params.democafa_env}"
    pip install "${params.democafa_package}"

    python3 -m democafa.datacollection.create_test_set \
      --terms "${trainTerms}" \
      --fasta_gz "${swissProt}" \
      --train_out_fasta train_sequences.fasta \
      --train_out_taxonomy train_taxonomy.tsv \
      --include_all
    """

    stub:
    """
    touch train_sequences.fasta
    touch train_taxonomy.tsv
    """
}

process CREATE_TEST_FASTA {
    label 'cpu_small'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(swissProt)

    output:
    tuple val(meta), path('test_sequences.fasta')

    script:
    """
    gunzip -c "${swissProt}" > test_sequences.fasta
    """

    stub:
    """
    touch test_sequences.fasta
    """
}

process PROPAGATE_AND_COMPUTE_IA {
    label 'cpu_medium'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(goBasic), path(trainTerms)

    output:
    tuple val(meta), path('train_terms_propagated.tsv'), path('IA.tsv')

    script:
    """
    module load micromamba || true
    export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "\$(micromamba shell hook --shell=bash)"
    micromamba activate "${params.democafa_env}"
    pip install "${params.democafa_package}"

    python3 -m democafa.datacollection.propagate_and_ia \
      --terms "${trainTerms}" \
      --graph "${goBasic}" \
      --tsv_propagated train_terms_propagated.tsv \
      --output_tsv IA.tsv
    """

    stub:
    """
    touch train_terms_propagated.tsv
    touch IA.tsv
    """
}

process SPLIT_TEST_FASTA {
    label 'cpu_small'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(testSequences)

    output:
    tuple val(meta), path('test_sequences_split')

    script:
    """
    mkdir -p test_sequences_split
    python3 - "${testSequences}" "${params.split_size}" <<'PY'
import pathlib
import sys

fasta_path = pathlib.Path(sys.argv[1])
chunk_size = int(sys.argv[2])
out_dir = pathlib.Path("test_sequences_split")

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
    mkdir -p test_sequences_split
    touch test_sequences_split/part_001.fasta
    """
}

process FREEZE_TIMEPOINT_RELEASE {
    label 'packaging'
    tag "${meta.timepoint_id}"

    input:
    tuple val(meta), path(goBasic), path(filteredGaf), path(trainTerms), path(trainSequences), path(trainTaxonomy), path(testSequences), path(trainTermsPropagated), path(iaTsv), path(splitDir)

    output:
    tuple val(meta), path(goBasic), path(filteredGaf), path(trainTerms), path(trainSequences), path(trainTaxonomy), path(testSequences), path(trainTermsPropagated), path(iaTsv), path(splitDir), path('release')

    script:
    """
    mkdir -p release
    cp "${goBasic}" release/go-basic.obo
    cp "${filteredGaf}" release/goa_uniprot_sprot.gaf.gz
    cp "${trainTerms}" release/train_terms.tsv
    cp "${trainSequences}" release/train_sequences.fasta
    cp "${trainTermsPropagated}" release/train_terms_propagated.tsv
    cp "${trainTaxonomy}" release/train_taxonomy.tsv
    cp "${testSequences}" release/test_sequences.fasta
    cp "${iaTsv}" release/IA.tsv
    """

    stub:
    """
    mkdir -p release
    touch release/go-basic.obo
    touch release/goa_uniprot_sprot.gaf.gz
    touch release/train_terms.tsv
    touch release/train_sequences.fasta
    touch release/train_terms_propagated.tsv
    touch release/train_taxonomy.tsv
    touch release/test_sequences.fasta
    touch release/IA.tsv
    """
}
