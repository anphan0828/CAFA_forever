include {
    VALIDATE_TIMEPOINT_INPUTS
    FILTER_GAF_TO_SWISSPROT
    RETRIEVE_EXPERIMENTAL_TERMS
    CREATE_TRAINING_SET
    CREATE_TEST_FASTA
    PROPAGATE_AND_COMPUTE_IA
    SPLIT_TEST_FASTA
    FREEZE_TIMEPOINT_RELEASE
} from '../modules/local/timepoint_build'

include {
    PREDICT_NAIVE
    PREDICT_GOA_NONEXP
    PREDICT_BLAST
    PREDICT_PROTT5
    INIT_EMPTY_PREDICTIONS_DIR
    ASSEMBLE_TIMEPOINT_PREDICTIONS
    PACKAGE_TIMEPOINT_DIRECTORY
} from '../modules/local/predictions'

workflow TIMEPOINT_BUILD {
    take:
    raw_inputs_ch

    main:
    validated_ch = VALIDATE_TIMEPOINT_INPUTS(raw_inputs_ch)
    filtered_ch  = FILTER_GAF_TO_SWISSPROT(validated_ch)
    terms_ch     = RETRIEVE_EXPERIMENTAL_TERMS(filtered_ch)
    train_ch     = CREATE_TRAINING_SET(terms_ch.map { meta, goBasic, filteredGaf, swissProt, trainTerms -> tuple(meta, trainTerms, swissProt) })
    test_ch      = CREATE_TEST_FASTA(validated_ch.map { meta, goBasic, goaAll, swissProt -> tuple(meta, swissProt) })
    ia_ch        = PROPAGATE_AND_COMPUTE_IA(terms_ch.map { meta, goBasic, filteredGaf, swissProt, trainTerms -> tuple(meta, goBasic, trainTerms) })
    split_ch     = SPLIT_TEST_FASTA(test_ch)

    release_inputs_ch = filtered_ch
        .map { meta, goBasic, filteredGaf, swissProt -> tuple(meta, goBasic, filteredGaf) }
        .join(terms_ch.map { meta, goBasic, filteredGaf, swissProt, trainTerms -> tuple(meta, trainTerms) })
        .join(train_ch.map { meta, trainTerms, trainSequences, trainTaxonomy -> tuple(meta, trainSequences, trainTaxonomy) })
        .join(test_ch.map { meta, testSequences -> tuple(meta, testSequences) })
        .join(ia_ch.map { meta, trainTermsPropagated, iaTsv -> tuple(meta, trainTermsPropagated, iaTsv) })
        .join(split_ch.map { meta, splitDir -> tuple(meta, splitDir) })
        .map { meta, goBasic, filteredGaf, trainTerms, trainSequences, trainTaxonomy, testSequences, trainTermsPropagated, iaTsv, splitDir ->
            tuple(meta, goBasic, filteredGaf, trainTerms, trainSequences, trainTaxonomy, testSequences, trainTermsPropagated, iaTsv, splitDir)
        }

    release_ch   = FREEZE_TIMEPOINT_RELEASE(release_inputs_ch)

    if( params.prediction_target_mode == 'window_groundtruth' ) {
        /*
         * In window-groundtruth mode, timepoint_build only materializes the canonical
         * release bundle. Predictions are deferred to EVALUATE_WINDOW because the
         * actual query FASTA is the window-specific groundtruth_targets.fasta, which
         * does not exist until the T0/T1 window has been constructed. To produce  
         * predictions in this mode, use the EVALUATE_WINDOW workflow
         */
        predictions_dir_ch = INIT_EMPTY_PREDICTIONS_DIR(release_ch.map { meta, go, gaf, terms, trainSeq, trainTax, testSeq, propTerms, ia, splitDir, releaseDir -> tuple(meta, releaseDir) })
    }
    else {
        /*
         * In the default 'test_sequences' mode, timepoint_build produces predictions for the test_sequences.fasta
         * generated as part of the release bundle. This allows us to have predictions ready immediately after
         * the release is frozen, independent of window construction and EVALUATE_WINDOW runs.
        */
        naive_ch     = PREDICT_NAIVE(release_ch.map { meta, go, gaf, terms, trainSeq, trainTax, testSeq, propTerms, ia, splitDir, releaseDir -> tuple(meta, releaseDir, 'test_sequences.fasta') })
        goa_ch       = PREDICT_GOA_NONEXP(release_ch.map { meta, go, gaf, terms, trainSeq, trainTax, testSeq, propTerms, ia, splitDir, releaseDir -> tuple(meta, releaseDir, 'test_sequences.fasta') })
        blastpred_ch = PREDICT_BLAST(release_ch.map { meta, go, gaf, terms, trainSeq, trainTax, testSeq, propTerms, ia, splitDir, releaseDir -> tuple(meta, releaseDir, 'test_sequences.fasta') })
        prott5_ch    = PREDICT_PROTT5(release_ch.map { meta, go, gaf, terms, trainSeq, trainTax, testSeq, propTerms, ia, splitDir, releaseDir -> tuple(meta, releaseDir, 'test_sequences.fasta') })

        prediction_bundle_ch = naive_ch
            .join(goa_ch)
            .join(blastpred_ch)
            .join(prott5_ch)

        predictions_dir_ch = ASSEMBLE_TIMEPOINT_PREDICTIONS(prediction_bundle_ch)
    }

    timepoint_bundle_ch = release_ch
        .join(predictions_dir_ch)

    packaged_timepoint_ch = PACKAGE_TIMEPOINT_DIRECTORY(timepoint_bundle_ch)

    emit:
    packaged_timepoint_ch
    release_ch
    predictions_dir_ch
}
