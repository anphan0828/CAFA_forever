include {
    INTERSECT_TARGETS
    CLASSIFY_GROUNDTRUTH
    EXTRACT_GROUNDTRUTH_TARGETS
    PREPARE_WINDOW_RELEASE
    SPLIT_WINDOW_TARGETS
    EVALUATE_NK
    EVALUATE_LK
    EVALUATE_PK
} from '../modules/local/evaluation'

include {
    PREDICT_NAIVE
    PREDICT_GOA_NONEXP
    PREDICT_BLAST
    PREDICT_PROTT5
    PREDICT_TRANSFEW
    PREDICT_FUNBIND
    PREDICT_DEEPGOPLUS
    ASSEMBLE_WINDOW_PREDICTIONS
    PACKAGE_WINDOW_PREDICTIONS
} from '../modules/local/predictions'

include {
    GENERATE_METHOD_NAMES
    GENERATE_METHOD_AVAILABILITY
    PACKAGE_RELEASE_DIRECTORY
    VALIDATE_RELEASE_DIRECTORY
    PUBLISH_RELEASE_DIRECTORY
    BUILD_RELEASE_CATALOG
} from '../modules/local/release_contract'

workflow EVALUATE_WINDOW {
    take:
    window_inputs_ch

    main:
    def knownWindowMethods = ['naive', 'goa_nonexp', 'blast', 'prott5', 'transfew', 'funbind', 'deepgoplus'] as Set
    def normalizedWindowMethodAliases = [
        naive      : 'naive',
        goanonexp  : 'goa_nonexp',
        blast      : 'blast',
        prott5     : 'prott5',
        transfew   : 'transfew',
        funbind    : 'funbind',
        deepgoplus : 'deepgoplus',
    ]
    def predictionFileByMethod = [
        naive      : 'naive_predictions.tsv',
        goa_nonexp : 'goa_nonexp_predictions.tsv',
        blast      : 'blast_predictions.tsv',
        prott5     : 'prott5_predictions.tsv',
        transfew   : 'transfew_predictions.tsv',
        funbind    : 'funbind_predictions.tsv',
        deepgoplus : 'deepgoplus_predictions.tsv',
    ]
    def normalizeWindowMethod = { methodName ->
        def compactName = methodName
            .toString()
            .trim()
            .toLowerCase()
            .replaceAll(/[^a-z0-9]+/, '')
        normalizedWindowMethodAliases[compactName]
    }
    def rawEnabledWindowParam = params.enabled_window_methods ?: params.enabledWindowMethods ?: (knownWindowMethods as List)
    def enabledWindowMethods

    if( rawEnabledWindowParam instanceof CharSequence ) {
        enabledWindowMethods = rawEnabledWindowParam
            .toString()
            .split(',')
            .collect { it.trim() }
            .findAll { it }
    }
    else {
        enabledWindowMethods = (rawEnabledWindowParam as List).collect { it.toString() }
    }
    enabledWindowMethods = enabledWindowMethods
        .collect { normalizeWindowMethod(it) ?: it.toString().trim() }
        .findAll { it }
        .unique()

    def unknownWindowMethods = enabledWindowMethods.findAll { !knownWindowMethods.contains(it) }
    def rawPredictionMode = (params.window_prediction_mode ?: params.windowPredictionMode ?: 'generate').toString()
    def predictionModeAliases = [
        generate    : 'generate',
        build       : 'generate',
        existing    : 'existing',
        existingdir : 'existing',
        reuse       : 'existing',
        skip        : 'existing',
    ]
    def predictionMode = predictionModeAliases[
        rawPredictionMode.trim().toLowerCase().replaceAll(/[^a-z0-9]+/, '')
    ]
    def windowPredictionsDirParam = params.window_predictions_dir ?: params.windowPredictionsDir

    if( unknownWindowMethods ) {
        error "Unknown window methods in --enabled_window_methods/--enabledWindowMethods: ${unknownWindowMethods.join(', ')}"
    }
    if( !enabledWindowMethods ) {
        error "--enabled_window_methods/--enabledWindowMethods must contain at least one method"
    }
    if( !predictionMode ) {
        error "Unknown window prediction mode '${rawPredictionMode}'. Expected generate or existing"
    }
    if( predictionMode == 'existing' && !windowPredictionsDirParam ) {
        error "Missing required parameter for existing window predictions: --window_predictions_dir or --windowPredictionsDir"
    }
    def expectedPredictionFiles = enabledWindowMethods.collect { predictionFileByMethod[it] }

    common_targets_ch = INTERSECT_TARGETS(window_inputs_ch)
    groundtruth_ch    = CLASSIFY_GROUNDTRUTH(common_targets_ch)
    target_fasta_ch   = EXTRACT_GROUNDTRUTH_TARGETS(groundtruth_ch)

    def assembled_predictions_ch

    if( predictionMode == 'existing' ) {
        existing_prediction_files_ch = window_inputs_ch.map { meta, t0Dir, t1Dir, methodNamesFile ->
            def predictionsDir = file(windowPredictionsDirParam)
            if( !predictionsDir.exists() ) {
                error "Existing window predictions directory does not exist: ${predictionsDir}"
            }

            def predictionFiles = expectedPredictionFiles.collect { predictionFile ->
                file("${predictionsDir}/${predictionFile}")
            }
            def missingFiles = predictionFiles.findAll { !it.exists() }.collect { it.getFileName().toString() }
            if( missingFiles ) {
                error "Missing prediction files in existing window predictions directory for ${meta.release_id}: ${missingFiles.join(', ')}"
            }

            tuple(meta, predictionFiles)
        }
        assembled_predictions_ch = ASSEMBLE_WINDOW_PREDICTIONS(existing_prediction_files_ch)
    }
    else {
        window_release_ch = PREPARE_WINDOW_RELEASE(target_fasta_ch)
        prediction_release_ch = SPLIT_WINDOW_TARGETS(window_release_ch)

        def predictionChannels = []

        if( enabledWindowMethods.contains('naive') ) {
            predictionChannels << PREDICT_NAIVE(
                prediction_release_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, windowReleaseReady ->
                    tuple(meta, windowReleaseReady, 'groundtruth_targets.fasta')
                }
            )
        }
        if( enabledWindowMethods.contains('goa_nonexp') ) {
            predictionChannels << PREDICT_GOA_NONEXP(
                prediction_release_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, windowReleaseReady ->
                    tuple(meta, windowReleaseReady, 'groundtruth_targets.fasta')
                }
            )
        }
        if( enabledWindowMethods.contains('blast') ) {
            predictionChannels << PREDICT_BLAST(
                prediction_release_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, windowReleaseReady ->
                    tuple(meta, windowReleaseReady, 'groundtruth_targets.fasta')
                }
            )
        }
        if( enabledWindowMethods.contains('prott5') ) {
            predictionChannels << PREDICT_PROTT5(
                prediction_release_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, windowReleaseReady ->
                    tuple(meta, windowReleaseReady, 'groundtruth_targets.fasta')
                }
            )
        }
        if( enabledWindowMethods.contains('transfew') ) {
            predictionChannels << PREDICT_TRANSFEW(
                prediction_release_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, windowReleaseReady ->
                    tuple(meta, windowReleaseReady, 'groundtruth_targets.fasta')
                }
            )
        }
        if( enabledWindowMethods.contains('funbind') ) {
            predictionChannels << PREDICT_FUNBIND(
                prediction_release_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, windowReleaseReady ->
                    tuple(meta, windowReleaseReady, 'groundtruth_targets_split')
                }
            )
        }
        if( enabledWindowMethods.contains('deepgoplus') ) {
            predictionChannels << PREDICT_DEEPGOPLUS(
                prediction_release_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, windowReleaseReady ->
                    tuple(meta, windowReleaseReady, 'groundtruth_targets.fasta')
                }
            )
        }

        all_predictions_ch = predictionChannels.tail().inject(predictionChannels.head()) { acc, ch -> acc.mix(ch) }
            .ifEmpty { error "No window prediction outputs were produced for ${enabledWindowMethods.join(', ')}" }
        grouped_predictions_ch = all_predictions_ch.groupTuple()
        validated_predictions_ch = grouped_predictions_ch.map { meta, predictionFiles ->
            def observedFiles = predictionFiles.collect { it.getFileName().toString() }
            def missingFiles = expectedPredictionFiles.findAll { !observedFiles.contains(it) }
            if( missingFiles ) {
                error "Missing prediction outputs before assembly for ${meta.release_id}: ${missingFiles.join(', ')}"
            }
            tuple(meta, predictionFiles)
        }

        assembled_predictions_ch = ASSEMBLE_WINDOW_PREDICTIONS(validated_predictions_ch)
    }

    packaged_window_predictions_ch = PACKAGE_WINDOW_PREDICTIONS(
        assembled_predictions_ch
            .join(window_inputs_ch.map { meta, t0Dir, t1Dir, methodNamesFile -> tuple(meta, t0Dir) })
            .map { meta, predictionsDir, t0Dir ->
                tuple(meta, t0Dir, predictionsDir)
            }
    )

    evaluation_inputs_ch = target_fasta_ch
        .join(assembled_predictions_ch)

    nk_results_ch = EVALUATE_NK(evaluation_inputs_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, predictionsDir -> tuple(meta, t0Dir, predictionsDir, nk, toi) })
    lk_results_ch = EVALUATE_LK(evaluation_inputs_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, predictionsDir -> tuple(meta, t0Dir, predictionsDir, lk, toi) })
    pk_results_ch = EVALUATE_PK(evaluation_inputs_ch.map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, predictionsDir -> tuple(meta, t0Dir, predictionsDir, pk, pkKnown, toi) })

    method_names_ch = assembled_predictions_ch
        .join(window_inputs_ch.map { meta, t0Dir, t1Dir, methodNamesFile -> tuple(meta, methodNamesFile) })
        .map { meta, predictionsDir, methodNamesFile ->
            tuple(meta, predictionsDir, methodNamesFile)
        }
    method_names_tsv_ch = GENERATE_METHOD_NAMES(method_names_ch)

    method_availability_inputs_ch = method_names_tsv_ch
        .join(nk_results_ch)
        .join(lk_results_ch)
        .join(pk_results_ch)
    method_availability_tsv_ch = GENERATE_METHOD_AVAILABILITY(method_availability_inputs_ch)

    release_bundle_inputs_ch = target_fasta_ch
        .join(nk_results_ch)
        .join(lk_results_ch)
        .join(pk_results_ch)
        .join(method_names_tsv_ch)
        .join(method_availability_tsv_ch)
        .map { meta, t0Dir, t1Dir, methodNamesFile, commonFasta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, nkDir, lkDir, pkDir, methodNamesTsv, methodAvailTsv ->
            tuple(meta, nk, lk, pk, pkKnown, toi, targetsTsv, targetsFasta, nkDir, lkDir, pkDir, methodNamesTsv, methodAvailTsv)
        }

    packaged_release_ch = PACKAGE_RELEASE_DIRECTORY(release_bundle_inputs_ch)
    validated_release_ch = VALIDATE_RELEASE_DIRECTORY(packaged_release_ch)
    published_release_ch = PUBLISH_RELEASE_DIRECTORY(validated_release_ch)
    release_catalog_ch = BUILD_RELEASE_CATALOG(published_release_ch)

    emit:
    published_release_ch
    release_catalog_ch
    packaged_window_predictions_ch
}
