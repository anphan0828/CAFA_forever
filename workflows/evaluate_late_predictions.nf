include {
    EVALUATE_LATE_NK
    EVALUATE_LATE_LK
    EVALUATE_LATE_PK
    MERGE_LATE_RESULTS
    FINALIZE_LATE_PREDICTIONS
} from '../modules/local/evaluation'

workflow EVALUATE_LATE_PREDICTIONS {
    take:
    late_eval_inputs_ch

    main:
    late_nk_ch = EVALUATE_LATE_NK(late_eval_inputs_ch)
    late_lk_ch = EVALUATE_LATE_LK(late_eval_inputs_ch)
    late_pk_ch = EVALUATE_LATE_PK(late_eval_inputs_ch)

    merged_results_ch = late_nk_ch
        .join(late_lk_ch)
        .join(late_pk_ch)

    merged_release_ch = MERGE_LATE_RESULTS(merged_results_ch)
    finalized_timepoint_ch = FINALIZE_LATE_PREDICTIONS(late_eval_inputs_ch)

    emit:
    merged_release_ch
    finalized_timepoint_ch
}
