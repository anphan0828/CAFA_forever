#!/usr/bin/env python3
"""
Interactive CAFA/LAFA results visualization with validated release discovery.
"""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    GO_ASPECTS,
    STATIC_DIR,
    STREAMLIT_CONFIG,
    SUBSETS,
    get_available_timepoints,
    get_release_catalog,
    get_release_dates,
    get_release_dir,
)

st.set_page_config(**STREAMLIT_CONFIG)

ASPECT_NAMES = {
    "P": "Biological Process",
    "F": "Molecular Function",
    "C": "Cellular Component",
    **GO_ASPECTS,
}

ONTOLOGY_DICT = {
    "biological_process": "BPO",
    "molecular_function": "MFO",
    "cellular_component": "CCO",
}

SUBSET_LABELS = {
    "NK": "No Knowledge",
    "LK": "Limited Knowledge",
    "PK": "Partial Knowledge",
}

MAX_SELECTED_METHODS = 30
ALLOWED_PLOT_TYPES = {"consolidated", "individual"}
ALLOWED_METRICS = {"pr_micro_w", "rc_micro_w", "f_micro_w", "cov_w"}
ALLOWED_ASPECTS = {"biological_process", "molecular_function", "cellular_component"}
GROUND_TRUTH_ASPECT_MAP = {
    "P": "biological_process",
    "F": "molecular_function",
    "C": "cellular_component",
}

REQUIRED_GT_COLUMNS = {"EntryID", "aspect"}
REQUIRED_BEST_COLUMNS = {
    "filename",
    "ns",
    "n",
    "pr_micro_w",
    "rc_micro_w",
    "f_micro_w",
    "cov_w",
    "tau",
}
REQUIRED_ALL_COLUMNS = {"filename", "ns", "tau", "cov", "rc_micro_w", "pr_micro_w", "f_micro_w"}
REQUIRED_METHOD_COLUMNS = {"filename", "label"}
REQUIRED_AVAILABILITY_COLUMNS = {"method", "NK", "LK", "PK"}
REQUIRED_OVERLAP_COLUMNS = {"combination", "count"}


def inject_iastate_theme():
    css_path = STATIC_DIR / "iastate" / "streamlit_iastate.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def _load_svg(filename, fallback_text):
    svg_path = STATIC_DIR / "iastate" / filename
    if svg_path.exists():
        return svg_path.read_text()
    return f"<span>{fallback_text}</span>"


def render_iastate_header():
    logo_svg = _load_svg(
        "iowa-state-university-logo-with-tagline-red.svg",
        "Iowa State University of Science and Technology",
    )
    st.markdown(
        f"""
        <header class="isu-header">
          <div class="isu-header__inner">
            <div class="isu-header__logo">{logo_svg}</div>
            <div class="isu-header__title">LAFA - Longitudinal Assessment of Functional Annotation</div>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_iastate_footer():
    logo_svg = _load_svg(
        "iowa-state-university-logo-with-tagline-sci-tech.svg",
        "Iowa State University of Science and Technology",
    )
    st.markdown(
        f"""
        <footer class="site-footer">
          <div class="site-footer__flex-wrap">
            <a href="https://www.iastate.edu" class="site-footer__logo">{logo_svg}</a>
          </div>
          <div class="site-footer__bottom-wrap">
            <div class="site-footer__copyright">
              <p>© Iowa State University of Science and Technology</p>
            </div>
          </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


inject_iastate_theme()


def validate_enum(value, allowed_values, field_name):
    if value not in allowed_values:
        raise ValueError(f"Invalid {field_name}: {value}")
    return value


def validate_required_columns(df, required_columns, dataframe_name):
    missing_cols = sorted(required_columns - set(df.columns))
    if missing_cols:
        raise ValueError(f"{dataframe_name} missing required columns: {', '.join(missing_cols)}")


def validate_numeric_columns(df, numeric_columns, dataframe_name):
    for col in numeric_columns:
        if col not in df.columns:
            continue
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if numeric_series.isna().any():
            raise ValueError(f"{dataframe_name} has non-numeric values in column '{col}'")


def validate_ns_values(df, dataframe_name):
    if "ns" not in df.columns:
        return
    invalid_ns = sorted(set(df["ns"].dropna().astype(str)) - ALLOWED_ASPECTS)
    if invalid_ns:
        raise ValueError(f"{dataframe_name} has invalid ns values: {', '.join(invalid_ns)}")


def validate_gt_aspects(df, dataframe_name):
    invalid_aspects = sorted(set(df["aspect"].dropna().astype(str)) - set(GROUND_TRUTH_ASPECT_MAP))
    if invalid_aspects:
        raise ValueError(f"{dataframe_name} has invalid aspect values: {', '.join(invalid_aspects)}")


def validate_selected_methods(selected_methods, available_methods):
    if not selected_methods:
        raise ValueError("Please select at least one method to compare.")
    if len(selected_methods) > MAX_SELECTED_METHODS:
        raise ValueError(f"Please select {MAX_SELECTED_METHODS} methods or fewer.")
    invalid_methods = sorted(set(selected_methods) - set(available_methods))
    if invalid_methods:
        raise ValueError(f"Invalid method selection: {', '.join(invalid_methods)}")
    return selected_methods


@st.cache_data(show_spinner=False)
def load_method_names(method_names_file):
    if method_names_file.exists():
        df_methods = pd.read_csv(method_names_file, sep="\t", dtype=str)
        validate_required_columns(df_methods, REQUIRED_METHOD_COLUMNS, method_names_file.name)
        df_methods = df_methods.dropna(subset=["filename", "label"])
        return dict(zip(df_methods["filename"].str.strip(), df_methods["label"].str.strip()))
    return {}


@st.cache_data(show_spinner=False)
def load_ground_truth_profile(ground_truth_file):
    df_gt = pd.read_csv(ground_truth_file, sep="\t", dtype=str)
    validate_required_columns(df_gt, REQUIRED_GT_COLUMNS, str(ground_truth_file))
    validate_gt_aspects(df_gt, str(ground_truth_file))

    df_gt = df_gt.dropna(subset=["EntryID", "aspect"])
    df_gt["aspect_full"] = df_gt["aspect"].map(GROUND_TRUTH_ASPECT_MAP)

    stats = {}
    for aspect in ALLOWED_ASPECTS:
        stats[aspect] = df_gt[df_gt["aspect_full"] == aspect]["EntryID"].nunique()
    stats["total"] = df_gt["EntryID"].nunique()

    return {
        "stats": stats,
        "entries": sorted(df_gt["EntryID"].astype(str).unique().tolist()),
    }


@st.cache_data(show_spinner=False)
def load_evaluation_data(results_dir, subset_name, method_names):
    best_f_file = results_dir / "evaluation_best_f_micro_w.tsv"
    df_best = pd.read_csv(best_f_file, sep="\t")
    validate_required_columns(df_best, REQUIRED_BEST_COLUMNS, str(best_f_file))
    validate_ns_values(df_best, str(best_f_file))
    validate_numeric_columns(
        df_best,
        ["n", "pr_micro_w", "rc_micro_w", "f_micro_w", "cov_w", "tau"],
        str(best_f_file),
    )
    df_best["method"] = df_best["filename"].map(method_names).fillna(df_best["filename"])
    df_best["subset"] = subset_name
    return df_best


@st.cache_data(show_spinner=False)
def load_all_evaluation_data(results_dir, method_names):
    eval_all_file = results_dir / "evaluation_all.tsv"
    df_all = pd.read_csv(eval_all_file, sep="\t")
    validate_required_columns(df_all, REQUIRED_ALL_COLUMNS, str(eval_all_file))
    validate_ns_values(df_all, str(eval_all_file))
    validate_numeric_columns(df_all, ["tau", "cov", "rc_micro_w", "pr_micro_w", "f_micro_w"], str(eval_all_file))
    df_all["method"] = df_all["filename"].map(method_names).fillna(df_all["filename"])
    return df_all


def _coerce_bool_series(series):
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        raise ValueError("boolean availability columns must contain true/false style values")
    return normalized


@st.cache_data(show_spinner=False)
def load_method_availability(release_dir, method_names):
    precomputed_file = release_dir / "method_availability.tsv"
    if precomputed_file.exists():
        df_availability = pd.read_csv(precomputed_file, sep="\t")
        validate_required_columns(df_availability, REQUIRED_AVAILABILITY_COLUMNS, str(precomputed_file))
        for subset in SUBSETS:
            df_availability[subset] = _coerce_bool_series(df_availability[subset])
        df_availability["Method"] = df_availability["method"].astype(str)
    else:
        subset_methods = {}
        for subset in SUBSETS:
            results_dir = release_dir / f"results_{subset}"
            df_subset = load_evaluation_data(results_dir, subset, method_names)
            subset_methods[subset] = set(df_subset["method"].astype(str).unique())

        all_methods = sorted(set().union(*subset_methods.values()))
        rows = []
        for method in all_methods:
            rows.append(
                {
                    "Method": method,
                    "NK": method in subset_methods["NK"],
                    "LK": method in subset_methods["LK"],
                    "PK": method in subset_methods["PK"],
                }
            )
        df_availability = pd.DataFrame(rows)

    df_availability["AvailableInAllSubsets"] = df_availability[SUBSETS].all(axis=1)
    return df_availability.sort_values("Method").reset_index(drop=True)


def _ordered_overlap_rows(counts):
    ordered_labels = [
        "NK only",
        "LK only",
        "PK only",
        "NK & LK",
        "NK & PK",
        "LK & PK",
        "NK & LK & PK",
    ]
    return pd.DataFrame(
        [{"Combination": label, "Count": int(counts.get(label, 0))} for label in ordered_labels]
    )


@st.cache_data(show_spinner=False)
def load_target_overlap_counts(release_dir):
    precomputed_file = release_dir / "target_overlap.tsv"
    if precomputed_file.exists():
        df_overlap = pd.read_csv(precomputed_file, sep="\t")
        validate_required_columns(df_overlap, REQUIRED_OVERLAP_COLUMNS, str(precomputed_file))
        validate_numeric_columns(df_overlap, ["count"], str(precomputed_file))
        counts = dict(zip(df_overlap["combination"].astype(str), df_overlap["count"].astype(int)))
        return _ordered_overlap_rows(counts)

    nk_ids = set(load_ground_truth_profile(release_dir / "groundtruth_NK.tsv")["entries"])
    lk_ids = set(load_ground_truth_profile(release_dir / "groundtruth_LK.tsv")["entries"])
    pk_ids = set(load_ground_truth_profile(release_dir / "groundtruth_PK.tsv")["entries"])

    counts = {
        "NK only": len(nk_ids - lk_ids - pk_ids),
        "LK only": len(lk_ids - nk_ids - pk_ids),
        "PK only": len(pk_ids - nk_ids - lk_ids),
        "NK & LK": len((nk_ids & lk_ids) - pk_ids),
        "NK & PK": len((nk_ids & pk_ids) - lk_ids),
        "LK & PK": len((lk_ids & pk_ids) - nk_ids),
        "NK & LK & PK": len(nk_ids & lk_ids & pk_ids),
    }
    return _ordered_overlap_rows(counts)


@st.cache_data(show_spinner=False)
def load_release_bundle(release_id):
    release_dir = get_release_dir(release_id)
    method_names = load_method_names(release_dir / "method_names.tsv")

    groundtruth = {}
    best = {}
    curves = {}
    for subset in SUBSETS:
        groundtruth[subset] = load_ground_truth_profile(release_dir / f"groundtruth_{subset}.tsv")
        results_dir = release_dir / f"results_{subset}"
        best[subset] = load_evaluation_data(results_dir, subset, method_names)
        curves[subset] = load_all_evaluation_data(results_dir, method_names)

    return {
        "release_id": release_id,
        "release_dir": release_dir,
        "dates": get_release_dates(release_id),
        "method_names": method_names,
        "groundtruth": groundtruth,
        "best": best,
        "all": curves,
        "method_availability": load_method_availability(release_dir, method_names),
        "target_overlap": load_target_overlap_counts(release_dir),
    }


def _metric_value(df, method, aspect, metric):
    filtered = df[(df["method"] == method) & (df["ns"] == aspect)]
    if filtered.empty:
        return 0.0
    return float(filtered.iloc[0][metric])


def create_interactive_performance_plot(bundle, selected_methods, selected_metric, release_label):
    aspects = ["biological_process", "molecular_function", "cellular_component"]
    aspect_symbols = ["circle", "square", "triangle-up"]
    aspect_labels = ["BPO", "MFO", "CCO"]
    colors = {"NK": "#b31b1b", "LK": "#1f77b4", "PK": "#2a9d5b"}

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"{ASPECT_NAMES[aspect]} ({release_label})" for aspect in aspects],
        shared_yaxes=True,
    )

    x_pos = np.arange(len(selected_methods))
    dodge_offset = 0.18

    for subset, legend_name in SUBSET_LABELS.items():
        fig.add_trace(
            go.Scatter(
                name=legend_name,
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=colors[subset], size=12, symbol="circle"),
                showlegend=True,
                legendgroup=subset,
            ),
            row=1,
            col=1,
        )

    for i, (aspect, symbol, label) in enumerate(zip(aspects, aspect_symbols, aspect_labels)):
        fig.add_trace(
            go.Scatter(
                name=f"{label} ({ASPECT_NAMES[aspect]})",
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="#212529", size=12, symbol=symbol),
                showlegend=True,
                legendgroup=f"aspect_{aspect}",
            ),
            row=1,
            col=1,
        )

        for subset_idx, subset in enumerate(SUBSETS):
            subset_df = bundle["best"][subset]
            values = [_metric_value(subset_df, method, aspect, selected_metric) for method in selected_methods]
            positions = [idx + (subset_idx - 1) * dodge_offset for idx in range(len(selected_methods))]

            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=values,
                    mode="markers",
                    marker=dict(
                        color=colors[subset],
                        size=12,
                        symbol=symbol,
                        line=dict(color="#111111", width=1),
                    ),
                    name="",
                    showlegend=False,
                    hovertemplate=(
                        f"{SUBSET_LABELS[subset]}<br>{ASPECT_NAMES[aspect]}<br>"
                        "Method: %{customdata}<br>Value: %{y:.3f}<extra></extra>"
                    ),
                    customdata=selected_methods,
                ),
                row=1,
                col=i + 1,
            )

        fig.update_xaxes(
            row=1,
            col=i + 1,
            tickvals=x_pos,
            ticktext=selected_methods,
            tickfont=dict(size=14, color="#212529"),
            linecolor="#212529",
        )

    fig.update_layout(
        height=520,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, xanchor="center", x=0.5),
    )
    fig.update_yaxes(
        title_text=selected_metric.upper(),
        row=1,
        col=1,
        range=[0, 1],
        tickfont=dict(size=14, color="#212529"),
        linecolor="#212529",
    )
    return fig


def create_consolidated_performance_plot(bundle, selected_methods, release_label):
    metrics = ["pr_micro_w", "rc_micro_w", "f_micro_w"]
    metric_names = ["Precision", "Recall", "F-score"]
    aspects = ["biological_process", "molecular_function", "cellular_component"]
    aspect_symbols = ["circle", "square", "triangle-up"]
    aspect_labels = ["BPO", "MFO", "CCO"]
    colors = {"NK": "#b31b1b", "LK": "#1f77b4", "PK": "#2a9d5b"}

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"{name} ({release_label})" for name in metric_names],
        shared_yaxes=True,
    )

    x_pos = np.arange(len(selected_methods))
    dodge_offset = 0.18

    for subset, legend_name in SUBSET_LABELS.items():
        fig.add_trace(
            go.Scatter(
                name=legend_name,
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=colors[subset], size=12, symbol="circle"),
                showlegend=True,
                legendgroup=subset,
            ),
            row=1,
            col=1,
        )

    for aspect, symbol, label in zip(aspects, aspect_symbols, aspect_labels):
        fig.add_trace(
            go.Scatter(
                name=f"{label} ({ASPECT_NAMES[aspect]})",
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="#212529", size=12, symbol=symbol),
                showlegend=True,
                legendgroup=f"aspect_{aspect}",
            ),
            row=1,
            col=1,
        )

    for metric_index, metric in enumerate(metrics):
        for aspect_index, aspect in enumerate(aspects):
            for subset_idx, subset in enumerate(SUBSETS):
                subset_df = bundle["best"][subset]
                values = [_metric_value(subset_df, method, aspect, metric) for method in selected_methods]
                positions = [idx + (subset_idx - 1) * dodge_offset for idx in range(len(selected_methods))]
                fig.add_trace(
                    go.Scatter(
                        x=positions,
                        y=values,
                        mode="markers",
                        marker=dict(
                            color=colors[subset],
                            size=10,
                            symbol=aspect_symbols[aspect_index],
                            line=dict(color="#111111", width=1),
                        ),
                        name="",
                        showlegend=False,
                        hovertemplate=(
                            f"{metric_names[metric_index]}<br>{SUBSET_LABELS[subset]}<br>{ASPECT_NAMES[aspect]}<br>"
                            "Method: %{customdata}<br>Value: %{y:.3f}<extra></extra>"
                        ),
                        customdata=selected_methods,
                    ),
                    row=1,
                    col=metric_index + 1,
                )

        fig.update_xaxes(
            row=1,
            col=metric_index + 1,
            tickvals=x_pos,
            ticktext=selected_methods,
            tickfont=dict(size=14, color="#212529"),
            linecolor="#212529",
        )

    fig.update_layout(
        height=520,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )
    fig.update_yaxes(
        title_text="Score",
        row=1,
        col=1,
        range=[0, 1],
        tickfont=dict(size=14, color="#212529"),
        linecolor="#212529",
    )
    return fig


def create_precision_recall_plot(bundle, selected_methods, release_label):
    aspects = ["biological_process", "molecular_function", "cellular_component"]
    colors = px.colors.qualitative.Set1
    color_map = {method: colors[idx % len(colors)] for idx, method in enumerate(selected_methods)}

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            f"{ONTOLOGY_DICT[aspect]} - {SUBSET_LABELS[subset]} ({release_label})"
            for subset in SUBSETS
            for aspect in aspects
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for row_idx, subset in enumerate(SUBSETS, start=1):
        df_all = bundle["all"][subset]
        df_best = bundle["best"][subset]

        for col_idx, aspect in enumerate(aspects, start=1):
            aspect_all = df_all[(df_all["ns"] == aspect) & (df_all["method"].isin(selected_methods))]
            aspect_best = df_best[(df_best["ns"] == aspect) & (df_best["method"].isin(selected_methods))]

            for method in selected_methods:
                method_curve = aspect_all[aspect_all["method"] == method].sort_values("rc_micro_w")
                if method_curve.empty:
                    continue

                showlegend = row_idx == 1 and col_idx == 1
                fig.add_trace(
                    go.Scatter(
                        x=method_curve["rc_micro_w"],
                        y=method_curve["pr_micro_w"],
                        mode="lines",
                        name=method,
                        line=dict(color=color_map[method], width=2),
                        showlegend=showlegend,
                        legendgroup=method,
                        hovertemplate=(
                            f"{method}<br>{SUBSET_LABELS[subset]}<br>{ASPECT_NAMES[aspect]}<br>"
                            "Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                best_point = aspect_best[aspect_best["method"] == method]
                if best_point.empty:
                    continue
                best_row = best_point.iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=[best_row["rc_micro_w"]],
                        y=[best_row["pr_micro_w"]],
                        mode="markers",
                        marker=dict(color=color_map[method], size=8, line=dict(color="#111111", width=1)),
                        name=f"{method} best",
                        showlegend=False,
                        legendgroup=method,
                        hovertemplate=(
                            f"{method} F-max<br>{SUBSET_LABELS[subset]}<br>{ASPECT_NAMES[aspect]}<br>"
                            f"F-score: {best_row['f_micro_w']:.3f}<br>"
                            "Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            fig.update_xaxes(row=row_idx, col=col_idx, range=[0, 1], linecolor="#212529")
            fig.update_yaxes(row=row_idx, col=col_idx, range=[0, 1], linecolor="#212529")

    fig.update_layout(
        height=980,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
    )
    for row in range(1, 4):
        for col in range(1, 4):
            fig.update_xaxes(title_text="Recall" if row == 3 else "", row=row, col=col)
            fig.update_yaxes(title_text="Precision" if col == 1 else "", row=row, col=col)
    return fig


def build_method_availability_matrix(release_bundles):
    merged = None
    comparable_sets = []

    for release_id, bundle in release_bundles.items():
        frame = bundle["method_availability"][["Method", "NK", "LK", "PK", "AvailableInAllSubsets"]].copy()
        frame = frame.rename(
            columns={
                "NK": f"{release_id} NK",
                "LK": f"{release_id} LK",
                "PK": f"{release_id} PK",
                "AvailableInAllSubsets": f"{release_id} comparable",
            }
        )
        comparable_sets.append(set(frame.loc[frame[f"{release_id} comparable"], "Method"]))
        merged = frame if merged is None else merged.merge(frame, on="Method", how="outer")

    bool_columns = [col for col in merged.columns if col != "Method"]
    merged[bool_columns] = merged[bool_columns].fillna(False).astype(bool)
    comparable_methods = sorted(set.intersection(*comparable_sets)) if comparable_sets else []

    display_df = merged.copy()
    for col in bool_columns:
        display_df[col] = display_df[col].map({True: "Yes", False: "No"})
    return display_df.sort_values("Method").reset_index(drop=True), comparable_methods


def build_overlap_dataframe(release_bundles):
    frames = []
    for release_id, bundle in release_bundles.items():
        frame = bundle["target_overlap"].copy()
        frame["Release"] = release_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def create_overlap_chart(release_bundles):
    overlap_df = build_overlap_dataframe(release_bundles)
    fig = px.bar(
        overlap_df,
        x="Combination",
        y="Count",
        color="Release",
        barmode="group",
        title="Overlap-aware target counts across NK/LK/PK",
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=overlap_df["Combination"].unique())
    return fig


def build_target_summary_table(release_id, bundle, selected_methods):
    aspects = ["biological_process", "molecular_function", "cellular_component"]
    rows = []

    for subset in SUBSETS:
        subset_df = bundle["best"][subset]
        gt_stats = bundle["groundtruth"][subset]["stats"]

        for aspect in aspects:
            aspect_df = subset_df[subset_df["ns"] == aspect]
            for method in selected_methods:
                method_row = aspect_df[aspect_df["method"] == method]
                if method_row.empty:
                    continue
                row = method_row.iloc[0]
                ground_truth_count = gt_stats.get(aspect, 0)
                predicted_count = int(row["n"])
                pct = (predicted_count / ground_truth_count * 100) if ground_truth_count else 0.0

                rows.append(
                    {
                        "Release": release_id,
                        "Subset": subset,
                        "Aspect": ASPECT_NAMES[aspect],
                        "Method": method,
                        "Targets Predicted": predicted_count,
                        "Ground Truth Targets": int(ground_truth_count),
                        "Target Coverage %": f"{pct:.1f}",
                        "Precision": f"{row['pr_micro_w']:.3f}",
                        "Recall": f"{row['rc_micro_w']:.3f}",
                        "F-score": f"{row['f_micro_w']:.3f}",
                        "Coverage": f"{row['cov_w']:.3f}",
                        "Threshold": f"{row['tau']:.3f}",
                    }
                )

    return pd.DataFrame(rows)


def render_release_card(release_id, bundle):
    dates = bundle["dates"]
    gt_union = len(
        set(bundle["groundtruth"]["NK"]["entries"])
        | set(bundle["groundtruth"]["LK"]["entries"])
        | set(bundle["groundtruth"]["PK"]["entries"])
    )
    st.subheader(release_id)
    st.markdown(f"GO start: {dates.get('go_start', 'N/A')}")
    st.markdown(f"GO end: {dates.get('go_end', 'N/A')}")
    st.markdown(f"UniProt start: {dates.get('uniprot_start', 'N/A')}")
    st.markdown(f"UniProt end: {dates.get('uniprot_end', 'N/A')}")
    st.markdown(f"Unique targets across NK/LK/PK: {gt_union}")


def render_invalid_release_warning(catalog):
    if not catalog["invalid"]:
        return

    with st.expander("Excluded releases", expanded=False):
        for release_id, reasons in catalog["invalid"].items():
            st.markdown(f"**{escape(str(release_id))}**")
            for reason in reasons:
                st.markdown(f"- {escape(str(reason))}")


def main():
    render_iastate_header()
    st.title("LAFA")
    st.markdown("Longitudinal Assessment of Functional Annotation.")

    catalog = get_release_catalog()
    available_timepoints = get_available_timepoints()
    render_invalid_release_warning(catalog)

    if not available_timepoints:
        st.error("No validated releases are available under data/releases.")
        return

    default_value = (
        (available_timepoints[0], available_timepoints[-1])
        if len(available_timepoints) > 1
        else (available_timepoints[0], available_timepoints[0])
    )

    selected_range = st.select_slider(
        "Compare release windows",
        options=available_timepoints,
        value=default_value,
        help="Select the two release windows to compare. Only validated releases are listed.",
    )
    selected_release_ids = list(dict.fromkeys(selected_range))

    with st.spinner("Loading validated release data..."):
        release_bundles = {}
        load_errors = {}
        for release_id in selected_release_ids:
            try:
                release_bundles[release_id] = load_release_bundle(release_id)
            except Exception as exc:
                load_errors[release_id] = str(exc)

    if load_errors:
        for release_id, error_message in load_errors.items():
            st.error(f"Failed to load {release_id}: {error_message}")
    if not release_bundles:
        return

    availability_matrix, comparable_methods = build_method_availability_matrix(release_bundles)
    if not comparable_methods:
        st.error("No methods are available across all selected releases and subsets.")
        return

    selected_methods = st.multiselect(
        "Select methods available across the selected releases",
        options=comparable_methods,
        default=comparable_methods[: min(4, len(comparable_methods))],
        help="Methods are restricted to the intersection of valid methods across the selected releases.",
    )
    try:
        selected_methods = validate_selected_methods(selected_methods, comparable_methods)
    except ValueError as exc:
        st.warning(str(exc))
        return

    release_columns = st.columns(len(release_bundles))
    for column, release_id in zip(release_columns, release_bundles):
        with column:
            render_release_card(release_id, release_bundles[release_id])

    st.header("Method Availability")
    st.markdown("Methods are exposed for comparison only when they are available in every subset of each selected release.")
    st.dataframe(availability_matrix, use_container_width=True, hide_index=True)

    st.header("Target Overlap")
    st.markdown(
        "This overlap-aware view counts unique proteins across NK, LK, and PK instead of treating the subsets as mutually exclusive."
    )
    st.plotly_chart(create_overlap_chart(release_bundles), use_container_width=True)

    tab_metrics, tab_curves, tab_summary = st.tabs(
        ["Performance Metrics", "Precision-Recall Curves", "Summary Tables"]
    )

    with tab_metrics:
        st.markdown(
            "Precision, recall, and F-score are micro-averaged across proteins. "
            "Coverage corresponds to the weighted coverage metric in the evaluation tables."
        )
        plot_type = st.radio(
            "Metric view",
            options=["consolidated", "individual"],
            format_func=lambda value: {
                "consolidated": "All Metrics (Precision, Recall, F-score)",
                "individual": "Single Metric",
            }[value],
            horizontal=True,
        )
        validate_enum(plot_type, ALLOWED_PLOT_TYPES, "plot type")

        selected_metric = "f_micro_w"
        if plot_type == "individual":
            selected_metric = st.selectbox(
                "Metric",
                options=["pr_micro_w", "rc_micro_w", "f_micro_w", "cov_w"],
                format_func=lambda value: {
                    "pr_micro_w": "Precision",
                    "rc_micro_w": "Recall",
                    "f_micro_w": "F-score",
                    "cov_w": "Coverage",
                }[value],
                index=2,
            )
            validate_enum(selected_metric, ALLOWED_METRICS, "metric")

        plot_columns = st.columns(len(release_bundles))
        for column, release_id in zip(plot_columns, release_bundles):
            with column:
                bundle = release_bundles[release_id]
                if plot_type == "consolidated":
                    fig = create_consolidated_performance_plot(bundle, selected_methods, release_id)
                else:
                    fig = create_interactive_performance_plot(bundle, selected_methods, selected_metric, release_id)
                st.plotly_chart(fig, use_container_width=True)

    with tab_curves:
        curve_tabs = st.tabs(list(release_bundles.keys()))
        for tab, release_id in zip(curve_tabs, release_bundles):
            with tab:
                fig = create_precision_recall_plot(release_bundles[release_id], selected_methods, release_id)
                st.plotly_chart(fig, use_container_width=True)

    with tab_summary:
        summary_frames = [
            build_target_summary_table(release_id, bundle, selected_methods)
            for release_id, bundle in release_bundles.items()
        ]
        df_summary = pd.concat(summary_frames, ignore_index=True)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Summary CSV",
            data=df_summary.to_csv(index=False),
            file_name="lafa_comparison_summary.csv",
            mime="text/csv",
        )

    render_iastate_footer()


if __name__ == "__main__":
    main()
