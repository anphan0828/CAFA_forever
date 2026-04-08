#!/usr/bin/env python3
"""
Interactive CAFA/LAFA results visualization with validated release discovery.
"""

from html import escape

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
    METHOD_HELP_MSG,
    get_available_release_ids,
    get_available_timepoints,
    get_release_catalog,
    get_release_dates,
    get_release_dir,
    split_release_id,
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
            <div class="isu-header__title">LAFA - Longitudinal Assessment of Function Annotation</div>
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
            <a
              href="https://www.iastate.edu"
              class="site-footer__logo"
              role="img"
              aria-label="Iowa State University of Science and Technology"
            >
              {logo_svg}
              <span class="sr-only">Iowa State University of Science and Technology</span>
            </a>
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


def render_skip_link():
    st.markdown(
        """
        <a class="skip-link" href="#main-content">Skip to main content</a>
        """,
        unsafe_allow_html=True,
    )


def render_main_content_anchor():
    st.markdown(
        """
        <div id="main-content" tabindex="-1" aria-label="Main content">
          <span class="sr-only">Start of main content</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def validate_release_period_selection(selection, allowed_timepoints, field_name):
    if not isinstance(selection, (list, tuple)) or len(selection) != 2:
        raise ValueError(f"Invalid {field_name}: expected exactly two time points.")

    normalized = tuple(str(value) for value in selection)
    invalid_timepoints = [value for value in normalized if value not in allowed_timepoints]
    if invalid_timepoints:
        raise ValueError(
            f"Invalid {field_name}: unknown time point(s) {', '.join(sorted(set(invalid_timepoints)))}."
        )

    return normalized


@st.cache_data(show_spinner=False)
def load_method_names(method_names_file):
    if method_names_file.exists():
        df_methods = pd.read_csv(method_names_file, sep="\t", dtype=str)
        validate_required_columns(df_methods, REQUIRED_METHOD_COLUMNS, method_names_file.name)
        df_methods = df_methods.dropna(subset=["filename", "label"])
        # Temporary tagging baseline methods
        df_methods["is_baseline"] = df_methods["label"].str.isin(["Naive", "BLAST", "ProtT5", "GOA Non-exp"], case=False, na=False)
        df_methods["label"] = df_methods.apply(lambda row: f"{row['label']} (Baseline)" if row["is_baseline"] else row["label"], axis=1)
        return dict(zip(df_methods["filename"].str.strip(), df_methods["label"].str.strip(), df_methods["is_baseline"]))
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
        "LK & PK",
    ]
    return pd.DataFrame(
        [{"Combination": label, "Count": int(counts.get(label, 0))} for label in ordered_labels]
    )


@st.cache_data(show_spinner=False)
def load_target_overlap_counts(release_dir):
    nk_ids = set(load_ground_truth_profile(release_dir / "groundtruth_NK.tsv")["entries"])
    lk_ids = set(load_ground_truth_profile(release_dir / "groundtruth_LK.tsv")["entries"])
    pk_ids = set(load_ground_truth_profile(release_dir / "groundtruth_PK.tsv")["entries"])

    counts = {
        "NK only": len(nk_ids),
        "LK only": len(lk_ids - pk_ids),
        "PK only": len(pk_ids - lk_ids),
        "LK & PK": len(lk_ids & pk_ids),
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
        subplot_titles=[f"{ASPECT_NAMES[aspect]}" for aspect in aspects],
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
                name=f"{ASPECT_NAMES[aspect]}",
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
            tickangle=-45,
            linecolor="#212529",
        )

    fig.update_layout(
        height=600,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
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
            contour_levels = np.arange(0.1, 1.0, 0.1)
            recall_grid = np.linspace(0.01, 0.99, 500)
            for level in contour_levels:
                valid_recall = recall_grid[recall_grid > (level / 2)]
                precision_vals = (level * valid_recall) / ((2 * valid_recall) - level)
                valid_mask = (precision_vals >= 0) & (precision_vals <= 1)
                valid_recall = valid_recall[valid_mask]
                precision_vals = precision_vals[valid_mask]
                if len(valid_recall) == 0:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=valid_recall,
                        y=precision_vals,
                        mode="lines",
                        line=dict(color="#b8b8b8", width=1.2, dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[valid_recall[-1]],
                        y=[precision_vals[-1]],
                        mode="text",
                        text=[f"F={level:.1f}"],
                        textfont=dict(size=10, color="#6c757d"),
                        textposition="middle left",
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

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
                        marker=dict(
                            color=color_map[method],
                            size=10,
                            symbol="circle-open",
                            line=dict(color=color_map[method], width=2),
                        ),
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
                fig.add_trace(
                    go.Scatter(
                        x=[best_row["rc_micro_w"]],
                        y=[best_row["pr_micro_w"]],
                        mode="markers",
                        marker=dict(color=color_map[method], size=5),
                        name=f"{method} best center",
                        showlegend=False,
                        legendgroup=method,
                        hoverinfo="skip",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            fig.update_xaxes(row=row_idx, col=col_idx, range=[0, 1], linecolor="#212529", tickfont=dict(size=14, color="#212529"))
            fig.update_yaxes(row=row_idx, col=col_idx, range=[0, 1], showgrid=False, linecolor="#212529", tickfont=dict(size=14, color="#212529"))

    fig.update_layout(
        height=980,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),   
    )
    for row in range(1, 4):
        for col in range(1, 4):
            fig.update_xaxes(title_text="Recall" if row == 3 else "", row=row, col=col, title_font_color="#212529")
            fig.update_yaxes(title_text="Precision" if col == 1 else "", row=row, col=col, title_font_color="#212529")
    return fig


def build_method_availability_lookup(release_bundles):
    release_lookup = {}
    comparable_sets = []

    for release_id, bundle in release_bundles.items():
        frame = bundle["method_availability"][["Method", "NK", "LK", "PK", "AvailableInAllSubsets"]].copy()
        release_lookup[release_id] = frame.set_index("Method")[["NK", "LK", "PK"]].astype(bool).to_dict("index")
        comparable_sets.append(set(frame.loc[frame["AvailableInAllSubsets"], "Method"]))

    comparable_methods = sorted(set.intersection(*comparable_sets)) if comparable_sets else []
    return release_lookup, comparable_methods


def create_average_f1_chart(release_bundles, selected_methods):
    rows = []
    for release_id, bundle in release_bundles.items():
        for method in selected_methods:
            values = []
            for subset in SUBSETS:
                subset_df = bundle["best"][subset]
                method_rows = subset_df[subset_df["method"] == method]
                values.extend(method_rows["f_micro_w"].astype(float).tolist())

            if not values:
                continue

            rows.append(
                {
                    "Release": release_id,
                    "Method": method,
                    "Average F-score": float(np.mean(values)),
                }
            )
    colors = px.colors.qualitative.Set2
    df_average = pd.DataFrame(rows)
    fig = px.bar(
        df_average,
        x="Method",
        y="Average F-score",
        color="Release",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=colors,
    )
    fig.update_layout(
        height=430,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529"),
        margin=dict(t=20, r=20, b=20, l=20),
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _subset_target_count(bundle, subset):
    return int(bundle["groundtruth"][subset]["stats"]["total"])

def create_subset_coverage_chart(release_bundles, selected_methods):
    subsets = ["NK", "LK", "PK"]
    subset_positions = {subset: idx for idx, subset in enumerate(subsets)}
    release_ids = list(release_bundles.keys())
    colors = px.colors.qualitative.Set2
    fig = go.Figure()

    for idx, release_id in enumerate(release_ids):
        bundle = release_bundles[release_id]
        bar_x = [subset_positions[subset] + (idx - (len(release_ids) - 1) / 2) * 0.28 for subset in subsets]
        bar_y = [_subset_target_count(bundle, subset) for subset in subsets]
        fig.add_trace(
            go.Bar(
                x=bar_x,
                y=bar_y,
                width=0.24,
                name=f"{release_id} targets",
                marker_color=colors[idx % len(colors)],
                opacity=0.78,
                hovertemplate="Release: "
                + release_id
                + "<br>Subset: %{customdata}<br>Targets: %{y}<extra></extra>",
                customdata=subsets,
            )
        )

    fig.update_layout(
        height=430,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529"),
        margin=dict(t=20, r=20, b=20, l=20),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(
            tickmode="array",
            tickvals=list(subset_positions.values()),
            ticktext=[SUBSET_LABELS[subset] for subset in subsets],
            title="Protein subset",
        ),
        yaxis=dict(title="Unique targets"),
    )
    return fig


def build_release_lookup(available_release_ids):
    return {split_release_id(release_id): release_id for release_id in available_release_ids}


def _default_selected_methods(comparable_methods):
    return comparable_methods[: min(4, len(comparable_methods))]


def render_method_selector(release_bundles):
    availability_lookup, comparable_methods = build_method_availability_lookup(release_bundles)
    # Tag baseline methods
    comparable_methods = [f"{method} (Baseline)" if method in ["Naive", "BLAST", "ProtT5", "GOA Non-exp"] else method for method in comparable_methods]
    if not comparable_methods:
        return [], comparable_methods

    st.markdown("**Methods available**")
    checkbox_columns = st.columns(1)
    selected_methods = []
    default_methods = set(_default_selected_methods(comparable_methods))

    for idx, method in enumerate(comparable_methods):
        release_details = []
        for release_id in release_bundles:
            subset_flags = availability_lookup[release_id].get(method, {})
            available_subsets = [subset for subset in SUBSETS if subset_flags.get(subset, False)]
            release_details.append(f"{release_id}: {', '.join(available_subsets) if available_subsets else 'Unavailable'}")

        with checkbox_columns[idx % len(checkbox_columns)]:
            if st.checkbox(
                method,
                value=method in default_methods,
                key=f"method_checkbox::{method}",
                help=METHOD_HELP_MSG.get(method, "No description available."),
            ):
                selected_methods.append(method)

    return selected_methods, comparable_methods


def resolve_selected_releases(available_release_ids):
    release_lookup = build_release_lookup(available_release_ids)
    available_timepoints = get_available_timepoints()
    allowed_timepoints = set(available_timepoints)
    default_primary = split_release_id(available_release_ids[0])
    default_secondary = split_release_id(available_release_ids[1]) if len(available_release_ids) > 1 else default_primary
    
    st.markdown(
        "Available time points: "
        + ", ".join(" - ".join(timepoint) for timepoint in allowed_timepoints)
    )
    selector_columns = st.columns(2)
    with selector_columns[0]:
        st.markdown("**Primary release time periods**")
        primary_timepoints = st.select_slider(
            "Select a primary release period",
            options=available_timepoints,
            value=default_primary,
            help="Time points are parsed directly from validated release folder names.",
        )

    with selector_columns[1]:
        # st.markdown("**Optional comparison release**")
        compare_two_periods = st.checkbox(
            "Compare a second release period",
            value=len(available_release_ids) > 1,
        )

    secondary_timepoints = None
    if compare_two_periods:
        with selector_columns[1]:
            secondary_timepoints = st.select_slider(
                "Select a release period to compare",
                options=available_timepoints,
                value=default_secondary,
                help="Choose a second release period to compare against the primary selection.",
            )

    try:
        primary_timepoints = validate_release_period_selection(
            primary_timepoints, allowed_timepoints, "primary release period"
        )
        if secondary_timepoints is not None:
            secondary_timepoints = validate_release_period_selection(
                secondary_timepoints, allowed_timepoints, "secondary release period"
            )
    except ValueError as exc:
        return [], [str(exc)], None, None

    selected_release_ids = []
    missing_periods = []
    for timepoint_pair in [primary_timepoints, secondary_timepoints]:
        if timepoint_pair is None:
            continue
        release_id = release_lookup.get(tuple(timepoint_pair))
        if release_id is None:
            missing_periods.append(" - ".join(timepoint_pair))
            continue
        if release_id not in selected_release_ids:
            selected_release_ids.append(release_id)

    

    primary_release_id = release_lookup.get(tuple(primary_timepoints))
    secondary_release_id = release_lookup.get(tuple(secondary_timepoints)) if secondary_timepoints else None
    return selected_release_ids, missing_periods, primary_release_id, secondary_release_id


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
    start_timepoint, end_timepoint = split_release_id(release_id)
    with st.container(border=True):
        st.markdown(f"**{start_timepoint} to {end_timepoint}**")
        row_one_left, row_one_right = st.columns(2)
        with row_one_left:
            st.markdown(f"**GOA start**: {dates.get('goa_start', 'N/A')}")
        with row_one_right:
            st.markdown(f"**GOA end**: {dates.get('goa_end', 'N/A')}")

        row_two_left, row_two_right = st.columns(2)
        with row_two_left:
            st.markdown(f"**UniProt start**: {dates.get('uniprot_start', 'N/A')}")
        with row_two_right:
            st.markdown(f"**UniProt end**: {dates.get('uniprot_end', 'N/A')}")

        st.markdown(f"**Unique targets across NK/LK/PK**: {gt_union}")


def render_invalid_release_warning(catalog):
    if not catalog["invalid"]:
        return

    with st.expander("Excluded releases", expanded=False):
        for release_id, reasons in catalog["invalid"].items():
            st.markdown(f"**{escape(str(release_id))}**")
            for reason in reasons:
                st.markdown(f"- {escape(str(reason))}")


def main():
    render_skip_link()
    render_iastate_header()
    render_main_content_anchor()
    st.title("LAFA")
    st.markdown("Longitudinal Assessment of Function Annotation")

    catalog = get_release_catalog()
    available_release_ids = get_available_release_ids()
    render_invalid_release_warning(catalog)

    if not available_release_ids:
        st.error("No validated releases are available under data/releases.")
        return

    selected_release_ids, missing_periods, primary_release_id, secondary_release_id = resolve_selected_releases(
        available_release_ids
    )
    if missing_periods:
        st.error(
            "No validated release folder matches: "
            + ", ".join(missing_periods)
            + ". Choose one of the available release periods listed below the sliders."
        )
        return
    if not selected_release_ids:
        st.error("Select at least one valid release period.")
        return

    with st.spinner("Loading validated release data..."):
        release_bundles = {}
        load_errors = {}
        for release_id in selected_release_ids:
            try:
                release_bundles[release_id] = load_release_bundle(release_id)
            except Exception as exc:
                load_errors[release_id] = exc

    if load_errors:
        failed_release_ids = ", ".join(sorted(load_errors))
        st.error(
            "One or more selected releases could not be loaded because their published data is unavailable or invalid."
        )
        st.caption(f"Affected release(s): {failed_release_ids}")
    if not release_bundles:
        return

    context_columns = st.columns(2)
    with context_columns[0]:
        if primary_release_id in release_bundles:
            render_release_card(primary_release_id, release_bundles[primary_release_id])
    with context_columns[1]:
        if secondary_release_id in release_bundles and secondary_release_id != primary_release_id:
            render_release_card(secondary_release_id, release_bundles[secondary_release_id])

    method_col, coverage_col = st.columns([1.5, 1])
    with method_col:
        col1, col2 = st.columns([0.3, 1.2])
        with col1:
            selected_methods, comparable_methods = render_method_selector(release_bundles)

    if not comparable_methods:
        st.error("No methods are available across all selected releases and subsets.")
        return
    try:
        selected_methods = validate_selected_methods(selected_methods, comparable_methods)
    except ValueError as exc:
        st.warning(str(exc))
        return

    with method_col:
        with col2:
            st.markdown("**Average F-score across all 3 GO aspects and all 3 protein subsets**")
            st.markdown("F-scores are micro-averaged across proteins.")
            st.plotly_chart(create_average_f1_chart(release_bundles, selected_methods), use_container_width=True)

    with coverage_col:
        st.markdown("**Target counts by protein subset**")
        st.markdown("Bars show unique targets in NK, LK, and PK for each selected release.")
        st.plotly_chart(create_subset_coverage_chart(release_bundles, selected_methods), use_container_width=True)

    tab_curves, tab_metrics, tab_summary = st.tabs(
        ["Precision-Recall Curves", "F-score Breakdown", "Summary Tables"]
    )

    with tab_curves:
        curve_tabs = st.tabs(list(release_bundles.keys()))
        for tab, release_id in zip(curve_tabs, release_bundles):
            with tab:
                fig = create_precision_recall_plot(release_bundles[release_id], selected_methods, release_id)
                st.plotly_chart(fig, use_container_width=True)

    with tab_metrics:
        selected_metric = "f_micro_w"
        validate_enum(selected_metric, ALLOWED_METRICS, "metric")
        st.markdown("**F-score by aspect and protein subset**")

        plot_columns = st.columns(len(release_bundles))
        for column, release_id in zip(plot_columns, release_bundles):
            with column:
                bundle = release_bundles[release_id]
                st.markdown(f"**{release_id}**")
                fig = create_interactive_performance_plot(bundle, selected_methods, selected_metric, release_id)
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
