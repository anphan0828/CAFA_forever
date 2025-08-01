#!/usr/bin/env python3
"""
Interactive CAFA Results Visualization with Streamlit

This Streamlit app provides interactive visualization of CAFA evaluation results.
Users can select which methods to compare using checkboxes and interactively
explore the results across different protein subsets and GO aspects.

Usage:
streamlit run streamlit_plot.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from config import get_available_timepoints, STREAMLIT_CONFIG, GO_ASPECTS

# Configure Streamlit page
st.set_page_config(**STREAMLIT_CONFIG)

# Use the GO_ASPECTS from config, but keep backward compatibility
ASPECT_NAMES = {
    'P': 'Biological Process',
    'F': 'Molecular Function', 
    'C': 'Cellular Component',
    **GO_ASPECTS
}

# Map ontology namespaces to full names (for plot titles)
ONTOLOGY_DICT = {
    'biological_process': 'BPO', 
    'molecular_function': 'MFO', 
    'cellular_component': 'CCO'
}

@st.cache_data
def load_method_names(method_names_file):
    """Load method names mapping from TSV file."""
    if method_names_file and Path(method_names_file).exists():
        df_methods = pd.read_csv(method_names_file, sep='\t')
        if 'filename' in df_methods.columns and 'method_name' in df_methods.columns:
            return dict(zip(df_methods['filename'], df_methods['method_name']))
        else:
            st.warning("method_names file should have 'filename' and 'method_name' columns")
            return {}
    else:
        st.warning("method_names file not found, using filenames as method names")
        return {}

@st.cache_data
def load_ground_truth_stats(ground_truth_file):
    """Load ground truth data and calculate statistics by aspect."""
    df_gt = pd.read_csv(ground_truth_file, sep='\t')
    
    # Map single letter aspects to full names for consistency
    df_gt['aspect_full'] = df_gt['aspect'].map({
        'P': 'biological_process',
        'F': 'molecular_function', 
        'C': 'cellular_component'
    })
    
    stats = {}
    
    # Count unique proteins per aspect
    for aspect in ['biological_process', 'molecular_function', 'cellular_component']:
        proteins_in_aspect = df_gt[df_gt['aspect_full'] == aspect]['EntryID'].nunique()
        stats[aspect] = proteins_in_aspect
    
    # Total unique proteins
    stats['total'] = df_gt['EntryID'].nunique()
    
    return stats

@st.cache_data
def load_evaluation_data(results_dir, subset_name, method_names):
    """Load and process evaluation data from a results directory."""
    
    # Load best F-score results for summary metrics
    best_f_file = results_dir / "evaluation_best_f.tsv"
    df_best = pd.read_csv(best_f_file, sep='\t')
    
    # Clean up method names
    df_best['method'] = df_best['filename'].map(method_names).fillna(df_best['filename'])
    df_best['subset'] = subset_name
    
    return df_best

@st.cache_data
def load_all_evaluation_data(results_dir, method_names):
    """Load all evaluation data for precision-recall curves."""
    eval_all_file = results_dir / "evaluation_all.tsv"
    df_all = pd.read_csv(eval_all_file, sep='\t')
    
    # Clean up method names
    df_all['method'] = df_all['filename'].map(method_names).fillna(df_all['filename'])
    
    return df_all

def create_interactive_target_count_plot(nk_data, lk_data, nk_gt_stats, lk_gt_stats, selected_methods):
    """Create interactive bar chart comparing number of predicted targets vs ground truth."""
    
    aspects = ['biological_process', 'molecular_function', 'cellular_component']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[ASPECT_NAMES[aspect] for aspect in aspects],
        shared_yaxes=True
    )
    
    colors = {"NK": "red", "LK": "blue"}

    y_max = max(*nk_gt_stats.values(), *lk_gt_stats.values())

    for i, aspect in enumerate(aspects):
        col = i + 1
        
        # Get ground truth counts
        nk_gt_count = nk_gt_stats.get(aspect, 0)
        lk_gt_count = lk_gt_stats.get(aspect, 0)
        
        # Filter data for current aspect
        nk_aspect = nk_data[nk_data['ns'] == aspect]
        lk_aspect = lk_data[lk_data['ns'] == aspect]
        
        # Add ground truth horizontal lines as scatter traces
        # Get x-axis range for the lines
        x_range = selected_methods  # Include all x-axis values
        
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[nk_gt_count] * len(x_range),
                mode='lines', line=dict(color=colors['NK'], dash='dash', width=2),
                name='Ground Truth: No Knowledge Proteins', showlegend=(i==1)
            ),
            row=1, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[lk_gt_count] * len(x_range), 
                mode='lines', line=dict(color=colors['LK'], dash='dash', width=2),
                name='Ground Truth: Limited Knowledge Proteins', showlegend=(i==1)
            ),
            row=1, col=col
        )
        
        # Add method prediction bars
        for j, method in enumerate(selected_methods):
            nk_count = nk_aspect[nk_aspect['method'] == method]['n'].iloc[0] if len(nk_aspect[nk_aspect['method'] == method]) > 0 else 0
            lk_count = lk_aspect[lk_aspect['method'] == method]['n'].iloc[0] if len(lk_aspect[lk_aspect['method'] == method]) > 0 else 0
            
            fig.add_trace(
                go.Bar(name=f'{method}', x=[method], y=[nk_count], 
                       marker_color=colors["NK"], opacity=1, 
                       showlegend=False),
                row=1, col=col
            )
            fig.add_trace(
                go.Bar(name=f'{method}', x=[method], y=[lk_count], 
                       marker_color=colors["LK"], opacity=1,
                       showlegend=False),
                row=1, col=col
            )
        fig.update_xaxes(
            row=1, col=col, 
            tickfont=dict(size=16), 
            title_font=dict(size=16),
            tickvals=x_range, 
            ticktext=selected_methods
        )
        
    fig.update_layout(
        height=400,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        legend_font=dict(size=16),
    )
    fig.update_yaxes(title_text="Number of Predicted Proteins", row=1, col=1, range=[0, y_max],
                     tickfont=dict(size=16),title_font=dict(size=18))
    
    
    return fig

def create_interactive_performance_plot(nk_data, lk_data, selected_methods, selected_metric):
    """Create interactive performance metrics plot with scatter points."""
    
    aspects = ['biological_process', 'molecular_function', 'cellular_component']
    aspect_symbols = ['circle', 'square', 'triangle-up']  # Different symbols for aspects
    aspect_labels = ['BPO', 'MFO', 'CCO']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{ASPECT_NAMES[aspect]}' for aspect in aspects],
        shared_yaxes=True
    )
    
    colors = px.colors.qualitative.Set1
    # Prepare data for plotting
    x_pos = np.arange(len(selected_methods))
    dodge_offset = 0.1  # offset for dodging points
    
    for i, aspect in enumerate(aspects):
        col = i + 1
        
        # Filter data for current aspect
        nk_aspect = nk_data[nk_data['ns'] == aspect]
        lk_aspect = lk_data[lk_data['ns'] == aspect]
        
        nk_values = []
        lk_values = []
        nk_x_positions = []
        lk_x_positions = []
        
        for j, method in enumerate(selected_methods):
            nk_val = nk_aspect[nk_aspect['method'] == method][selected_metric].iloc[0] if len(nk_aspect[nk_aspect['method'] == method]) > 0 else 0
            lk_val = lk_aspect[lk_aspect['method'] == method][selected_metric].iloc[0] if len(lk_aspect[lk_aspect['method'] == method]) > 0 else 0

            nk_values.append(nk_val)
            lk_values.append(lk_val)
            nk_x_positions.append(j - dodge_offset)
            lk_x_positions.append(j + dodge_offset)

        # Add NK scatter points
        fig.add_trace(
            go.Scatter(
                name=f'No Knowledge Proteins' if i == 0 else '',
                x=nk_x_positions, 
                y=nk_values,
                mode='markers', 
                marker=dict(
                    color='red', size=12, symbol=aspect_symbols[i],
                    line=dict(color='darkred', width=1)
                ),
                showlegend=(i == 0),
                legendgroup='NK',
                text=[f'{val:.3f}, {aspect_labels[i]}' for val in nk_values], 
                hoverinfo='text',
                textposition='top center'
            ),
            row=1, col=col
        )
        
        # Add LK scatter points
        fig.add_trace(
            go.Scatter(
                name=f'Limited Knowledge Proteins' if i == 0 else '',
                x=lk_x_positions, 
                y=lk_values,
                mode='markers',
                marker=dict(
                    color='blue', size=12, symbol=aspect_symbols[i],
                    line=dict(color='darkblue', width=1)
                ),
                showlegend=(i == 0),
                legendgroup='LK',
                text=[f'{val:.3f}, {aspect_labels[i]}' for val in lk_values], 
                hoverinfo='text',
                textposition='bottom center'
            ),
            row=1, col=col
        )
        
        fig.update_xaxes(
            row=1, col=col, 
            tickfont=dict(size=16), 
            title_font=dict(size=16),
            tickvals=x_pos, 
            ticktext=selected_methods
        )

    
    fig.update_layout(
        title_text=f"{selected_metric.upper()} Comparison (Best F-score threshold)",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        legend_font=dict(size=16)
    )
    fig.update_yaxes(title_text=selected_metric.upper(), row=1, col=1, range=[0, 1],
                     tickfont=dict(size=16), title_font=dict(size=18))
    
    return fig

def create_consolidated_performance_plot(nk_data, lk_data, selected_methods):
    """Create consolidated performance metrics plot showing precision, recall, and F-score."""
    
    metrics = ['pr', 'rc', 'f']
    metric_names = ['Precision', 'Recall', 'F-score']
    aspects = ['biological_process', 'molecular_function', 'cellular_component']
    aspect_symbols = ['circle', 'square', 'triangle-up']
    aspect_labels = ['BPO', 'MFO', 'CCO']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=metric_names,
        shared_yaxes=True
    )
    
    # Prepare data for plotting with dodging
    x_pos = np.arange(len(selected_methods))
    dodge_offset = 0.15  # offset for dodging points
    
    for i, metric in enumerate(metrics):
        col = i + 1
        
        for j, aspect in enumerate(aspects):
            # Filter data for current aspect
            nk_aspect = nk_data[nk_data['ns'] == aspect]
            lk_aspect = lk_data[lk_data['ns'] == aspect]
            
            # Collect data for this aspect
            nk_values = []
            lk_values = []
            nk_x_positions = []
            lk_x_positions = []
            
            for k, method in enumerate(selected_methods):
                nk_val = nk_aspect[nk_aspect['method'] == method][metric].iloc[0] if len(nk_aspect[nk_aspect['method'] == method]) > 0 else 0
                lk_val = lk_aspect[lk_aspect['method'] == method][metric].iloc[0] if len(lk_aspect[lk_aspect['method'] == method]) > 0 else 0
                
                nk_values.append(nk_val)
                lk_values.append(lk_val)
                nk_x_positions.append(k - dodge_offset)
                lk_x_positions.append(k + dodge_offset)
            
            # Add NK scatter points for this aspect
            fig.add_trace(
                go.Scatter(
                    name=f'No Knowledge Proteins' if i == 0 else '',
                    x=nk_x_positions,
                    y=nk_values,
                    mode='markers', 
                    marker=dict(
                        color='red', size=10, symbol=aspect_symbols[j],
                        line=dict(color='darkred', width=1)
                    ),
                    showlegend=(i == 0),
                    legendgroup='NK',
                    text=[f'{val:.3f}, {aspect_labels[j]}' for val in nk_values],
                    hoverinfo='text',
                    textposition='top center'
                ),
                row=1, col=col
            )
            
            # Add LK scatter points for this aspect
            fig.add_trace(
                go.Scatter(
                    name=f'Limited Knowledge Proteins' if i == 0 else '',
                    x=lk_x_positions,
                    y=lk_values,
                    mode='markers',
                    marker=dict(
                        color='blue', size=10, symbol=aspect_symbols[j],
                        line=dict(color='darkblue', width=1)
                    ),
                    showlegend=(i == 0),
                    legendgroup='LK',
                    text=[f'{val:.3f}, {aspect_labels[j]}' for val in lk_values], 
                    hoverinfo='text',
                    textposition='bottom center'
                ),
                row=1, col=col
            )
        
        # Update x-axis with proper tick positions and labels
        fig.update_xaxes(
            row=1, col=col, 
            tickfont=dict(size=16), 
            title_font=dict(size=16),
            tickvals=x_pos, 
            ticktext=selected_methods
        )
    
    fig.update_layout(
        title_text="Performance Metrics Comparison (Best F-score threshold)",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        legend_font=dict(size=16)
    )
    fig.update_yaxes(title_text="Score", row=1, col=1, range=[0, 1],
                     tickfont=dict(size=16), title_font=dict(size=18))
    
    return fig

def create_interactive_precision_recall_plot(all_data, selected_methods, subset_name, selected_aspect):
    """Create interactive precision-recall curves."""
    
    # Configuration for precision-recall curves
    metric = 'f'
    cols = ['rc', 'pr']
    cumulate = True
    add_extreme_points = True
    coverage_threshold = 0.01
    
    # Set method information
    all_data['group'] = all_data['method']
    all_data['label'] = all_data['method']
    
    df = all_data.drop(columns='filename').set_index(['group', 'label', 'ns', 'tau'])
    
    # Filter by coverage and aspect
    df = df[df['cov'] >= coverage_threshold]
    df = df[df.index.get_level_values('ns') == selected_aspect]
    
    # Filter by selected methods
    df = df[df.index.get_level_values('group').isin(selected_methods)]
    
    if df.empty:
        st.warning(f"No data available for selected methods in {selected_aspect}")
        return go.Figure()
    
    # Identify the best methods and thresholds
    index_best = df.groupby(level=['group', 'ns'])[metric].idxmax()
    
    # Filter the dataframe for the best methods
    df_methods = df.reset_index('tau').loc[[ele[:-1] for ele in index_best], ['tau', 'cov'] + cols + [metric]].sort_index()
    
    # Makes the curves monotonic
    if cumulate:
        df_methods[cols[-1]] = df_methods.groupby(level=['label', 'ns'])[cols[-1]].cummax()
    
    # Add extreme points
    def add_points(df_):
        df_ = pd.concat([df_.iloc[0:1], df_])
        df_.iloc[0, df_.columns.get_indexer(['tau', cols[0], cols[1]])] = [0, 1, 0]  # tau, rc, pr
        df_ = pd.concat([df_, df_.iloc[-1:]])
        df_.iloc[-1, df_.columns.get_indexer(['tau', cols[0], cols[1]])] = [1.1, 0, 1]
        return df_
    
    if add_extreme_points:
        df_methods = df_methods.reset_index().groupby(['group', 'label', 'ns'], as_index=False).apply(add_points).set_index(['group', 'label', 'ns'])
    
    # Filter the dataframe for the best method and threshold
    df_best = df.loc[index_best, ['cov'] + cols + [metric]]
    
    # Calculate average precision score 
    df_best['aps'] = df_methods.groupby(level=['group', 'label', 'ns'])[[cols[0], cols[1]]].apply(lambda x: (x[cols[0]].diff(-1).shift(1) * x[cols[1]]).sum())
    
    # Calculate the max coverage across all thresholds
    df_best['max_cov'] = df_methods.groupby(level=['group', 'label', 'ns'])['cov'].max()
    
    # Create the plot
    fig = go.Figure()
    
    # Add F-score contour lines
    x = np.arange(0.01, 1, 0.01)
    y = np.arange(0.01, 1, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = 2 * X * Y / (X + Y)
    
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        contours=dict(start=0.1, end=0.9, size=0.1),
        showscale=False,
        line=dict(color='gray', width=1),
        opacity=0.3,
        name="F-score contours"
    ))
    
    colors = px.colors.qualitative.Set1
    
    # Plot curves for each method
    for i, (method_idx, method_data) in enumerate(df_methods.groupby(level=['group', 'label'])):
        method_name = method_idx[0]
        method_data_single = method_data.droplevel(['group', 'label'])
        
        # Get best point
        best_point = df_best.loc[method_idx + (selected_aspect,)]
        
        # Plot precision-recall curve
        fig.add_trace(go.Scatter(
            x=method_data_single[cols[0]], 
            y=method_data_single[cols[1]],
            mode='lines',
            name=f"{method_name} (F={best_point[metric].values[0]}, APS={best_point['aps'].values[0]})",
            line=dict(color=colors[i % len(colors)], width=3)
        ))
        
        # Add best point marker
        fig.add_trace(go.Scatter(
            x=[best_point[cols[0]]], 
            y=[best_point[cols[1]]],
            mode='markers',
            marker=dict(color=colors[i % len(colors)], size=10, symbol='circle'),
            showlegend=False,
            name=f"{method_name} F-max"
        ))
    
    fig.update_layout(
        title=f"Precision-Recall Curves - {ONTOLOGY_DICT.get(selected_aspect, selected_aspect)} ({subset_name})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=600,
        width=800
    )
    
    return fig

def main():
    st.title("CAFA Forever")
    st.markdown("Continuous Critical Assessment of Functional Annotation.")
    
    # Get available timepoints
    available_timepoints = get_available_timepoints()
    
    if not available_timepoints:
        st.error("No valid timepoint directories found! Please ensure you have directories with results_NK/ and results_LK/ subdirectories.")
        return
    
    # Sidebar for configuration
    st.sidebar.header("📁 Data Configuration")
    
    # Timepoint selection
    if len(available_timepoints) > 1:
        selected_timepoint = st.sidebar.selectbox(
            "Select Timepoint",
            options=available_timepoints,
            help="Choose which evaluation timepoint to analyze"
        )
    else:
        selected_timepoint = available_timepoints[0]
        st.sidebar.info(f"Using timepoint: {selected_timepoint}")
    
    # Set up paths based on selected timepoint
    results_dir = Path(selected_timepoint)
    NK_RESULTS_DIR = results_dir / "results_NK"
    LK_RESULTS_DIR = results_dir / "results_LK"
    
    # Auto-detect ground truth files
    gt_files_nk = list(results_dir.glob("groundtruth_*_NK.tsv"))
    gt_files_lk = list(results_dir.glob("groundtruth_*_LK.tsv"))
    
    if not gt_files_nk or not gt_files_lk:
        st.error(f"Ground truth files not found in {results_dir}")
        return
        
    GROUND_TRUTH_NK = gt_files_nk[0]  # Use first match
    GROUND_TRUTH_LK = gt_files_lk[0]  # Use first match
    
    method_names_file = results_dir / "method_names.tsv"
    
    # Load data
    if st.sidebar.button("Load Data") or 'data_loaded' not in st.session_state or st.session_state.get('current_timepoint') != selected_timepoint:
        with st.spinner("Loading data..."):
            try:
                # Load method names mapping
                method_names = load_method_names(method_names_file if method_names_file.exists() else None)
                
                # Load ground truth statistics
                nk_gt_stats = load_ground_truth_stats(GROUND_TRUTH_NK)
                lk_gt_stats = load_ground_truth_stats(GROUND_TRUTH_LK)
                
                # Load evaluation data
                nk_data = load_evaluation_data(NK_RESULTS_DIR, 'NK', method_names)
                lk_data = load_evaluation_data(LK_RESULTS_DIR, 'LK', method_names)
                
                # Load all evaluation data for precision-recall curves
                nk_all_data = load_all_evaluation_data(NK_RESULTS_DIR, method_names)
                lk_all_data = load_all_evaluation_data(LK_RESULTS_DIR, method_names)
                
                # Store in session state
                st.session_state.update({
                    'data_loaded': True,
                    'current_timepoint': selected_timepoint,
                    'method_names': method_names,
                    'nk_gt_stats': nk_gt_stats,
                    'lk_gt_stats': lk_gt_stats,
                    'nk_data': nk_data,
                    'lk_data': lk_data,
                    'nk_all_data': nk_all_data,
                    'lk_all_data': lk_all_data
                })
                
                st.success("Data loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
    
    if 'data_loaded' not in st.session_state:
        st.warning("Please configure data paths and click 'Load Data' to begin.")
        return
    
    # Get available methods
    all_methods = list(set(
        st.session_state.nk_data['method'].tolist() + 
        st.session_state.lk_data['method'].tolist()
    ))
    
    # Sidebar for method selection
    st.sidebar.header("Method Selection")
    selected_methods = st.sidebar.multiselect(
        "Select methods to compare:",
        options=all_methods,
        default=all_methods[:3] if len(all_methods) >= 3 else all_methods,
        help="Choose which prediction methods to include in the comparison"
    )
    
    if not selected_methods:
        st.warning("Please select at least one method to compare.")
        return
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Precision-Recall", "Summary Table"])
    
    with tab1:
        st.header("Number of Targets Predicted vs Ground Truth")
        st.markdown("Compare the number of proteins each method made predictions for versus the total available in ground truth.")

        fig = create_interactive_target_count_plot(
            st.session_state.nk_data, 
            st.session_state.lk_data, 
            st.session_state.nk_gt_stats, 
            st.session_state.lk_gt_stats, 
            selected_methods
        )
        st.plotly_chart(fig, use_container_width=True)
    
        st.header("Performance Metrics Comparison")
        st.markdown("Compare the performance of methods with precision, recall, and F-score metrics.")
        plot_type = st.radio(
            "Select plot type:",
            options=['consolidated', 'individual'],
            format_func=lambda x: {
                'consolidated': 'All Metrics (Precision, Recall, F-score)', 
                'individual': 'Individual Metric'
            }[x],
            index=0,
            horizontal=True
        )
        
        if plot_type == 'consolidated':
            fig = create_consolidated_performance_plot(
                st.session_state.nk_data, 
                st.session_state.lk_data, 
                selected_methods
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            selected_metric = st.selectbox(
                "Select metric:",
                options=['pr', 'rc', 'f', 'cov'],
                format_func=lambda x: {'pr': 'Precision', 'rc': 'Recall', 'f': 'F-score', 'cov': 'Coverage'}[x],
                index=2
            )
            
            fig = create_interactive_performance_plot(
                st.session_state.nk_data, 
                st.session_state.lk_data, 
                selected_methods, 
                selected_metric
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Precision-Recall Curves")
        st.markdown("Interactive precision-recall curves showing the trade-off between precision and recall across different thresholds.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_subset = st.selectbox(
                "Select protein subset:",
                options=['NK', 'LK'],
                format_func=lambda x: f"{x} (No Knowledge)" if x == 'NK' else f"{x} (Limited Knowledge)"
            )
        
        with col2:
            selected_aspect = st.selectbox(
                "Select GO aspect:",
                options=['biological_process', 'molecular_function', 'cellular_component'],
                format_func=lambda x: ASPECT_NAMES[x]
            )
        
        all_data = st.session_state.nk_all_data if selected_subset == 'NK' else st.session_state.lk_all_data
        
        fig = create_interactive_precision_recall_plot(
            all_data, 
            selected_methods, 
            selected_subset, 
            selected_aspect
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Summary Table")
        st.markdown("Detailed performance metrics for all selected methods across subsets and aspects.")
        
        # Create summary data
        aspects = ['biological_process', 'molecular_function', 'cellular_component']
        summary_data = []
        
        for subset, data in [('NK', st.session_state.nk_data), ('LK', st.session_state.lk_data)]:
            for aspect in aspects:
                aspect_data = data[data['ns'] == aspect]
                
                for method in selected_methods:
                    method_data = aspect_data[aspect_data['method'] == method]
                    
                    if len(method_data) > 0:
                        row = method_data.iloc[0]
                        summary_data.append({
                            'Subset': subset,
                            'Aspect': ASPECT_NAMES[aspect],
                            'Method': method,
                            'Targets_Predicted': int(row['n']),
                            'Precision': f"{row['pr']:.3f}",
                            'Recall': f"{row['rc']:.3f}",
                            'F-score': f"{row['f']:.3f}",
                            'Coverage': f"{row['cov']:.3f}",
                            'Threshold': f"{row['tau']:.3f}"
                        })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Download button
        csv = df_summary.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=csv,
            file_name="cafa6_summary_metrics.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

