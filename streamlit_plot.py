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
from config import get_available_timepoints, STREAMLIT_CONFIG, GO_ASPECTS, DATA_DATES

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
        if 'filename' in df_methods.columns and 'label' in df_methods.columns:
            return dict(zip(df_methods['filename'], df_methods['label']))
        else:
            st.warning("method_names file should have 'filename' and 'label' columns")
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

def create_interactive_target_count_plot(nk_data, lk_data, nk_gt_stats, lk_gt_stats, pk_gt_stats, selected_methods):
    """Create interactive bar chart comparing number of predicted targets vs ground truth."""
    
    aspects = ['biological_process', 'molecular_function', 'cellular_component']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[ASPECT_NAMES[aspect] for aspect in aspects],
        shared_yaxes=True
    )
    
    colors = {"NK": "red", "LK": "blue", "PK": "green"}

    y_max = max(*nk_gt_stats.values(), *lk_gt_stats.values())

    for i, aspect in enumerate(aspects):
        col = i + 1
        
        # Get ground truth counts
        nk_gt_count = nk_gt_stats.get(aspect, 0)
        lk_gt_count = lk_gt_stats.get(aspect, 0)
        pk_gt_count = pk_gt_stats.get(aspect, 0)

        # Filter data for current aspect
        nk_aspect = nk_data[nk_data['ns'] == aspect]
        lk_aspect = lk_data[lk_data['ns'] == aspect]
        pk_aspect = pk_data[pk_data['ns'] == aspect]

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
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[pk_gt_count] * len(x_range), 
                mode='lines', line=dict(color=colors['PK'], dash='dash', width=2),
                name='Ground Truth: Partial Knowledge Proteins', showlegend=(i==1)
            ),
            row=1, col=col
        )
        # Add method prediction bars
        for j, method in enumerate(selected_methods):
            nk_count = nk_aspect[nk_aspect['method'] == method]['n'].iloc[0] if len(nk_aspect[nk_aspect['method'] == method]) > 0 else 0
            lk_count = lk_aspect[lk_aspect['method'] == method]['n'].iloc[0] if len(lk_aspect[lk_aspect['method'] == method]) > 0 else 0
            pk_count = pk_aspect[pk_aspect['method'] == method]['n'].iloc[0] if len(pk_aspect[pk_aspect['method'] == method]) > 0 else 0

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
            fig.add_trace(
                go.Bar(name=f'{method}', x=[method], y=[pk_count], 
                       marker_color=colors["PK"], opacity=1,
                       showlegend=False),
                row=1, col=col
            )
        fig.update_xaxes(
            row=1, col=col, 
            tickfont=dict(size=16, color='black'), 
            title_font=dict(size=16, color='black'),
            tickvals=x_range, 
            ticktext=selected_methods,
            linecolor='black'
        )
        
    fig.update_layout(
        height=350,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.5,
            xanchor="center",
            x=0.5,
        ),
        legend_font=dict(size=16),
    )
    fig.update_yaxes(title_text="Number of Predicted Proteins", row=1, col=1, range=[0, y_max],
                     tickfont=dict(size=16, color='black'),title_font=dict(size=18, color='black'),
                     linecolor='black')
    
    
    return fig

def create_interactive_performance_plot(nk_data, lk_data, pk_data, selected_methods, selected_metric):
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
    dodge_offset = 0.15  # offset for dodging points
    
    # Add legend traces for knowledge types (only once)
    fig.add_trace(
        go.Scatter(
            name='No Knowledge Proteins',
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='red', size=12, symbol='circle'),
            showlegend=True,
            legendgroup='NK_legend'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            name='Limited Knowledge Proteins',
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='blue', size=12, symbol='circle'),
            showlegend=True,
            legendgroup='LK_legend'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            name='Partial Knowledge Proteins',
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='green', size=12, symbol='circle'),
            showlegend=True,
            legendgroup='PK_legend'
        ),
        row=1, col=1
    )

    # Add legend traces for aspect symbols (only once)
    for i, (aspect, symbol, label) in enumerate(zip(aspects, aspect_symbols, aspect_labels)):
        fig.add_trace(
            go.Scatter(
                name=f'{label} ({ASPECT_NAMES[aspect]})',
                x=[None], y=[None],
                mode='markers',
                marker=dict(color='black', size=12, symbol=symbol),
                showlegend=True,
                legendgroup=f'aspect_{i}'
            ),
            row=1, col=1
        )
    
    for i, aspect in enumerate(aspects):
        col = i + 1
        
        # Filter data for current aspect
        nk_aspect = nk_data[nk_data['ns'] == aspect]
        lk_aspect = lk_data[lk_data['ns'] == aspect]
        pk_aspect = pk_data[pk_data['ns'] == aspect]
        
        nk_values = []
        lk_values = []
        pk_values = []
        nk_x_positions = []
        lk_x_positions = []
        pk_x_positions = []
        
        for j, method in enumerate(selected_methods):
            nk_val = nk_aspect[nk_aspect['method'] == method][selected_metric].iloc[0] if len(nk_aspect[nk_aspect['method'] == method]) > 0 else 0
            lk_val = lk_aspect[lk_aspect['method'] == method][selected_metric].iloc[0] if len(lk_aspect[lk_aspect['method'] == method]) > 0 else 0
            pk_val = pk_aspect[pk_aspect['method'] == method][selected_metric].iloc[0] if len(pk_aspect[pk_aspect['method'] == method]) > 0 else 0

            nk_values.append(nk_val)
            lk_values.append(lk_val)
            pk_values.append(pk_val)
            nk_x_positions.append(j - dodge_offset)
            lk_x_positions.append(j)
            pk_x_positions.append(j + dodge_offset)

        # Add NK scatter points (no legend)
        fig.add_trace(
            go.Scatter(
                name='',
                x=nk_x_positions, 
                y=nk_values,
                mode='markers', 
                marker=dict(
                    color='red', size=12, symbol=aspect_symbols[i],
                    line=dict(color='darkred', width=1)
                ),
                showlegend=False,
                legendgroup='NK',
                text=[f'{val:.3f}, {aspect_labels[i]}' for val in nk_values], 
                hoverinfo='text',
                textposition='top center'
            ),
            row=1, col=col
        )
        
        # Add LK scatter points (no legend)
        fig.add_trace(
            go.Scatter(
                name='',
                x=lk_x_positions, 
                y=lk_values,
                mode='markers',
                marker=dict(
                    color='blue', size=12, symbol=aspect_symbols[i],
                    line=dict(color='darkblue', width=1)
                ),
                showlegend=False,
                legendgroup='LK',
                text=[f'{val:.3f}, {aspect_labels[i]}' for val in lk_values], 
                hoverinfo='text',
                textposition='bottom center'
            ),
            row=1, col=col
        )
        # Add PK scatter points (no legend)
        fig.add_trace(
            go.Scatter(
                name='',
                x=pk_x_positions, 
                y=pk_values,
                mode='markers',
                marker=dict(
                    color='green', size=12, symbol=aspect_symbols[i],
                    line=dict(color='darkgreen', width=1)
                ),
                showlegend=False,
                legendgroup='PK',
                text=[f'{val:.3f}, {aspect_labels[i]}' for val in pk_values], 
                hoverinfo='text',
                textposition='middle center'
            ),
            row=1, col=col
        )
        
        fig.update_xaxes(
            row=1, col=col, 
            tickfont=dict(size=16, color='black'), 
            title_font=dict(size=16, color='black'),
            tickvals=x_pos, 
            ticktext=selected_methods,
            linecolor='black'
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
                     tickfont=dict(size=16, color='black'), title_font=dict(size=18, color='black'),
                     linecolor='black')
    
    return fig

def create_consolidated_performance_plot(nk_data, lk_data, pk_data, selected_methods):
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
    
    # Add legend traces for knowledge types (only once)
    fig.add_trace(
        go.Scatter(
            name='No Knowledge Proteins',
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='red', size=12, symbol='circle'),
            showlegend=True,
            legendgroup='NK_legend',
            legendrank=1
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            name='Limited Knowledge Proteins',
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='blue', size=12, symbol='circle'),
            showlegend=True,
            legendgroup='LK_legend',
            legendrank=1
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            name='Partial Knowledge Proteins',
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='green', size=12, symbol='circle'),
            showlegend=True,
            legendgroup='PK_legend',
            legendrank=1
        ),
        row=1, col=1
    )
    
    # Add legend traces for aspect symbols (only once)
    for j, (aspect, symbol, label) in enumerate(zip(aspects, aspect_symbols, aspect_labels)):
        fig.add_trace(
            go.Scatter(
                name=f'{label} ({ASPECT_NAMES[aspect]})',
                x=[None], y=[None],
                mode='markers',
                marker=dict(color='black', size=12, symbol=symbol),
                showlegend=True,
                legendgroup=f'aspect_{j}',
                legendrank=2
            ),
            row=1, col=1
        )
    
    for i, metric in enumerate(metrics):
        col = i + 1
        
        for j, aspect in enumerate(aspects):
            # Filter data for current aspect
            nk_aspect = nk_data[nk_data['ns'] == aspect]
            lk_aspect = lk_data[lk_data['ns'] == aspect]
            pk_aspect = pk_data[pk_data['ns'] == aspect]
            
            # Collect data for this aspect
            nk_values = []
            lk_values = []
            pk_values = []
            nk_x_positions = []
            lk_x_positions = []
            pk_x_positions = []

            for k, method in enumerate(selected_methods):
                nk_val = nk_aspect[nk_aspect['method'] == method][metric].iloc[0] if len(nk_aspect[nk_aspect['method'] == method]) > 0 else 0
                lk_val = lk_aspect[lk_aspect['method'] == method][metric].iloc[0] if len(lk_aspect[lk_aspect['method'] == method]) > 0 else 0
                pk_val = pk_aspect[pk_aspect['method'] == method][metric].iloc[0] if len(pk_aspect[pk_aspect['method'] == method]) > 0 else 0

                nk_values.append(nk_val)
                lk_values.append(lk_val)
                pk_values.append(pk_val)
                nk_x_positions.append(k - dodge_offset)
                lk_x_positions.append(k)
                pk_x_positions.append(k + dodge_offset)

            # Add NK scatter points for this aspect (no legend)
            fig.add_trace(
                go.Scatter(
                    name='',
                    x=nk_x_positions,
                    y=nk_values,
                    mode='markers', 
                    marker=dict(
                        color='red', size=10, symbol=aspect_symbols[j],
                        line=dict(color='darkred', width=1)
                    ),
                    showlegend=False,
                    legendgroup='NK',
                    text=[f'{val:.3f}, {aspect_labels[j]}' for val in nk_values],
                    hoverinfo='text',
                    textposition='top center'
                ),
                row=1, col=col
            )
            
            # Add LK scatter points for this aspect (no legend)
            fig.add_trace(
                go.Scatter(
                    name='',
                    x=lk_x_positions,
                    y=lk_values,
                    mode='markers',
                    marker=dict(
                        color='blue', size=10, symbol=aspect_symbols[j],
                        line=dict(color='darkblue', width=1)
                    ),
                    showlegend=False,
                    legendgroup='LK',
                    text=[f'{val:.3f}, {aspect_labels[j]}' for val in lk_values], 
                    hoverinfo='text',
                    textposition='bottom center'
                ),
                row=1, col=col
            )
            
            # Add PK scatter points for this aspect (no legend)
            fig.add_trace(
                go.Scatter(
                    name='',
                    x=pk_x_positions,
                    y=pk_values,
                    mode='markers',
                    marker=dict(
                        color='green', size=10, symbol=aspect_symbols[j],
                        line=dict(color='darkgreen', width=1)
                    ),
                    showlegend=False,
                    legendgroup='PK',
                    text=[f'{val:.3f}, {aspect_labels[j]}' for val in pk_values], 
                    hoverinfo='text',
                    textposition='middle center'
                ),
                row=1, col=col
            )
        
        # Update x-axis with proper tick positions and labels
        fig.update_xaxes(
            row=1, col=col, 
            tickfont=dict(size=16, color='black'), 
            title_font=dict(size=16, color='black'),
            tickvals=x_pos, 
            ticktext=selected_methods,
            linecolor='black'
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
                     tickfont=dict(size=16, color='black'), title_font=dict(size=18, color='black'),
                     linecolor='black')
    
    return fig

def create_interactive_precision_recall_plot(nk_all_data, lk_all_data, pk_all_data, selected_methods):
    """Create interactive precision-recall curves for all subsets and aspects."""
    
    # Configuration for precision-recall curves
    metric = 'f'
    cols = ['rc', 'pr']
    cumulate = True
    add_extreme_points = False
    coverage_threshold = 0.01
    
    aspects = ['biological_process', 'molecular_function', 'cellular_component']
    subsets = [('NK', nk_all_data), ('LK', lk_all_data), ('PK', pk_all_data)]

    # Create subplots: 3 rows (NK, LK, PK) x 3 cols (aspects)
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f"{ONTOLOGY_DICT.get(aspect, aspect)} - No Knowledge" for aspect in aspects] + 
                      [f"{ONTOLOGY_DICT.get(aspect, aspect)} - Limited Knowledge" for aspect in aspects] +
                      [f"{ONTOLOGY_DICT.get(aspect, aspect)} - Partial Knowledge" for aspect in aspects],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    fig.update_annotations(font_size=16, font_color='black')
    colors = px.colors.qualitative.Set1
    
    # F-score contour lines (same for all subplots)
    x = np.arange(0.01, 1, 0.01)
    y = np.arange(0.01, 1, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = 2 * X * Y / (X + Y)
    
    for subset_idx, (subset_name, all_data) in enumerate(subsets):
        for aspect_idx, selected_aspect in enumerate(aspects):
            row = subset_idx + 1
            col = aspect_idx + 1
            
            # Set method information
            all_data_copy = all_data.copy()
            all_data_copy['group'] = all_data_copy['method']
            all_data_copy['label'] = all_data_copy['method']
            
            df = all_data_copy.drop(columns='filename').set_index(['group', 'label', 'ns', 'tau'])
            
            # Filter by coverage and aspect
            df = df[df['cov'] >= coverage_threshold]
            df = df[df.index.get_level_values('ns') == selected_aspect]
            
            # Filter by selected methods
            df = df[df.index.get_level_values('group').isin(selected_methods)]
            
            if df.empty:
                continue
            
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
            df_best['aps'] = df_best['aps'].apply(round, ndigits=3)

            # Calculate the max coverage across all thresholds
            df_best['max_cov'] = df_methods.groupby(level=['group', 'label', 'ns'])['cov'].max()
            df_best['max_cov'] = df_best['max_cov'].apply(round, ndigits=4)
            
            # Add colors and labels to df_best for sorting (like in matplotlib version)
            df_best['colors'] = [colors[i % len(colors)] for i in range(len(df_best))]
            df_best['label'] = df_best.index.get_level_values('group')
            
            # Add F-score contour lines with labels (equivalent to matplotlib contour + clabel)
            contour_levels = np.arange(0.1, 1.0, 0.1)
            fig.add_trace(go.Contour(
                x=x, y=y, z=Z,
                contours=dict(
                    start=0.1, 
                    end=0.9, 
                    size=0.1,
                    showlabels=True,
                    labelfont=dict(size=16, color='gray')
                ),
                showscale=False,
                line=dict(color='gray', width=1),
                colorscale=[[0, 'rgba(128,128,128,0)'], [1, 'rgba(128,128,128,0)']],
                name="F-score contours",
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)
            
            # Sort methods by metric and max_cov (matching matplotlib implementation)
            df_best_sorted = df_best.sort_values(by=[metric, 'max_cov'], ascending=[False, False])
            
            # Create legend text for this subplot
            legend_text = []
            
            # Plot curves for each method in sorted order
            for i, (index, row_data) in enumerate(df_best_sorted.iterrows()):
                method_name = index[0]  # group name
                method_data = df_methods.loc[index[:-1]]  # Remove ns from index to match df_methods
                
                # Add to legend text with F-score and coverage
                legend_text.append(f"{method_name} (F={row_data[metric]:.3f}, Cov={row_data['max_cov']:.4f})")

                # Plot precision-recall curve with thick line (lw=2 -> width=4 for plotly equivalent)
                fig.add_trace(go.Scatter(
                    x=method_data[cols[0]], 
                    y=method_data[cols[1]],
                    mode='lines',
                    name=f"{method_name}",
                    line=dict(color=row_data['colors'], width=2),
                    showlegend=False,  # Turn off global legend
                    legendgroup=f"method_{method_name}_{row}_{col}",
                    hovertemplate=f"<b>{method_name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
                ), row=row, col=col)
                
                # Add F-max dots (two markers for hollow + filled effect)
                # Outer hollow marker (equivalent to mfc='none')
                fig.add_trace(go.Scatter(
                    x=[row_data[cols[0]]], 
                    y=[row_data[cols[1]]],
                    mode='markers',
                    marker=dict(
                        color='rgba(0,0,0,0)',  # transparent fill
                        size=12,  # markersize=12 -> size=24 for plotly
                        symbol='circle',
                        line=dict(color=row_data['colors'], width=3)
                    ),
                    showlegend=False,
                    name=f"{method_name} F-max outer",
                    legendgroup=f"method_{method_name}_{row}_{col}",
                    hovertemplate=f"<b>{method_name} F-max</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<br>F-score: {row_data[metric]:.3f}<extra></extra>"
                ), row=row, col=col)
                
                # Inner filled marker
                fig.add_trace(go.Scatter(
                    x=[row_data[cols[0]]], 
                    y=[row_data[cols[1]]],
                    mode='markers',
                    marker=dict(
                        color=row_data['colors'], 
                        size=6,  # markersize=6 -> size=12 for plotly
                        symbol='circle'
                    ),
                    showlegend=False,
                    name=f"{method_name} F-max inner",
                    legendgroup=f"method_{method_name}_{row}_{col}",
                    hoverinfo='skip'
                ), row=row, col=col)
            
            # Add custom legend as annotation for each subplot
            if legend_text:
                # Create colored legend with method colors
                legend_html = "<br>".join([
                    f"<span style='color:{df_best_sorted.iloc[i]['colors']};'>●</span> {text}" 
                    for i, text in enumerate(legend_text)
                ])
                
                # Calculate subplot reference
                if row == 1 and col == 1:
                    xref, yref = "x", "y"
                elif row == 1:
                    xref, yref = f"x{col}", f"y{col}"
                elif col == 1:
                    xref, yref = f"x{3 + col}", f"y{3 + col}"
                else:
                    xref, yref = f"x{3 + col}", f"y{3 + col}"
                
                fig.add_annotation(
                    text=legend_html,
                    xref=xref,
                    yref=yref,
                    x=1,  # Top right corner
                    y=1,
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=4
                )
    
    # Update layout with larger font sizes (matching rcParams font.size=22)
    fig.update_layout(
        height=1000,
        width=1500,
        showlegend=False  # Turn off global legend since we use subplot-specific legends
    )
    
    # Update all x-axes with larger fonts
    for row in range(1, 4):
        for col in range(1, 4):
            fig.update_xaxes(
                title_text="Recall" if row == 2 else "",
                title_font=dict(size=22, color='black'),  # matching rcParams font.size=22
                range=[0, 1],
                tickfont=dict(size=18, color='black'),
                linecolor='black',
                row=row, col=col
            )
    
    # Update all y-axes with larger fonts  
    for row in range(1, 4):
        for col in range(1, 4):
            fig.update_yaxes(
                title_text="Precision" if col == 1 else "",
                title_font=dict(size=22, color='black'),  # matching rcParams font.size=22
                range=[0, 1],
                tickfont=dict(size=18, color='black'),
                linecolor='black',
                row=row, col=col
            )
    
    return fig

def main():
    st.title("LAFA")
    st.markdown("Longitudinal Assessment of Functional Annotation.")
    
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
    PK_RESULTS_DIR = results_dir / "results_PK"

    # Auto-detect ground truth files
    gt_files_nk = list(results_dir.glob("groundtruth_*_NK.tsv"))
    gt_files_lk = list(results_dir.glob("groundtruth_*_LK.tsv"))
    gt_files_pk = list(results_dir.glob("groundtruth_*_PK.tsv"))

    if not gt_files_nk or not gt_files_lk or not gt_files_pk:
        st.error(f"Ground truth files not found in {results_dir}")
        return
        
    GROUND_TRUTH_NK = gt_files_nk[0]  # Use first match
    GROUND_TRUTH_LK = gt_files_lk[0]  # Use first match
    GROUND_TRUTH_PK = gt_files_pk[0]  # Use first match
    
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
                pk_gt_stats = load_ground_truth_stats(GROUND_TRUTH_PK)

                # Load evaluation data
                nk_data = load_evaluation_data(NK_RESULTS_DIR, 'NK', method_names)
                lk_data = load_evaluation_data(LK_RESULTS_DIR, 'LK', method_names)
                pk_data = load_evaluation_data(PK_RESULTS_DIR, 'PK', method_names)

                # Load all evaluation data for precision-recall curves
                nk_all_data = load_all_evaluation_data(NK_RESULTS_DIR, method_names)
                lk_all_data = load_all_evaluation_data(LK_RESULTS_DIR, method_names)
                pk_all_data = load_all_evaluation_data(PK_RESULTS_DIR, method_names)

                # Store in session state
                st.session_state.update({
                    'data_loaded': True,
                    'current_timepoint': selected_timepoint,
                    'method_names': method_names,
                    'nk_gt_stats': nk_gt_stats,
                    'lk_gt_stats': lk_gt_stats,
                    'pk_gt_stats': pk_gt_stats,
                    'nk_data': nk_data,
                    'lk_data': lk_data,
                    'pk_data': pk_data,
                    'nk_all_data': nk_all_data,
                    'lk_all_data': lk_all_data,
                    'pk_all_data': pk_all_data
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
        default=all_methods[:4] if len(all_methods) >= 4 else all_methods,
        help="Choose which prediction methods to include in the comparison"
    )
    
    if not selected_methods:
        st.warning("Please select at least one method to compare.")
        return
    
    
    # Display versions and target count plot
    col1, col2 = st.columns([0.2,0.8])
    
    with col1: 
        st.header("Data Versions")
        st.markdown(f"GO version: {DATA_DATES[selected_timepoint].get('go_start', 'N/A')}")
        st.markdown(f"UniProt version (for predictions): {DATA_DATES[selected_timepoint].get('uniprot_start', 'N/A')}")
        st.markdown(f"UniProt version (for ground truth): {DATA_DATES[selected_timepoint].get('uniprot_end', 'N/A')}")
    with col2:
        st.header("Number of Targets")
        st.markdown("Compare the number of proteins each method made predictions for versus the total available in ground truth.")

        # Create target count table
        aspects = ['biological_process', 'molecular_function', 'cellular_component']
        
        # Prepare table data
        table_data = []
        gt_data = {
            'Method': "ALL TARGETS"}  
        
        for method in selected_methods:
            row_data = {'Method': method}
            
            for aspect in aspects:
                aspect_name = ASPECT_NAMES[aspect]
                
                # Get ground truth counts
                nk_gt_count = st.session_state.nk_gt_stats.get(aspect, 0)
                lk_gt_count = st.session_state.lk_gt_stats.get(aspect, 0)
                pk_gt_count = st.session_state.pk_gt_stats.get(aspect, 0)

                # Get predicted counts for this method and aspect
                nk_aspect = st.session_state.nk_data[st.session_state.nk_data['ns'] == aspect]
                lk_aspect = st.session_state.lk_data[st.session_state.lk_data['ns'] == aspect]
                pk_aspect = st.session_state.pk_data[st.session_state.pk_data['ns'] == aspect]

                nk_pred = nk_aspect[nk_aspect['method'] == method]['n'].iloc[0] if len(nk_aspect[nk_aspect['method'] == method]) > 0 else 0
                lk_pred = lk_aspect[lk_aspect['method'] == method]['n'].iloc[0] if len(lk_aspect[lk_aspect['method'] == method]) > 0 else 0
                pk_pred = pk_aspect[pk_aspect['method'] == method]['n'].iloc[0] if len(pk_aspect[pk_aspect['method'] == method]) > 0 else 0

                # Calculate percentages
                nk_pct = (nk_pred / nk_gt_count * 100) if nk_gt_count > 0 else 0
                lk_pct = (lk_pred / lk_gt_count * 100) if lk_gt_count > 0 else 0
                pk_pct = (pk_pred / pk_gt_count * 100) if pk_gt_count > 0 else 0

                # Add columns for this aspect with nested structure
                gt_data[f'{aspect_name}_No Knowledge'] = f"{nk_gt_count}"
                gt_data[f'{aspect_name}_Limited Knowledge'] = f"{lk_gt_count}"
                gt_data[f'{aspect_name}_Partial Knowledge'] = f"{pk_gt_count}"
                row_data[f'{aspect_name}_No Knowledge'] = f"{int(nk_pred)} ({nk_pct:.1f}%)"
                row_data[f'{aspect_name}_Limited Knowledge'] = f"{int(lk_pred)} ({lk_pct:.1f}%)"
                row_data[f'{aspect_name}_Partial Knowledge'] = f"{int(pk_pred)} ({pk_pct:.1f}%)"

            table_data.append(row_data)
        table_data.append(gt_data)
        
        # Create DataFrame and display
        df_targets = pd.DataFrame(table_data)
        df_targets = df_targets.sort_values(by='Method', ascending=True)
        
        # Create nested column structure for display
        # First, reorder columns to group by aspect
        column_order = ['Method']
        for aspect in aspects:
            aspect_name = ASPECT_NAMES[aspect]
            column_order.extend([f'{aspect_name}_No Knowledge', f'{aspect_name}_Limited Knowledge', f'{aspect_name}_Partial Knowledge'])

        df_targets = df_targets[column_order]
        
        
        # Create HTML table with nested headers in a scrollable container
        html_table = """
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px;">
            <table style="width:100%; border-collapse: collapse; margin: 0;">
                <thead style="position: sticky; top: 0; z-index: 10;">
                    <tr style="background-color: #f0f2f6;">
                        <th rowspan="2" style="border: 1px solid #ddd; padding: 4px; text-align: center; vertical-align: middle; background-color: #f0f2f6;">Method</th>
        """
        
        # Add top-level headers (aspects)
        for aspect in aspects:
            aspect_name = ASPECT_NAMES[aspect]
            html_table += f'<th colspan="3" style="border: 1px solid #ddd; padding: 4px; text-align: center; background-color: #e1e5e9;">{aspect_name}</th>'
        
        html_table += "</tr><tr style='background-color: #f0f2f6;'>"
        
        # Add second-level headers (No Knowledge / Limited Knowledge)
        for aspect in aspects:
            html_table += '<th style="border: 1px solid #ddd; padding: 4px; text-align: center; background-color: #f0f2f6;">No Knowledge</th>'
            html_table += '<th style="border: 1px solid #ddd; padding: 4px; text-align: center; background-color: #f0f2f6;">Limited Knowledge</th>'
            html_table += '<th style="border: 1px solid #ddd; padding: 4px; text-align: center; background-color: #f0f2f6;">Partial Knowledge</th>'

        html_table += "</tr></thead><tbody>"
        
        # Add data rows
        for _, row in df_targets.iterrows():
            html_table += "<tr>"
            html_table += f'<td style="border: 1px solid #ddd; padding: 4px; font-weight: {"bold" if "ALL TARGETS" in str(row["Method"]) else "normal"};">{row["Method"]}</td>'
            
            for aspect in aspects:
                aspect_name = ASPECT_NAMES[aspect]
                nk_value = row[f'{aspect_name}_No Knowledge']
                lk_value = row[f'{aspect_name}_Limited Knowledge']
                pk_value = row[f'{aspect_name}_Partial Knowledge']
                
                html_table += f'<td style="border: 1px solid #ddd; padding: 4px; text-align: center;">{nk_value}</td>'
                html_table += f'<td style="border: 1px solid #ddd; padding: 4px; text-align: center;">{lk_value}</td>'
                html_table += f'<td style="border: 1px solid #ddd; padding: 4px; text-align: center;">{pk_value}</td>'
            
            html_table += "</tr>"
        
        html_table += "</tbody></table></div>"
        
        st.markdown(html_table, unsafe_allow_html=True)

        # Comment out the original plot
        # fig = create_interactive_target_count_plot(
        #     st.session_state.nk_data, 
        #     st.session_state.lk_data, 
        #     st.session_state.nk_gt_stats, 
        #     st.session_state.lk_gt_stats, 
        #     selected_methods
        # )
        # st.plotly_chart(fig, use_container_width=True)
        
        
    # Main content tabs
    tab1, tab2 = st.tabs(["Performance Metrics", "Summary Table"])
    
    with tab1:
        
    
        st.header("Performance Metrics Comparison")
        st.markdown("Compare the performance of methods with precision, recall, and F-score metrics.")
        plot_type = st.radio(
            "Select plot type:",
            options=['consolidated', 'individual'],
            format_func=lambda x: {
                'consolidated': 'All Metrics (Precision, Recall, F-score)', 
                'individual': 'Individual Metrics'
            }[x],
            index=0,
            horizontal=True
        )
        
        if plot_type == 'consolidated':
            fig = create_consolidated_performance_plot(
                st.session_state.nk_data, 
                st.session_state.lk_data,
                st.session_state.pk_data,
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
                st.session_state.pk_data,
                selected_methods,
                selected_metric
            )
            st.plotly_chart(fig, use_container_width=True)
    

        st.header("Precision-Recall Curves")
        st.markdown("Interactive precision-recall curves showing the trade-off between precision and recall across different thresholds for all aspects and subsets.")
        
        fig = create_interactive_precision_recall_plot(
            st.session_state.nk_all_data,
            st.session_state.lk_all_data,
            st.session_state.pk_all_data,
            selected_methods
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Summary Table")
        st.markdown("Detailed performance metrics for all selected methods across subsets and aspects.")
        
        # Create summary data
        aspects = ['biological_process', 'molecular_function', 'cellular_component']
        summary_data = []

        for subset, data in [('NK', st.session_state.nk_data), ('LK', st.session_state.lk_data), ('PK', st.session_state.pk_data)]:
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

