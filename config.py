#!/usr/bin/env python3
"""
Configuration for CAFA Forever application
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIRS = {
    'AprJun': BASE_DIR / 'AprJun',
    # Add more timepoints as they become available
    # 'JulSep': BASE_DIR / 'JulSep',
    # 'OctDec': BASE_DIR / 'OctDec',
}

DATA_DATES = {
    'AprJun': {
        'go_start': '2025-03-16',
        'go_end': '2025-06-01',
        'uniprot_start': '2025_02',
        'uniprot_end': '2025_03',
    }
}

# Default file patterns
GROUND_TRUTH_PATTERN = "groundtruth_*_{subset}.tsv"
METHOD_NAMES_FILE = "method_names.tsv"

# Default subset names
SUBSETS = ['NK', 'LK']

# GO aspects
GO_ASPECTS = {
    'biological_process': 'Biological Process',
    'molecular_function': 'Molecular Function', 
    'cellular_component': 'Cellular Component'
}

# For dynamic discovery of available timepoints
def get_available_timepoints():
    """Discover available timepoint directories."""
    timepoints = []
    for item in BASE_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name not in ['cafa6', '__pycache__']:
            # Check if it has the expected structure
            if (item / 'results_NK').exists() and (item / 'results_LK').exists():
                timepoints.append(item.name)
    return sorted(timepoints)

# Web deployment settings
STREAMLIT_CONFIG = {
    'page_title': "CAFA Forever",
    'page_icon': "📊",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}
