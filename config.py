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
    'Apr_2025_Jun_2025': BASE_DIR / 'Apr_2025_Jun_2025',
    'Jun_2025_Oct_2025': BASE_DIR / 'Jun_2025_Oct_2025',
    # Add more timepoints as they become available
    # 'JulSep': BASE_DIR / 'JulSep',
    # 'OctDec': BASE_DIR / 'OctDec',
}

DATA_DATES = {
    'Apr_2025_Jun_2025': {
        'go_start': '2025-03-16',
        'go_end': '2025-06-01',
        'uniprot_start': '2025_02 (2025-04-09)',
        'uniprot_end': '2025_03 (2025-06-18)',
    },
    'Jun_2025_Oct_2025': {
        'go_start': '2025-06-01',
        'go_end': '2025-10-10',
        'uniprot_start': '2025_03 (2025-06-18)',
        'uniprot_end': '2025_04 (2025-10-15)',
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

# Docker-specific settings
if os.getenv('DOCKER_ENV'):
    STREAMLIT_CONFIG.update({
        "server.port": int(os.getenv('STREAMLIT_SERVER_PORT', 8501)),
        "server.address": os.getenv('STREAMLIT_SERVER_ADDRESS', '0.0.0.0'),
        "server.headless": True,
        "browser.gatherUsageStats": False
    })