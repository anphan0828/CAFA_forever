# CAFA Forever - Continuous Functional Annotation Assessment

Interactive visualization of CAFA (Critical Assessment of Functional Annotation) evaluation results using Streamlit.

## Quick Start

### Local Development

1. **Setup Environment:**
   ```bash
   # Make setup script executable (if not already)
   chmod +x setup_env.sh
   
   # Run setup
   ./setup_env.sh
   ```

2. **Activate Environment:**
   ```bash
   source lafa/bin/activate
   ```

3. **Run Application:**
   ```bash
   streamlit run streamlit_plot.py
   ```

## Project Structure

```
CAFA-forever/
├── streamlit_plot.py       # Main Streamlit application
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── setup_env.sh           # Environment setup script
├── AprJun/                # April-June timepoint data
│   ├── results_NK/        # No Knowledge protein results
│   ├── results_LK/        # Limited Knowledge protein results
│   ├── groundtruth_*.tsv  # Ground truth annotations
│   └── method_names.tsv   # Method name mappings

```

## Adding New Timepoints

To add a new evaluation timepoint (e.g., JunAug):

1. Create a new directory following the same structure as `AprJun/`
2. Ensure it contains:
   - `results_NK/` and `results_LK/` directories
   - Ground truth files: `groundtruth_*_NK.tsv` and `groundtruth_*_LK.tsv`
   - `method_names.tsv` file
3. The application will automatically detect the new timepoint (needs testing)


## Dependencies

- Python 3.9+
- Streamlit ≥1.28.0
- Pandas ≥1.5.0
- Plotly ≥5.15.0
- Matplotlib ≥3.5.0
- NumPy ≥1.20.0

## Features

- Interactive method comparison with checkboxes
- Performance metrics visualization (Precision, Recall, F-score)
- Precision-Recall curves with F-score contours
- Target count comparison vs ground truth
- Multi-timepoint support (automatically detected)
- Export functionality for summary tables

## Usage

The application provides three main visualization tabs:

1. **Performance Metrics**: Compare methods across different metrics and GO aspects
2. **Precision-Recall**: Interactive P-R curves with threshold exploration
3. **Summary Table**: Detailed metrics table with export functionality

Use the sidebar to:
- Select timepoint (if multiple available)
- Choose methods to compare
- Configure visualization parameters
