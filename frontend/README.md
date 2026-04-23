# LAFA Frontend

React-based frontend for the LAFA (Longitudinal Assessment of Function Annotation) leaderboard. This replaces the previous Streamlit implementation.

> **Note**: For comprehensive setup and deployment instructions, see the [main README](../README.md) in the repository root.

## Why React Instead of Streamlit?

### Performance
- **Client-side rendering**: Interactions are instant, no server round-trips.
- **Static assets**: Built files are CDN-friendly and cache-efficient.
- **Lazy loading**: PR curve data is loaded only when the Curves tab is selected.

### Deployment
- **Simple infrastructure**: A single Nginx container serves everything. No need for Streamlit server, websocket connections, or session management.
- **Health checks**: Standard HTTP health endpoint at `/_health`.
- **Same port**: Runs on port 8501 to match the previous Streamlit deployment.

### Trade-offs
- **Build step required**: Changes to the UI require `npm run build`. Streamlit's hot-reload was more immediate for prototyping.
- **JavaScript knowledge**: Maintainers need familiarity with React/TypeScript rather than pure Python.
- **Data pipeline**: The `generate_data.py` script must run whenever evaluation data updates.

## Tech Stack

- **Vite** - Build tool
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Visx** - PR curve visualizations
- **Recharts** - Bar charts
- **Nginx** - Production serving

## Development

### Prerequisites

- Node.js 20+
- Python 3.9+ (for data generation)

### Setup

```bash
# Install dependencies
npm install

# Generate JSON data from TSV files
python scripts/generate_data.py

# Start development server
npm run dev
```

Open http://localhost:5173 in your browser.

### Build

```bash
npm run build
```

Output is in `dist/`.

## Data Pipeline

The `scripts/generate_data.py` script transforms TSV evaluation data into JSON:

**Input:** `data/releases/{release_id}/*.tsv`

**Output:** `public/data/`
- `catalog.json` - Available releases
- `methods.json` - Method configurations
- `releases/{id}/meta.json` - Release metadata
- `releases/{id}/methods.json` - Method availability
- `releases/{id}/best.json` - Best metrics by subset/aspect
- `releases/{id}/curves.json` - Full PR curve data

## Production Deployment

### Docker

From repository root:

```bash
# Build
docker build -f frontend/Dockerfile.full -t lafa-frontend .

# Run
docker run -p 8501:8501 lafa-frontend
```

### Docker Compose

```bash
cd frontend
docker compose up -d
```

Access at http://localhost:8501

### Health Check

```bash
curl http://localhost:8501/_health
```

## Project Structure

```
frontend/
├── public/
│   ├── assets/iastate/     # ISU brand logos
│   └── data/               # Generated JSON data
├── scripts/
│   └── generate_data.py    # TSV → JSON transformation
├── src/
│   ├── components/
│   │   ├── charts/         # Visualizations (Recharts, Visx)
│   │   ├── layout/         # Header, Footer, Section
│   │   ├── methods/        # Method selection UI
│   │   ├── release/        # Release cards, selectors
│   │   ├── table/          # Data table, CSV export
│   │   └── ui/             # Tabs, Checkbox, etc.
│   ├── context/            # React Context state
│   ├── hooks/              # Data fetching hooks
│   ├── lib/                # Utilities (F-score contours)
│   └── types/              # TypeScript interfaces
├── package.json
├── vite.config.ts
├── tsconfig.json
├── nginx.conf
└── Dockerfile.full
```

## Key Components

### PR Curves (`PRCurveGrid`)
- 3x3 grid: subsets (NK, LK, PK) × aspects (BP, MF, CC)
- F-score iso-contours (0.2, 0.4, 0.6, 0.8)
- Best points marked on curves
- Built with Visx for performance

### Summary Charts
- `TargetCountChart` - Stacked bar by subset
- `AverageFmaxChart` - Grouped bars by aspect
- `TopMethodsChart` - Ranked horizontal bars

### Data Table
- Sortable columns
- Subset/aspect filtering
- CSV export

### Window Comparison (`ComparisonTab`)
- Compare method performance across two evaluation windows
- Subtabs for NK, LK, PK knowledge levels
- Grouped bar charts showing F-max, Precision, Recall, Coverage
- Responds to method selector and baseline toggle

## Integration with Backend

The frontend reads published releases from `../data/releases/`. When the backend publishes new releases:

```bash
# Regenerate JSON data
python scripts/generate_data.py

# For Docker deployment, rebuild the image
cd .. && docker build -f frontend/Dockerfile.full -t lafa-frontend .
```

## ISU Branding

CSS variables in `src/index.css`:
```css
:root {
  --isu-cardinal: #c8102e;
  --isu-cardinal-dark: #7c2529;
  --isu-gold: #f1be48;
  --isu-charcoal: #212529;
}
```
