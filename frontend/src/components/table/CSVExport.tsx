import type { BestMetrics } from '../../types'
import './CSVExport.css'

interface CSVExportProps {
  data: BestMetrics[]
  filename: string
  disabled?: boolean
}

export function CSVExport({ data, filename, disabled }: CSVExportProps) {
  const handleExport = () => {
    if (data.length === 0) return

    const headers = ['Method', 'Subset', 'Aspect', 'F-max', 'Precision', 'Recall', 'Coverage', 'Threshold', 'N']
    const rows = data.map((row) => [
      row.method,
      row.subset,
      row.aspect,
      row.fmax.toFixed(6),
      row.precision.toFixed(6),
      row.recall.toFixed(6),
      row.coverage.toFixed(6),
      row.threshold.toFixed(4),
      row.n.toString(),
    ])

    const csvContent = [
      headers.join(','),
      ...rows.map((row) => row.join(',')),
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.setAttribute('href', url)
    link.setAttribute('download', filename)
    link.style.visibility = 'hidden'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  return (
    <button
      className="csv-export"
      onClick={handleExport}
      disabled={disabled}
      aria-label="Export data as CSV"
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
      Export CSV
    </button>
  )
}
