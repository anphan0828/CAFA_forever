import { useMemo, useState } from 'react'
import type { BestMetricsMap, Subset, Aspect } from '../../types'
import { SUBSETS, ASPECTS, SUBSET_LABELS, ASPECT_LABELS } from '../../types'
import { Tooltip } from '../ui'
import { CSVExport } from './CSVExport'
import './SummaryTable.css'

type SubsetFilter = Subset | 'AVG'

const SUBSET_FILTER_LABELS: Record<SubsetFilter, string> = {
  ...SUBSET_LABELS,
  AVG: 'Average',
}

const SUBSET_DESCRIPTIONS: Record<SubsetFilter, string> = {
  NK: 'No Knowledge: Proteins with no prior experimental GO annotations - the hardest prediction task',
  LK: 'Limited Knowledge: Proteins with some non-experimental annotations',
  PK: 'Prior Knowledge: Proteins with existing experimental annotations in some GO aspects',
  AVG: 'Average across all knowledge subsets (NK, LK, PK) - matches the Summary chart values',
}

interface SummaryTableProps {
  bestMetrics: BestMetricsMap
  selectedMethods: string[]
  releaseId: string
}

type SortKey = 'method' | 'fmax' | 'precision' | 'recall' | 'coverage' | 'threshold'
type SortDirection = 'asc' | 'desc'

const SUBSET_FILTERS: SubsetFilter[] = ['AVG', ...SUBSETS]

export function SummaryTable({
  bestMetrics,
  selectedMethods,
  releaseId,
}: SummaryTableProps) {
  const [activeSubset, setActiveSubset] = useState<SubsetFilter>('AVG')
  const [activeAspect, setActiveAspect] = useState<Aspect>('biological_process')
  const [sortKey, setSortKey] = useState<SortKey>('fmax')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  const filteredData = useMemo(() => {
    if (activeSubset === 'AVG') {
      // Compute average across all subsets for the active aspect
      const methodAverages = new Map<string, {
        method: string
        fmax: number
        precision: number
        recall: number
        coverage: number
        threshold: number
        count: number
      }>()

      // Initialize for selected methods
      selectedMethods.forEach((method) => {
        methodAverages.set(method, {
          method,
          fmax: 0,
          precision: 0,
          recall: 0,
          coverage: 0,
          threshold: 0,
          count: 0,
        })
      })

      // Aggregate across all subsets for the active aspect
      SUBSETS.forEach((subset) => {
        const key = `${subset}_${activeAspect}`
        const metrics = bestMetrics[key] || []

        metrics.forEach((metric) => {
          if (selectedMethods.includes(metric.method)) {
            const entry = methodAverages.get(metric.method)!
            entry.fmax += metric.fmax
            entry.precision += metric.precision
            entry.recall += metric.recall
            entry.coverage += metric.coverage
            entry.threshold += metric.threshold
            entry.count++
          }
        })
      })

      // Calculate averages
      return Array.from(methodAverages.values())
        .filter((entry) => entry.count > 0)
        .map((entry) => ({
          method: entry.method,
          subset: 'AVG' as Subset,
          aspect: activeAspect,
          fmax: entry.fmax / entry.count,
          precision: entry.precision / entry.count,
          recall: entry.recall / entry.count,
          coverage: entry.coverage / entry.count,
          threshold: entry.threshold / entry.count,
          n: 0,
        }))
    }

    const key = `${activeSubset}_${activeAspect}`
    const data = bestMetrics[key] || []
    return data.filter((row) => selectedMethods.includes(row.method))
  }, [bestMetrics, selectedMethods, activeSubset, activeAspect])

  const sortedData = useMemo(() => {
    return [...filteredData].sort((a, b) => {
      const aVal = a[sortKey]
      const bVal = b[sortKey]
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      const aNum = aVal as number
      const bNum = bVal as number
      return sortDirection === 'asc' ? aNum - bNum : bNum - aNum
    })
  }, [filteredData, sortKey, sortDirection])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDirection('desc')
    }
  }

  const columns = [
    { key: 'method' as SortKey, label: 'Method' },
    { key: 'fmax' as SortKey, label: 'F-max' },
    { key: 'precision' as SortKey, label: 'Precision' },
    { key: 'recall' as SortKey, label: 'Recall' },
    { key: 'coverage' as SortKey, label: 'Coverage' },
    { key: 'threshold' as SortKey, label: 'Threshold' },
  ]

  return (
    <div className="summary-table">
      <div className="summary-table__header">
        <h4 className="summary-table__title">Evaluation Metrics</h4>
        <CSVExport
          data={sortedData}
          filename={`lafa_${releaseId}_${activeSubset}_${activeAspect}.csv`}
          disabled={sortedData.length === 0}
        />
      </div>

      <div className="summary-table__filters">
        <div className="summary-table__filter-group">
          <label>Subset:</label>
          <div className="summary-table__filter-buttons">
            {SUBSET_FILTERS.map((subset) => (
              <Tooltip key={subset} content={SUBSET_DESCRIPTIONS[subset]} position="bottom">
                <button
                  className={`summary-table__filter-btn ${activeSubset === subset ? 'summary-table__filter-btn--active' : ''}`}
                  onClick={() => setActiveSubset(subset)}
                >
                  {SUBSET_FILTER_LABELS[subset]}
                </button>
              </Tooltip>
            ))}
          </div>
        </div>

        <div className="summary-table__filter-group">
          <label>Aspect:</label>
          <div className="summary-table__filter-buttons">
            {ASPECTS.map((aspect) => (
              <button
                key={aspect}
                className={`summary-table__filter-btn ${activeAspect === aspect ? 'summary-table__filter-btn--active' : ''}`}
                onClick={() => setActiveAspect(aspect)}
              >
                {ASPECT_LABELS[aspect]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="summary-table__wrapper">
        <table className="summary-table__table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className={sortKey === col.key ? 'summary-table__th--sorted' : ''}
                >
                  <span>{col.label}</span>
                  {sortKey === col.key && (
                    <span className="summary-table__sort-icon">
                      {sortDirection === 'asc' ? '\u2191' : '\u2193'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedData.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="summary-table__empty">
                  No data available for selected methods
                </td>
              </tr>
            ) : (
              sortedData.map((row, i) => (
                <tr key={row.method} className={i === 0 ? 'summary-table__tr--best' : ''}>
                  <td className="summary-table__td--method">{row.method}</td>
                  <td>{row.fmax.toFixed(4)}</td>
                  <td>{row.precision.toFixed(4)}</td>
                  <td>{row.recall.toFixed(4)}</td>
                  <td>{row.coverage.toFixed(4)}</td>
                  <td>{row.threshold.toFixed(2)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
