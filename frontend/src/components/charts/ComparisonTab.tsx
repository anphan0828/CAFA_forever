import { useState, Fragment } from 'react'
import { ComparisonFacet } from './ComparisonFacet'
import type { ComparisonData } from '../../hooks/useComparisonData'
import type { Subset, Aspect } from '../../types'
import { ASPECTS, SUBSET_LABELS, ASPECT_SHORT, makeBestKey } from '../../types'
import { InfoIcon } from '../ui'
import './ComparisonTab.css'

interface ComparisonTabProps {
  comparisonData: ComparisonData
  windowALabel: string
  windowBLabel: string
  loading?: boolean
}

const METRICS = [
  { key: 'fmax', label: 'F-max' },
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'coverage', label: 'Coverage' },
] as const

const SUBSET_TABS: Subset[] = ['NK', 'LK', 'PK']

export function ComparisonTab({
  comparisonData,
  windowALabel,
  windowBLabel,
  loading = false,
}: ComparisonTabProps) {
  const [activeSubset, setActiveSubset] = useState<Subset>('NK')

  if (loading) {
    return (
      <div className="comparison-tab comparison-tab--loading">
        <div className="loading-spinner" />
        <p>Loading comparison data...</p>
      </div>
    )
  }

  if (!comparisonData.ready) {
    return (
      <div className="comparison-tab comparison-tab--empty">
        <p>Select two evaluation windows to compare methods.</p>
      </div>
    )
  }

  if (comparisonData.commonMethods.length === 0) {
    return (
      <div className="comparison-tab comparison-tab--empty">
        <p>No common methods found between the selected windows.</p>
        <p className="comparison-tab__hint">
          Try selecting windows that share overlapping methods.
        </p>
      </div>
    )
  }

  return (
    <div className="comparison-tab">
      <div className="comparison-tab__header">
        <h3 className="comparison-tab__title">
          Window Comparison
          <InfoIcon tooltip="Compare method performance across two evaluation windows. Switch between knowledge levels using the tabs below." />
        </h3>
        <div className="comparison-tab__legend">
          <span className="comparison-tab__legend-item comparison-tab__legend-item--a">
            {windowALabel}
          </span>
          <span className="comparison-tab__legend-vs">vs</span>
          <span className="comparison-tab__legend-item comparison-tab__legend-item--b">
            {windowBLabel}
          </span>
        </div>
        <p className="comparison-tab__summary">
          Comparing {comparisonData.commonMethods.length} methods
        </p>
      </div>

      {/* Subset tabs */}
      <div className="comparison-tab__subtabs">
        {SUBSET_TABS.map((subset) => (
          <button
            key={subset}
            className={`comparison-tab__subtab ${activeSubset === subset ? 'comparison-tab__subtab--active' : ''}`}
            onClick={() => setActiveSubset(subset)}
          >
            {SUBSET_LABELS[subset]}
          </button>
        ))}
      </div>

      {/* Grid for active subset: 3 aspects × 4 metrics */}
      <div className="comparison-tab__table">
        {/* Column headers */}
        <div className="comparison-tab__corner" />
        {METRICS.map((metric) => (
          <div key={metric.key} className="comparison-tab__col-header">
            {metric.label}
          </div>
        ))}

        {/* Rows - one per aspect */}
        {ASPECTS.map((aspect: Aspect) => {
          const key = makeBestKey(activeSubset, aspect)
          const data = comparisonData.data[key] || []

          return (
            <Fragment key={key}>
              <div className="comparison-tab__row-header">
                {ASPECT_SHORT[aspect]}
              </div>
              {METRICS.map((metric) => (
                <div key={`${key}-${metric.key}`} className="comparison-tab__cell">
                  <ComparisonFacet
                    data={data}
                    metric={metric.key}
                  />
                </div>
              ))}
            </Fragment>
          )
        })}
      </div>
    </div>
  )
}
