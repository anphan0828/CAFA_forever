import { Fragment, useMemo } from 'react'
import type { CurvesMap, BestMetricsMap, Subset, Aspect } from '../../types'
import { SUBSETS, ASPECTS, SUBSET_LABELS, ASPECT_LABELS, makeCurveKey, makeBestKey } from '../../types'
import { useMethodColors } from '../../hooks'
import { InfoIcon } from '../ui'
import { PRCurvePlot } from './PRCurvePlot'
import './PRCurveGrid.css'

interface PRCurveGridProps {
  curves: CurvesMap
  bestMetrics: BestMetricsMap
  selectedMethods: string[]
  colorDomain?: string[]
  totalSelectedCount?: number
  subsets?: Subset[]
  aspects?: Aspect[]
}

export function PRCurveGrid({
  curves,
  bestMetrics,
  selectedMethods,
  colorDomain = selectedMethods,
  totalSelectedCount,
  subsets = SUBSETS,
  aspects = ASPECTS,
}: PRCurveGridProps) {
  const showingLimitedMethods = totalSelectedCount !== undefined && totalSelectedCount > selectedMethods.length
  const { getColor } = useMethodColors(colorDomain)

  const gridPlots = useMemo(() => {
    const plots: Array<{
      subset: Subset
      aspect: Aspect
      curveData: Array<{
        method: string
        color: string
        points: Array<{ tau: number; precision: number; recall: number }>
        best?: {
          method: string
          precision: number
          recall: number
          threshold: number
          fmax: number
          coverage: number
          subset: Subset
          aspect: Aspect
          n: number
        }
      }>
    }> = []

    for (const subset of subsets) {
      for (const aspect of aspects) {
        const curveData = selectedMethods.map((method) => {
          const curveKey = makeCurveKey(subset, aspect, method)
          const points = curves[curveKey] || []

          const bestKey = makeBestKey(subset, aspect)
          const bestList = bestMetrics[bestKey] || []
          const best = bestList.find((b) => b.method === method)

          return {
            method,
            color: getColor(method),
            points,
            best,
          }
        }).filter((d) => d.points.length > 0)

        plots.push({ subset, aspect, curveData })
      }
    }

    return plots
  }, [curves, bestMetrics, selectedMethods, subsets, aspects, getColor])

  if (selectedMethods.length === 0) {
    return (
      <div className="pr-curve-grid pr-curve-grid--empty">
        <p>Select methods to view PR curves</p>
      </div>
    )
  }

  return (
    <div className="pr-curve-grid">
      <div className="pr-curve-grid__header-row">
        <h4 className="pr-curve-grid__title">
          Precision-Recall Curves
          <InfoIcon tooltip="PR curves show the trade-off between precision and recall at different prediction thresholds. The dashed curves show iso-F contours. Points mark each method's best F-max threshold." />
        </h4>
        {showingLimitedMethods && (
          <span className="pr-curve-grid__limit-notice">
            Showing first {selectedMethods.length} of {totalSelectedCount} selected methods
          </span>
        )}
      </div>
      {/* Legend */}
      <div className="pr-curve-grid__legend">
        {selectedMethods.map((method) => (
          <div key={method} className="pr-curve-grid__legend-item">
            <span
              className="pr-curve-grid__legend-color"
              style={{ backgroundColor: getColor(method) }}
            />
            <span className="pr-curve-grid__legend-label">{method}</span>
          </div>
        ))}
      </div>

      <div className="pr-curve-grid__plot-area">
        <div className="pr-curve-grid__plot-stack">
          {/* Grid */}
          <div
            className="pr-curve-grid__container"
            style={{
              gridTemplateColumns: `repeat(${aspects.length}, minmax(0, 1fr))`,
            }}
          >
            {/* Plots with row headers */}
            {subsets.map((subset, rowIndex) => (
              <Fragment key={subset}>
                <div
                  className="pr-curve-grid__row-label"
                  style={{ gridColumn: `1 / span ${aspects.length}` }}
                >
                  {SUBSET_LABELS[subset]}
                </div>
                {aspects.map((aspect, colIndex) => (
                  <div
                    key={`${subset}_${aspect}`}
                    className="pr-curve-grid__cell"
                  >
                    <PRCurvePlot
                      curves={gridPlots.find(
                        (p) => p.subset === subset && p.aspect === aspect
                      )?.curveData || []}
                      width={250}
                      height={250}
                      title={ASPECT_LABELS[aspect]}
                      showAxisLabels={false}
                      showYAxisLabel={colIndex === 0}
                      showXAxisLabel={rowIndex === subsets.length - 1}
                    />
                  </div>
                ))}
              </Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
