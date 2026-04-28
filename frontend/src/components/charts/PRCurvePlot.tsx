import { useMemo } from 'react'
import { Group } from '@visx/group'
import { scaleLinear } from '@visx/scale'
import { LinePath, Circle } from '@visx/shape'
import { AxisBottom, AxisLeft } from '@visx/axis'
import { useTooltip } from '@visx/tooltip'
import type { CurvePoint, BestMetrics } from '../../types'
import { generateAllContours, type Point } from '../../lib/fscoreContour'
import './PRCurvePlot.css'

interface CurveData {
  method: string
  color: string
  points: CurvePoint[]
  best?: BestMetrics
}

interface PRCurvePlotProps {
  curves: CurveData[]
  width?: number
  height?: number
  title?: string
  showContours?: boolean
  showAxisLabels?: boolean
  showXAxisLabel?: boolean
  showYAxisLabel?: boolean
  margin?: { top: number; right: number; bottom: number; left: number }
}

const DEFAULT_MARGIN = { top: 30, right: 20, bottom: 68, left: 58 }
const AXIS_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1]
const EPSILON = 1e-9

function sortCurvePoints(points: CurvePoint[]): CurvePoint[] {
  return [...points].sort((a, b) => {
    const recallDiff = a.recall - b.recall
    if (Math.abs(recallDiff) > EPSILON) return recallDiff

    const precisionDiff = a.precision - b.precision
    if (Math.abs(precisionDiff) > EPSILON) return precisionDiff

    return a.tau - b.tau
  })
}

function getBestMarkerPoint(curve: CurveData): CurvePoint | BestMetrics | null {
  if (!curve.best) return null

  const thresholdPoint = curve.points.find(
    (point) => Math.abs(point.tau - curve.best!.threshold) <= EPSILON
  )

  return thresholdPoint ?? curve.best
}

export function PRCurvePlot({
  curves,
  width = 350,
  height = 300,
  title,
  showContours = true,
  showAxisLabels = true,
  showXAxisLabel = showAxisLabels,
  showYAxisLabel = showAxisLabels,
  margin = DEFAULT_MARGIN,
}: PRCurvePlotProps) {
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom

  // Scales
  const xScale = useMemo(
    () => scaleLinear({ domain: [0, 1], range: [0, innerWidth] }),
    [innerWidth]
  )

  const yScale = useMemo(
    () => scaleLinear({ domain: [0, 1], range: [innerHeight, 0] }),
    [innerHeight]
  )

  // F-score contours
  const contours = useMemo(() => {
    if (!showContours) return null
    return generateAllContours([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 100)
  }, [showContours])

  const renderedCurves = useMemo(
    () =>
      curves.map((curve) => ({
        ...curve,
        points: sortCurvePoints(curve.points),
        bestMarker: getBestMarkerPoint(curve),
      })),
    [curves]
  )

  // Tooltip
  const {
    showTooltip,
    hideTooltip,
    tooltipData,
    tooltipLeft,
    tooltipTop,
    tooltipOpen,
  } = useTooltip<{ method: string; precision: number; recall: number; threshold: number }>()

  return (
    <div className="pr-curve-plot">
      {title && <h5 className="pr-curve-plot__title">{title}</h5>}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <Group left={margin.left} top={margin.top}>
          {/* F-score contours */}
          {contours &&
            Array.from(contours.entries()).map(([fScore, points]) => (
              <Group key={fScore}>
                <LinePath<Point>
                  data={points}
                  x={(d) => xScale(d.x)}
                  y={(d) => yScale(d.y)}
                  stroke="#ddd"
                  strokeWidth={1}
                  strokeDasharray="4,4"
                />
                {/* Label */}
                {points.length > 0 && (
                  <text
                    x={xScale(points[Math.floor(points.length * 0.7)]?.x ?? 0)}
                    y={yScale(points[Math.floor(points.length * 0.7)]?.y ?? 0) - 5}
                    fontSize={10}
                    fill="#999"
                  >
                    F={fScore}
                  </text>
                )}
              </Group>
            ))}

          {/* PR Curves */}
          {renderedCurves.map((curve) => (
            <Group key={curve.method}>
              <LinePath<CurvePoint>
                data={curve.points}
                x={(d) => xScale(d.recall)}
                y={(d) => yScale(d.precision)}
                stroke={curve.color}
                strokeWidth={2}
                strokeLinecap="round"
              />
              {/* Best point marker */}
              {curve.best && curve.bestMarker && (
                <Circle
                  cx={xScale(curve.bestMarker.recall)}
                  cy={yScale(curve.bestMarker.precision)}
                  r={5}
                  fill={curve.color}
                  stroke="white"
                  strokeWidth={2}
                  onMouseEnter={(e) => {
                    const rect = (e.target as SVGElement).getBoundingClientRect()
                    showTooltip({
                      tooltipData: {
                        method: curve.method,
                        precision: curve.best!.precision,
                        recall: curve.best!.recall,
                        threshold: curve.best!.threshold,
                      },
                      tooltipLeft: rect.left + rect.width / 2,
                      tooltipTop: rect.top - 10,
                    })
                  }}
                  onMouseLeave={hideTooltip}
                  style={{ cursor: 'pointer' }}
                />
              )}
            </Group>
          ))}

          {/* Axes */}
          <AxisBottom
            scale={xScale}
            top={innerHeight}
            label={showXAxisLabel ? 'Recall' : undefined}
            labelOffset={35}
            tickValues={AXIS_TICKS}
            tickFormat={(v) => Number(v).toFixed(1)}
            tickLabelProps={() => ({ fontSize: 11, textAnchor: 'middle' })}
          />
          <AxisLeft
            scale={yScale}
            label={showYAxisLabel ? 'Precision' : undefined}
            labelOffset={35}
            tickValues={AXIS_TICKS}
            tickFormat={(v) => Number(v).toFixed(1)}
            tickLabelProps={() => ({ fontSize: 11, textAnchor: 'end' })}
          />
        </Group>
      </svg>

      {/* Tooltip - rendered with fixed positioning */}
      {tooltipOpen && tooltipData && (
        <div
          className="pr-curve-plot__tooltip"
          style={{
            position: 'fixed',
            left: tooltipLeft,
            top: tooltipTop,
            transform: 'translate(-50%, -100%)',
            pointerEvents: 'none',
          }}
        >
          <strong>{tooltipData.method}</strong>
          <div>Precision: {tooltipData.precision.toFixed(4)}</div>
          <div>Recall: {tooltipData.recall.toFixed(4)}</div>
          <div>Threshold: {tooltipData.threshold.toFixed(2)}</div>
        </div>
      )}
    </div>
  )
}
