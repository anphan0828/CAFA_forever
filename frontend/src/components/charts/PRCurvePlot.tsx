import { useMemo } from 'react'
import { Group } from '@visx/group'
import { scaleLinear } from '@visx/scale'
import { LinePath, Circle } from '@visx/shape'
import { AxisBottom, AxisLeft } from '@visx/axis'
import { GridRows, GridColumns } from '@visx/grid'
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
  margin?: { top: number; right: number; bottom: number; left: number }
}

const DEFAULT_MARGIN = { top: 30, right: 20, bottom: 50, left: 50 }

export function PRCurvePlot({
  curves,
  width = 350,
  height = 300,
  title,
  showContours = true,
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
    return generateAllContours([0.2, 0.4, 0.6, 0.8], 100)
  }, [showContours])

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
      <svg width={width} height={height}>
        <Group left={margin.left} top={margin.top}>
          {/* Grid */}
          <GridRows
            scale={yScale}
            width={innerWidth}
            stroke="#e5e5e5"
            strokeDasharray="2,2"
          />
          <GridColumns
            scale={xScale}
            height={innerHeight}
            stroke="#e5e5e5"
            strokeDasharray="2,2"
          />

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
          {curves.map((curve) => (
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
              {curve.best && (
                <Circle
                  cx={xScale(curve.best.recall)}
                  cy={yScale(curve.best.precision)}
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
            label="Recall"
            labelOffset={35}
            tickFormat={(v) => Number(v).toFixed(1)}
            tickLabelProps={() => ({ fontSize: 10, textAnchor: 'middle' })}
          />
          <AxisLeft
            scale={yScale}
            label="Precision"
            labelOffset={35}
            tickFormat={(v) => Number(v).toFixed(1)}
            tickLabelProps={() => ({ fontSize: 10, textAnchor: 'end' })}
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
