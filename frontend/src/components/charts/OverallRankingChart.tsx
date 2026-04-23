import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from 'recharts'
import type { BestMetricsMap } from '../../types'
import { SUBSETS, ASPECTS, SUBSET_LABELS } from '../../types'
import { useMethodColors } from '../../hooks'
import { InfoIcon } from '../ui'
import './OverallRankingChart.css'

interface OverallRankingChartProps {
  bestMetrics: BestMetricsMap
  selectedMethods: string[]
}

interface MethodScore {
  method: string
  avgFmax: number
  scores: Record<string, number>
  count: number
}

export function OverallRankingChart({
  bestMetrics,
  selectedMethods,
}: OverallRankingChartProps) {
  const chartData = useMemo(() => {
    // Compute average F-max across all subset/aspect combinations for each method
    const methodScores = new Map<string, MethodScore>()

    // Initialize scores for selected methods
    selectedMethods.forEach((method) => {
      methodScores.set(method, {
        method,
        avgFmax: 0,
        scores: {},
        count: 0,
      })
    })

    // Aggregate scores across all subset/aspect combinations
    for (const subset of SUBSETS) {
      for (const aspect of ASPECTS) {
        const key = `${subset}_${aspect}`
        const metrics = bestMetrics[key] || []

        metrics.forEach((metric) => {
          if (selectedMethods.includes(metric.method)) {
            const entry = methodScores.get(metric.method)!
            entry.scores[key] = metric.fmax
            entry.count++
          }
        })
      }
    }

    // Calculate averages and sort
    const result = Array.from(methodScores.values())
      .map((entry) => {
        const values = Object.values(entry.scores)
        const avgFmax = values.length > 0
          ? values.reduce((sum, v) => sum + v, 0) / values.length
          : 0
        return { ...entry, avgFmax }
      })
      .filter((entry) => entry.count > 0)
      .sort((a, b) => b.avgFmax - a.avgFmax)

    return result
  }, [bestMetrics, selectedMethods])

  const { getColor } = useMethodColors(selectedMethods)

  if (chartData.length === 0) {
    return (
      <div className="overall-ranking-chart overall-ranking-chart--empty">
        <p>Select methods to view rankings</p>
      </div>
    )
  }

  return (
    <div className="overall-ranking-chart">
      <h4 className="overall-ranking-chart__title">
        Overall Method Ranking
        <InfoIcon tooltip="Average F-max across all 3 knowledge subsets (NK, LK, PK) and all 3 GO aspects (BP, MF, CC). Higher is better." />
      </h4>
      <div className="overall-ranking-chart__container">
        <ResponsiveContainer width="100%" height={Math.max(250, chartData.length * 36)}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 120, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
            <XAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(value) => value.toFixed(2)}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              type="category"
              dataKey="method"
              tick={{ fontSize: 12 }}
              width={110}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                const data = payload[0].payload as MethodScore
                return (
                  <div className="overall-ranking-chart__tooltip">
                    <strong>{data.method}</strong>
                    <div className="overall-ranking-chart__tooltip-avg">
                      Average F-max: {data.avgFmax.toFixed(4)}
                    </div>
                    <div className="overall-ranking-chart__tooltip-grid">
                      {SUBSETS.map((subset) => (
                        <div key={subset} className="overall-ranking-chart__tooltip-row">
                          <span className="overall-ranking-chart__tooltip-label">
                            {SUBSET_LABELS[subset]}:
                          </span>
                          {ASPECTS.map((aspect) => {
                            const key = `${subset}_${aspect}`
                            const score = data.scores[key]
                            return (
                              <span key={key} className="overall-ranking-chart__tooltip-cell">
                                {score !== undefined ? score.toFixed(3) : '-'}
                              </span>
                            )
                          })}
                        </div>
                      ))}
                    </div>
                  </div>
                )
              }}
            />
            <Bar dataKey="avgFmax" radius={[0, 4, 4, 0]}>
              {chartData.map((entry) => (
                <Cell key={entry.method} fill={getColor(entry.method)} />
              ))}
              <LabelList
                dataKey="avgFmax"
                position="right"
                formatter={(value: number) => value.toFixed(2)}
                style={{ fontSize: 11, fill: 'var(--isu-charcoal)' }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
