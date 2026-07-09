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
} from 'recharts'
import type { BestMetrics } from '../../types'
import { useMethodColors } from '../../hooks'
import './TopMethodsChart.css'

interface TopMethodsChartProps {
  data: BestMetrics[]
  maxMethods?: number
  metricKey?: 'fmax' | 'precision' | 'recall' | 'coverage'
  title?: string
}

export function TopMethodsChart({
  data,
  maxMethods = 10,
  metricKey = 'fmax',
  title = 'Top Methods by F-max',
}: TopMethodsChartProps) {
  // Sort and slice data
  const chartData = useMemo(() => {
    const sorted = [...data].sort((a, b) => b[metricKey] - a[metricKey])
    return sorted.slice(0, maxMethods).map((item) => ({
      method: item.method,
      value: item[metricKey],
      precision: item.precision,
      recall: item.recall,
      fmax: item.fmax,
      coverage: item.coverage,
      threshold: item.threshold,
    }))
  }, [data, maxMethods, metricKey])

  const allMethods = useMemo(() => chartData.map((d) => d.method), [chartData])
  const { getColor } = useMethodColors(allMethods)

  const metricLabels: Record<string, string> = {
    fmax: 'F-max',
    precision: 'Precision',
    recall: 'Recall',
    coverage: 'Coverage',
  }

  if (chartData.length === 0) {
    return (
      <div className="top-methods-chart top-methods-chart--empty">
        <p>No data available</p>
      </div>
    )
  }

  return (
    <div className="top-methods-chart">
      <h4 className="top-methods-chart__title">{title}</h4>
      <div className="top-methods-chart__container">
        <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 40)}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
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
              width={90}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                const data = payload[0].payload
                return (
                  <div className="top-methods-chart__tooltip">
                    <strong>{data.method}</strong>
                    <div>{metricLabels[metricKey]}: {data.value.toFixed(4)}</div>
                    <div>Precision: {data.precision.toFixed(4)}</div>
                    <div>Recall: {data.recall.toFixed(4)}</div>
                    <div>Threshold: {data.threshold.toFixed(2)}</div>
                  </div>
                )
              }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry) => (
                <Cell key={entry.method} fill={getColor(entry.method)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
