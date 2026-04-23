import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { BestMetricsMap, Subset } from '../../types'
import { ASPECTS, ASPECT_SHORT, SUBSET_LABELS } from '../../types'
import { useMethodColors } from '../../hooks'
import { InfoIcon } from '../ui'
import './AverageFmaxChart.css'

interface AverageFmaxChartProps {
  bestMetrics: BestMetricsMap
  selectedMethods: string[]
  subset: Subset
}

export function AverageFmaxChart({
  bestMetrics,
  selectedMethods,
  subset,
}: AverageFmaxChartProps) {
  const { getColor } = useMethodColors(selectedMethods)

  const chartData = useMemo(() => {
    return ASPECTS.map((aspect) => {
      const key = `${subset}_${aspect}`
      const metrics = bestMetrics[key] || []

      const row: Record<string, unknown> = {
        aspect: ASPECT_SHORT[aspect],
        aspectFull: aspect,
      }

      selectedMethods.forEach((method) => {
        const metric = metrics.find((m) => m.method === method)
        row[method] = metric?.fmax ?? null
      })

      return row
    })
  }, [bestMetrics, selectedMethods, subset])

  if (selectedMethods.length === 0) {
    return (
      <div className="average-fmax-chart average-fmax-chart--empty">
        <p>Select methods to view F-max comparison</p>
      </div>
    )
  }

  return (
    <div className="average-fmax-chart">
      <h4 className="average-fmax-chart__title">
        F-max by GO Aspect ({SUBSET_LABELS[subset]})
        <InfoIcon tooltip="F-max is the maximum F₁ score across all prediction thresholds. Higher values indicate better precision-recall balance." />
      </h4>
      <div className="average-fmax-chart__container">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="aspect" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} tick={{ fontSize: 12 }} />
            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null
                const aspectData = chartData.find((d) => d.aspect === label)
                return (
                  <div className="average-fmax-chart__tooltip">
                    <strong>{aspectData?.aspectFull as string}</strong>
                    {payload.map((entry) => (
                      <div key={entry.dataKey as string} style={{ color: entry.color }}>
                        {entry.dataKey}: {(entry.value as number)?.toFixed(4) ?? 'N/A'}
                      </div>
                    ))}
                  </div>
                )
              }}
            />
            <Legend />
            {selectedMethods.map((method) => (
              <Bar
                key={method}
                dataKey={method}
                fill={getColor(method)}
                radius={[4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
