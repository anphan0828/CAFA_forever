import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LabelList,
  ResponsiveContainer,
} from 'recharts'
import type { BestMetricsMap, Subset } from '../../types'
import { SUBSETS, ASPECTS, ASPECT_SHORT, SUBSET_LABELS } from '../../types'
import { useMethodColors } from '../../hooks'
import { InfoIcon } from '../ui'
import './FmaxSmallMultiples.css'

interface FmaxSmallMultiplesProps {
  bestMetrics: BestMetricsMap
  selectedMethods: string[]
  colorDomain?: string[]
}

export function FmaxSmallMultiples({
  bestMetrics,
  selectedMethods,
  colorDomain = selectedMethods,
}: FmaxSmallMultiplesProps) {
  const { getColor } = useMethodColors(colorDomain)

  // Data structure: one entry per aspect, with method values
  const chartDataBySubset = useMemo(() => {
    const result: Record<Subset, Array<Record<string, unknown>>> = {
      NK: [],
      LK: [],
      PK: [],
    }

    SUBSETS.forEach((subset) => {
      result[subset] = ASPECTS.map((aspect) => {
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
    })

    return result
  }, [bestMetrics, selectedMethods])

  if (selectedMethods.length === 0) {
    return (
      <div className="fmax-small-multiples fmax-small-multiples--empty">
        <p>Select methods to view F-max comparison</p>
      </div>
    )
  }

  return (
    <div className="fmax-small-multiples">
      <h4 className="fmax-small-multiples__title">
        F-max by Knowledge Subset
        <InfoIcon tooltip="F-max is the maximum F₁ score across all prediction thresholds. Each chart shows performance for a different knowledge subset: NK (No Knowledge), LK (Limited Knowledge), PK (Partial Knowledge)." />
      </h4>

      <div className="fmax-small-multiples__grid">
        {SUBSETS.map((subset, index) => (
          <div key={subset} className="fmax-small-multiples__chart">
            <h5 className="fmax-small-multiples__chart-title">
              {SUBSET_LABELS[subset]}
            </h5>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={chartDataBySubset[subset]}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="aspect" tick={{ fontSize: 12 }} />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(2)}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    const aspectData = chartDataBySubset[subset].find(
                      (d) => d.aspect === label
                    )
                    return (
                      <div className="fmax-small-multiples__tooltip">
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
                {index === 0 && <Legend />}
                {selectedMethods.map((method) => (
                  <Bar
                    key={method}
                    dataKey={method}
                    fill={getColor(method)}
                    radius={[4, 4, 0, 0]}
                  >
                    <LabelList
                      dataKey={method}
                      position="top"
                      formatter={(value: number | null) =>
                        value !== null ? value.toFixed(2) : ''
                      }
                      style={{ fontSize: 10, fill: 'var(--isu-charcoal)' }}
                    />
                  </Bar>
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </div>
  )
}
