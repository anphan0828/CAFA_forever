import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { ComparisonMetric } from '../../hooks/useComparisonData'
import './ComparisonFacet.css'

interface ComparisonFacetProps {
  data: ComparisonMetric[]
  metric: 'fmax' | 'precision' | 'recall' | 'coverage'
  maxMethods?: number
}

export function ComparisonFacet({
  data,
  metric,
  maxMethods = 10,
}: ComparisonFacetProps) {
  // Limit methods for readability
  const displayData = data.slice(0, maxMethods)

  if (displayData.length === 0) {
    return (
      <div className="comparison-facet comparison-facet--empty">
        <span className="comparison-facet__empty-message">No data</span>
      </div>
    )
  }

  return (
    <div className="comparison-facet">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={displayData}
          margin={{ top: 10, right: 10, left: 5, bottom: 60 }}
          barCategoryGap="15%"
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            type="category"
            dataKey="method"
            tick={{ fontSize: 10 }}
            angle={-45}
            textAnchor="end"
            interval={0}
            height={55}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tick={{ fontSize: 10 }}
            tickFormatter={(v) => v.toFixed(1)}
            width={30}
          />
          <Tooltip
            formatter={(value) =>
              typeof value === 'number' ? value.toFixed(4) : 'N/A'
            }
            labelFormatter={(label) => `${label}`}
            contentStyle={{
              fontSize: '12px',
              backgroundColor: 'var(--isu-white)',
              border: '1px solid var(--isu-border)',
              padding: '6px 10px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            }}
            wrapperStyle={{ zIndex: 1000 }}
          />
          <Bar
            dataKey={`${metric}_A`}
            fill="var(--isu-cardinal)"
            radius={[3, 3, 0, 0]}
          />
          <Bar
            dataKey={`${metric}_B`}
            fill="var(--isu-gold)"
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
