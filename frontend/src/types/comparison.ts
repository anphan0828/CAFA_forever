import type { BestMetricsMap, CurvesMap } from './evaluation'
import type { ReleaseMethods } from './method'
import type { ReleaseMeta } from './release'

export interface SelectedReleaseBundle {
  releaseId: string
  label: string
  meta: ReleaseMeta
  methods: ReleaseMethods
  best: BestMetricsMap
  curves?: CurvesMap | null
}

export interface ComparisonSummaryRow {
  release: string
  subset: string
  aspect: string
  method: string
  targetsPredicted: number
  groundTruthTargets: number
  targetCoveragePct: number
  precision: number
  recall: number
  fmax: number
  coverage: number
  threshold: number
}
