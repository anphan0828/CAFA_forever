/**
 * Types for evaluation data (best.json, curves.json)
 */

import type { Subset, Aspect } from './method'

/**
 * Best metrics for a single method in a specific subset/aspect
 */
export interface BestMetrics {
  method: string
  subset: Subset
  aspect: Aspect
  precision: number
  recall: number
  fmax: number
  coverage: number
  threshold: number
  n: number
}

/**
 * Best metrics organized by subset_aspect key
 * e.g., "NK_biological_process" -> BestMetrics[]
 */
export type BestMetricsMap = Record<string, BestMetrics[]>

/**
 * Single point on a PR curve
 */
export interface CurvePoint {
  tau: number       // threshold
  precision: number
  recall: number
}

/**
 * PR curves organized by subset_aspect_method key
 * e.g., "NK_biological_process_BLAST" -> CurvePoint[]
 */
export type CurvesMap = Record<string, CurvePoint[]>

/**
 * Parsed curve key components
 */
export interface CurveKey {
  subset: Subset
  aspect: Aspect
  method: string
}

/**
 * Helper to parse a curve key string
 */
export function parseCurveKey(key: string): CurveKey | null {
  const parts = key.split('_')
  if (parts.length < 3) return null

  const subset = parts[0] as Subset
  const aspect = parts[1] as Aspect
  const method = parts.slice(2).join('_')

  return { subset, aspect, method }
}

/**
 * Helper to create a curve key string
 */
export function makeCurveKey(subset: Subset, aspect: Aspect, method: string): string {
  return `${subset}_${aspect}_${method}`
}

/**
 * Helper to create a best metrics key string
 */
export function makeBestKey(subset: Subset, aspect: Aspect): string {
  return `${subset}_${aspect}`
}
