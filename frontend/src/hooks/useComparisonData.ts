import { useMemo } from 'react'
import type { BestMetricsMap, Subset, Aspect } from '../types'
import { SUBSETS, ASPECTS, makeBestKey } from '../types'

export interface ComparisonMetric {
  method: string
  fmax_A: number | null
  fmax_B: number | null
  precision_A: number | null
  precision_B: number | null
  recall_A: number | null
  recall_B: number | null
  coverage_A: number | null
  coverage_B: number | null
}

export interface ComparisonData {
  /** Keyed by "NK_biological_process" format */
  data: Record<string, ComparisonMetric[]>
  /** Methods present in both releases */
  commonMethods: string[]
  /** Whether data is ready for display */
  ready: boolean
}

/**
 * Hook to compute comparison metrics between two releases.
 * Only includes methods that exist in both releases.
 */
export function useComparisonData(
  primaryBest: BestMetricsMap | null,
  secondaryBest: BestMetricsMap | null,
  primaryMethods: string[],
  secondaryMethods: string[]
): ComparisonData {
  return useMemo(() => {
    // Early return if data not ready
    if (!primaryBest || !secondaryBest) {
      return {
        data: {},
        commonMethods: [],
        ready: false,
      }
    }

    // Find methods common to both releases
    const primarySet = new Set(primaryMethods)
    const commonMethods = secondaryMethods.filter(m => primarySet.has(m))

    // Build comparison data for each subset × aspect
    const data: Record<string, ComparisonMetric[]> = {}

    SUBSETS.forEach((subset: Subset) => {
      ASPECTS.forEach((aspect: Aspect) => {
        const key = makeBestKey(subset, aspect)
        const primaryMetrics = primaryBest[key] || []
        const secondaryMetrics = secondaryBest[key] || []

        // Create lookup maps for faster access
        const primaryMap = new Map(primaryMetrics.map(m => [m.method, m]))
        const secondaryMap = new Map(secondaryMetrics.map(m => [m.method, m]))

        // Build comparison metrics for common methods
        const comparisonMetrics: ComparisonMetric[] = commonMethods
          .map(method => {
            const primary = primaryMap.get(method)
            const secondary = secondaryMap.get(method)

            return {
              method,
              fmax_A: primary?.fmax ?? null,
              fmax_B: secondary?.fmax ?? null,
              precision_A: primary?.precision ?? null,
              precision_B: secondary?.precision ?? null,
              recall_A: primary?.recall ?? null,
              recall_B: secondary?.recall ?? null,
              coverage_A: primary?.coverage ?? null,
              coverage_B: secondary?.coverage ?? null,
            }
          })
          // Sort by primary fmax descending, then by method name
          .sort((a, b) => {
            const fmaxA = a.fmax_A ?? 0
            const fmaxB = b.fmax_A ?? 0
            if (fmaxB !== fmaxA) return fmaxB - fmaxA
            return a.method.localeCompare(b.method)
          })

        data[key] = comparisonMetrics
      })
    })

    return {
      data,
      commonMethods,
      ready: true,
    }
  }, [primaryBest, secondaryBest, primaryMethods, secondaryMethods])
}
