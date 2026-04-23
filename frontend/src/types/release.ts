/**
 * Types for per-release data (meta.json)
 */

import type { Aspect } from './method'

export interface ReleaseDates {
  goaStart: string
  goaEnd: string
  uniprotStart: string
  uniprotEnd: string
}

export interface AspectCounts {
  [key: string]: number
}

export interface SubsetTargetCount {
  total: number
  byAspect: Partial<Record<Aspect, number>>
}

export interface TargetCounts {
  NK: SubsetTargetCount
  LK: SubsetTargetCount
  PK: SubsetTargetCount
}

export interface ReleaseMeta {
  releaseId: string
  startTimepoint: string
  endTimepoint: string
  dates: ReleaseDates
  targetCounts: TargetCounts
}
