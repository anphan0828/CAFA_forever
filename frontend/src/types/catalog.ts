/**
 * Types for catalog.json - global release listing
 */

export interface ReleaseEntry {
  id: string
  startTimepoint: string
  endTimepoint: string
  status: 'ready' | 'pending' | 'invalid'
}

export interface TimepointDates {
  goa: string
  uniprot: string
}

export interface Catalog {
  releases: ReleaseEntry[]
  invalidReleases?: Record<string, string[]>
  timepoints: string[]
  timepointDates: Record<string, TimepointDates>
  generatedAt: string
}
