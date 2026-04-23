/**
 * Types for catalog.json - global release listing
 */

export interface ReleaseEntry {
  id: string
  startTimepoint: string
  endTimepoint: string
  status: 'ready' | 'pending' | 'invalid'
}

export interface Catalog {
  releases: ReleaseEntry[]
  timepoints: string[]
  generatedAt: string
}
