/**
 * Types for method configuration and per-release method data
 */

export type Subset = 'NK' | 'LK' | 'PK'
export type Aspect = 'biological_process' | 'molecular_function' | 'cellular_component'

export const MAX_SELECTED_METHODS = 30
export const SUBSETS: Subset[] = ['NK', 'LK', 'PK']
export const ASPECTS: Aspect[] = ['biological_process', 'molecular_function', 'cellular_component']

export const ASPECT_LABELS: Record<Aspect, string> = {
  biological_process: 'Biological Process',
  molecular_function: 'Molecular Function',
  cellular_component: 'Cellular Component',
}

export const ASPECT_SHORT: Record<Aspect, string> = {
  biological_process: 'BP',
  molecular_function: 'MF',
  cellular_component: 'CC',
}

export const SUBSET_LABELS: Record<Subset, string> = {
  NK: 'No Knowledge',
  LK: 'Limited Knowledge',
  PK: 'Partial Knowledge',
}

/**
 * Global method configuration from methods.json
 */
export interface MethodConfig {
  label: string
  description: string
  dockerUrl: string
  isBaseline: boolean
}

export interface MethodsConfig {
  methods: Record<string, MethodConfig>
  aspects: Record<Aspect, string>
  subsets: Record<Subset, string>
}

/**
 * Per-release method availability from releases/{id}/methods.json
 */
export interface ReleaseMethod {
  filename: string
  label: string
  group: string
  availability: Record<Subset, boolean>
}

export interface ReleaseMethods {
  methods: Record<string, ReleaseMethod>
}
