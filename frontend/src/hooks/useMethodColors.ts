import { useMemo } from 'react'

const METHOD_COLORS = [
  '#0072B2', // Blue
  '#D55E00', // Vermillion
  '#009E73', // Bluish green
  '#CC79A7', // Reddish purple
  '#E69F00', // Orange
  '#56B4E9', // Sky blue
  '#B58900', // Gold
  '#332288', // Indigo
]

const EXTENDED_COLORS = [
  '#117733',
  '#88CCEE',
  '#AA4499',
  '#44AA99',
  '#882255',
  '#999933',
  '#CC6677',
  '#6699CC',
]

const CANONICAL_METHOD_ORDER = [
  'BLAST',
  'GOA Non-exp',
  'Naive',
  'ProtT5',
  'DeepGOPlus',
  'FunBind',
  'TransFew',
]

interface UseMethodColorsResult {
  getColor: (method: string) => string
  colorMap: Map<string, string>
}

/**
 * Hook to get consistent colors for methods
 * Colors are assigned based on method order for consistency across views
 */
export function useMethodColors(methods: string[]): UseMethodColorsResult {
  const colorMap = useMemo(() => {
    const map = new Map<string, string>()
    const allColors = [...METHOD_COLORS, ...EXTENDED_COLORS]
    const orderedMethods = [...methods].sort((a, b) => {
      const aIndex = CANONICAL_METHOD_ORDER.indexOf(a)
      const bIndex = CANONICAL_METHOD_ORDER.indexOf(b)

      if (aIndex !== -1 || bIndex !== -1) {
        if (aIndex === -1) return 1
        if (bIndex === -1) return -1
        return aIndex - bIndex
      }

      return a.localeCompare(b)
    })

    orderedMethods.forEach((method, index) => {
      map.set(method, allColors[index % allColors.length])
    })

    return map
  }, [methods])

  const getColor = (method: string): string => {
    return colorMap.get(method) ?? METHOD_COLORS[0]
  }

  return { getColor, colorMap }
}

/**
 * Get a consistent color for a specific method by name
 * Useful when you don't have the full methods list
 */
export function getMethodColor(method: string, allMethods: string[]): string {
  const orderedMethods = [...allMethods].sort((a, b) => {
    const aIndex = CANONICAL_METHOD_ORDER.indexOf(a)
    const bIndex = CANONICAL_METHOD_ORDER.indexOf(b)

    if (aIndex !== -1 || bIndex !== -1) {
      if (aIndex === -1) return 1
      if (bIndex === -1) return -1
      return aIndex - bIndex
    }

    return a.localeCompare(b)
  })
  const index = orderedMethods.indexOf(method)
  if (index === -1) return METHOD_COLORS[0]

  const allColors = [...METHOD_COLORS, ...EXTENDED_COLORS]
  return allColors[index % allColors.length]
}
