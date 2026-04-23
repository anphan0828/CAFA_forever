import { useMemo } from 'react'

/**
 * Color palette for methods - designed for colorblind accessibility
 * Based on Paul Tol's colorblind-friendly palette
 */
const METHOD_COLORS = [
  '#4477AA', // Blue
  '#EE6677', // Red/Pink
  '#228833', // Green
  '#CCBB44', // Yellow
  '#66CCEE', // Cyan
  '#AA3377', // Purple
  '#BBBBBB', // Grey
  '#000000', // Black (for additional methods)
]

/**
 * Additional colors if we need more than 8
 */
const EXTENDED_COLORS = [
  '#332288', // Indigo
  '#88CCEE', // Light Blue
  '#44AA99', // Teal
  '#117733', // Dark Green
  '#999933', // Olive
  '#DDCC77', // Sand
  '#CC6677', // Rose
  '#882255', // Wine
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

    methods.forEach((method, index) => {
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
  const index = allMethods.indexOf(method)
  if (index === -1) return METHOD_COLORS[0]

  const allColors = [...METHOD_COLORS, ...EXTENDED_COLORS]
  return allColors[index % allColors.length]
}
