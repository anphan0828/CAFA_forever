import { useMemo } from 'react'
import type { MethodConfig, ReleaseMethods } from '../types'
import { SUBSETS } from '../types'

function sortMethods(methods: string[], methodConfigs: Record<string, MethodConfig>): string[] {
  return [...methods].sort((a, b) => {
    const aBaseline = methodConfigs[a]?.isBaseline ?? false
    const bBaseline = methodConfigs[b]?.isBaseline ?? false
    if (aBaseline !== bBaseline) return bBaseline ? 1 : -1
    return a.localeCompare(b)
  })
}

export function useComparableMethods(
  releaseMethods: Array<ReleaseMethods | null | undefined>,
  methodConfigs: Record<string, MethodConfig> | null | undefined,
  compareMode: boolean
): string[] {
  return useMemo(() => {
    const loaded = releaseMethods.filter((methods): methods is ReleaseMethods => Boolean(methods))
    if (loaded.length === 0 || !methodConfigs) return []

    if (!compareMode) {
      return sortMethods(Object.keys(loaded[0].methods), methodConfigs)
    }

    const [first, ...rest] = loaded
    const comparable = Object.entries(first.methods)
      .filter(([, method]) => SUBSETS.every((subset) => method.availability[subset]))
      .map(([name]) => name)
      .filter((name) =>
        rest.every((release) => {
          const method = release.methods[name]
          return Boolean(method) && SUBSETS.every((subset) => method.availability[subset])
        })
      )

    return sortMethods(comparable, methodConfigs)
  }, [releaseMethods, methodConfigs, compareMode])
}
