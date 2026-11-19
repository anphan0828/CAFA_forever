import { useState, useEffect, useCallback, useRef } from 'react'
import type {
  ReleaseMeta,
  ReleaseMethods,
  BestMetricsMap,
  CurvesMap,
  MethodsConfig,
} from '../types'
import {
  validateBestMetrics,
  validateCurvesMap,
  validateMethodsConfig,
  validateReleaseMeta,
  validateReleaseMethods,
} from '../lib/dataValidation'

interface ReleaseData {
  meta: ReleaseMeta | null
  methods: ReleaseMethods | null
  best: BestMetricsMap | null
  curves: CurvesMap | null
}

interface UseReleaseDataResult {
  data: ReleaseData
  loading: boolean
  error: string | null
  loadCurves: () => Promise<void>
  curvesLoading: boolean
}

/**
 * Hook to load release-specific data
 * Loads meta, methods, and best metrics immediately
 * Curves are loaded on demand via loadCurves() due to their size
 */
interface ReleaseState {
  releaseId: string | null
  data: ReleaseData
  loading: boolean
  curvesLoading: boolean
  error: string | null
}

function emptyRelease(releaseId: string | null): ReleaseState {
  return {
    releaseId,
    data: { meta: null, methods: null, best: null, curves: null },
    loading: releaseId !== null,
    curvesLoading: false,
    error: null,
  }
}

export function useReleaseData(releaseId: string | null): UseReleaseDataResult {
  const [state, setState] = useState(() => emptyRelease(releaseId))
  const activeRequest = useRef<{
    releaseId: string
    controller: AbortController
    curvesPromise: Promise<void> | null
  } | null>(null)

  useEffect(() => {
    setState(emptyRelease(releaseId))
    if (!releaseId) return

    const request = { releaseId, controller: new AbortController(), curvesPromise: null as Promise<void> | null }
    activeRequest.current = request
    const { signal } = request.controller

    async function fetchData() {
      try {
        const basePath = `/data/releases/${releaseId}`
        const [metaRes, methodsRes, bestRes] = await Promise.all([
          fetch(`${basePath}/meta.json`, { cache: 'no-store', signal }),
          fetch(`${basePath}/methods.json`, { cache: 'no-store', signal }),
          fetch(`${basePath}/best.json`, { cache: 'no-store', signal }),
        ])
        if (!metaRes.ok || !methodsRes.ok || !bestRes.ok) {
          throw new Error('Failed to load release data')
        }
        const [metaRaw, methodsRaw, bestRaw] = await Promise.all([
          metaRes.json(), methodsRes.json(), bestRes.json(),
        ])
        const meta = validateReleaseMeta(metaRaw)
        const methods = validateReleaseMethods(methodsRaw)
        const best = validateBestMetrics(bestRaw)
        if (!signal.aborted) {
          setState(prev => signal.aborted ? prev : {
            ...prev, data: { ...prev.data, meta, methods, best }, loading: false,
          })
        }
      } catch (err) {
        if (!signal.aborted) {
          setState(prev => signal.aborted ? prev : {
            ...prev, loading: false, error: err instanceof Error ? err.message : 'Unknown error',
          })
        }
      }
    }

    void fetchData()
    return () => {
      request.controller.abort()
      if (activeRequest.current === request) activeRequest.current = null
    }
  }, [releaseId])

  // One lazy request per selection. Cleanup also invalidates responses whose
  // body parsing finishes after abort, including an A -> B -> A selection.
  const loadCurves = useCallback(async () => {
    const request = activeRequest.current
    if (!request || request.releaseId !== releaseId) return
    if (request.curvesPromise) return request.curvesPromise
    const { signal } = request.controller
    setState(prev => ({ ...prev, curvesLoading: true }))
    request.curvesPromise = (async () => {
      try {
        const response = await fetch(`/data/releases/${releaseId}/curves.json`, { cache: 'no-store', signal })
        if (!response.ok) throw new Error('Failed to load curves data')
        const curves = validateCurvesMap(await response.json())
        if (!signal.aborted) {
          setState(prev => signal.aborted ? prev : { ...prev, data: { ...prev.data, curves } })
        }
      } catch (err) {
        if (!signal.aborted) {
          setState(prev => signal.aborted ? prev : {
            ...prev, error: err instanceof Error ? err.message : 'Failed to load curves',
          })
        }
      } finally {
        if (!signal.aborted) {
          setState(prev => signal.aborted ? prev : { ...prev, curvesLoading: false })
        }
      }
    })()
    return request.curvesPromise
  }, [releaseId])

  // Hide the previous release immediately, before the new selection's effect
  // runs, so a render can never label old data with the newly selected window.
  const current = state.releaseId === releaseId ? state : emptyRelease(releaseId)
  return { data: current.data, loading: current.loading, error: current.error, loadCurves, curvesLoading: current.curvesLoading }
}

/**
 * Hook to load global methods configuration
 */
export function useMethodsConfig() {
  const [config, setConfig] = useState<MethodsConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function fetchConfig() {
      try {
        const response = await fetch('/data/methods.json', { cache: 'no-store' })
        if (!response.ok) {
          throw new Error('Failed to load methods config')
        }
        const data = validateMethodsConfig(await response.json())
        if (!cancelled) {
          setConfig(data)
          setLoading(false)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unknown error')
          setLoading(false)
        }
      }
    }

    fetchConfig()

    return () => {
      cancelled = true
    }
  }, [])

  return { config, loading, error }
}
