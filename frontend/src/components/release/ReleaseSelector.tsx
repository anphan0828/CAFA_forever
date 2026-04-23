import { useState, useEffect } from 'react'
import type { ReleaseEntry, ReleaseMeta } from '../../types'
import { ReleaseCard } from './ReleaseCard'
import { Section } from '../layout'
import './ReleaseSelector.css'

interface ReleaseSelectorProps {
  releases: ReleaseEntry[]
  selectedRelease: string | null
  onSelectRelease: (releaseId: string) => void
  loadMeta?: (releaseId: string) => Promise<ReleaseMeta | null>
}

export function ReleaseSelector({
  releases,
  selectedRelease,
  onSelectRelease,
  loadMeta,
}: ReleaseSelectorProps) {
  const [metaCache, setMetaCache] = useState<Record<string, ReleaseMeta | null>>({})
  const [loadingMeta, setLoadingMeta] = useState<Set<string>>(new Set())

  // Load meta for visible releases
  useEffect(() => {
    if (!loadMeta) return

    const loadMetas = async () => {
      const toLoad = releases
        .filter((r) => r.status === 'ready' && !(r.id in metaCache) && !loadingMeta.has(r.id))
        .slice(0, 6) // Limit concurrent loads

      if (toLoad.length === 0) return

      setLoadingMeta((prev) => {
        const next = new Set(prev)
        toLoad.forEach((r) => next.add(r.id))
        return next
      })

      const results = await Promise.all(
        toLoad.map(async (r) => {
          try {
            const meta = await loadMeta(r.id)
            return { id: r.id, meta }
          } catch {
            return { id: r.id, meta: null }
          }
        })
      )

      setMetaCache((prev) => {
        const next = { ...prev }
        results.forEach(({ id, meta }) => {
          next[id] = meta
        })
        return next
      })

      setLoadingMeta((prev) => {
        const next = new Set(prev)
        results.forEach(({ id }) => next.delete(id))
        return next
      })
    }

    loadMetas()
  }, [releases, loadMeta, metaCache, loadingMeta])

  const readyReleases = releases.filter((r) => r.status === 'ready')
  const otherReleases = releases.filter((r) => r.status !== 'ready')

  return (
    <Section id="releases" title="Available Releases">
      <div className="release-selector">
        {readyReleases.length > 0 && (
          <div className="release-selector__grid">
            {readyReleases.map((release) => (
              <ReleaseCard
                key={release.id}
                release={release}
                meta={metaCache[release.id]}
                isSelected={selectedRelease === release.id}
                onClick={() => onSelectRelease(release.id)}
              />
            ))}
          </div>
        )}

        {readyReleases.length === 0 && (
          <p className="release-selector__empty">
            No releases are currently available. Please check back later.
          </p>
        )}

        {otherReleases.length > 0 && (
          <div className="release-selector__other">
            <h4>Unavailable Releases</h4>
            <ul>
              {otherReleases.map((release) => (
                <li key={release.id}>
                  {release.id} - {release.status}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Section>
  )
}
