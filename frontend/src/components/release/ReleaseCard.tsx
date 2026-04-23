import type { ReleaseEntry } from '../../types'
import type { ReleaseMeta } from '../../types'
import './ReleaseCard.css'

interface ReleaseCardProps {
  release: ReleaseEntry
  meta?: ReleaseMeta | null
  isSelected?: boolean
  onClick?: () => void
}

export function ReleaseCard({ release, meta, isSelected, onClick }: ReleaseCardProps) {
  const formatTimepoint = (tp: string) => tp.replace('_', ' ')

  const totalTargets = meta
    ? meta.targetCounts.NK.total + meta.targetCounts.LK.total + meta.targetCounts.PK.total
    : null

  return (
    <button
      className={`release-card ${isSelected ? 'release-card--selected' : ''}`}
      onClick={onClick}
      aria-pressed={isSelected}
    >
      <div className="release-card__header">
        <h3 className="release-card__title">
          {formatTimepoint(release.startTimepoint)} &rarr; {formatTimepoint(release.endTimepoint)}
        </h3>
        <span className={`release-card__status release-card__status--${release.status}`}>
          {release.status}
        </span>
      </div>

      {meta && (
        <div className="release-card__meta">
          <div className="release-card__dates">
            <div className="release-card__date">
              <span className="release-card__label">GOA:</span>
              <span>{meta.dates.goaStart} &rarr; {meta.dates.goaEnd}</span>
            </div>
          </div>

          <div className="release-card__counts">
            <div className="release-card__count">
              <span className="release-card__count-value">{meta.targetCounts.NK.total}</span>
              <span className="release-card__count-label">NK</span>
            </div>
            <div className="release-card__count">
              <span className="release-card__count-value">{meta.targetCounts.LK.total}</span>
              <span className="release-card__count-label">LK</span>
            </div>
            <div className="release-card__count">
              <span className="release-card__count-value">{meta.targetCounts.PK.total}</span>
              <span className="release-card__count-label">PK</span>
            </div>
          </div>

          {totalTargets && (
            <p className="release-card__total">
              {totalTargets.toLocaleString()} total targets
            </p>
          )}
        </div>
      )}

      {isSelected && (
        <div className="release-card__selected-indicator">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>
      )}
    </button>
  )
}
