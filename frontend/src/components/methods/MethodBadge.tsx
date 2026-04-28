import { Tooltip } from '../ui'
import './MethodBadge.css'

interface MethodBadgeProps {
  label: string
  color: string
  isBaseline?: boolean
  description?: string
  dockerUrl?: string
  selected?: boolean
  onClick?: () => void
}

export function MethodBadge({
  label,
  color,
  isBaseline,
  description,
  dockerUrl,
  selected,
  onClick,
}: MethodBadgeProps) {
  const badge = (
    <div className="method-badge-row" style={{ '--method-color': color } as React.CSSProperties}>
      {dockerUrl && (
        <a
          className="method-badge__link"
          href={dockerUrl}
          target="_blank"
          rel="noreferrer"
          aria-label={`Open ${label} method container`}
          onClick={(event) => event.stopPropagation()}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
            <path
              d="M14 3h7v7m0-7-9 9m7 2v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </a>
      )}
      {!dockerUrl && <span className="method-badge__link-spacer" aria-hidden="true" />}
      <div
        className={`method-badge ${selected ? 'method-badge--selected' : ''} ${isBaseline ? 'method-badge--baseline' : ''}`}
      >
        <label className="method-badge__choice">
          <input
            className="method-badge__checkbox"
            type="checkbox"
            checked={selected ?? false}
            onChange={onClick}
          />
          <span className="method-badge__text">
            <span className="method-badge__label">{label}</span>
            {isBaseline && <span className="method-badge__tag">Baseline</span>}
          </span>
        </label>
      </div>
    </div>
  )

  if (description) {
    return (
      <Tooltip content={description} position="top">
        {badge}
      </Tooltip>
    )
  }

  return badge
}
