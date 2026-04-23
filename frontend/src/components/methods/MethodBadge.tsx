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
  const handleClick = (e: React.MouseEvent) => {
    if (dockerUrl && (e.ctrlKey || e.metaKey)) {
      e.stopPropagation()
      e.preventDefault()
      window.open(dockerUrl, '_blank', 'noopener,noreferrer')
      return
    }
    onClick?.()
  }

  const badge = (
    <button
      className={`method-badge ${selected ? 'method-badge--selected' : ''} ${isBaseline ? 'method-badge--baseline' : ''}`}
      style={{ '--method-color': color } as React.CSSProperties}
      onClick={handleClick}
      aria-pressed={selected}
      title={dockerUrl ? 'Ctrl+click to open Docker container' : undefined}
    >
      <span className="method-badge__dot" aria-hidden="true" />
      <span className={`method-badge__label ${dockerUrl ? 'method-badge__label--link' : ''}`}>
        {label}
      </span>
      {isBaseline && <span className="method-badge__tag">Baseline</span>}
    </button>
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
