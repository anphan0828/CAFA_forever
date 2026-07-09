import { useMemo, useState } from 'react'
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

type DetailFields = Record<string, string>

function parseDetailFields(description?: string): DetailFields {
  const text = description?.trim()
  if (!text) return {}

  const matches = Array.from(text.matchAll(/\b(Summary|Input|Publication|Documentation):/g))
  if (matches.length === 0) return { Summary: text }

  return matches.reduce<DetailFields>((fields, match, index) => {
    const fieldName = match[1]
    const fieldStart = match.index + match[0].length
    const fieldEnd = matches[index + 1]?.index ?? text.length
    fields[fieldName] = text.slice(fieldStart, fieldEnd).trim()
    return fields
  }, {})
}

function firstUrl(text?: string): string | null {
  const match = text?.match(/https?:\/\/\S+/)
  return match ? match[0].replace(/[.,);]+$/, '') : null
}

function stripUrl(text: string): string {
  return text.replace(/https?:\/\/\S+/, '').trim()
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
  const [expanded, setExpanded] = useState(false)
  const fields = useMemo(() => parseDetailFields(description), [description])
  const documentationUrl = firstUrl(fields.Documentation)
  const publicationUrl = firstUrl(fields.Publication)
  const hasDetails = Object.keys(fields).length > 0 || dockerUrl

  return (
    <div className="method-badge-row" style={{ '--method-color': color } as React.CSSProperties}>
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

        <button
          type="button"
          className="method-badge__details-toggle"
          aria-expanded={expanded}
          aria-label={`${expanded ? 'Hide' : 'Show'} details for ${label}`}
          disabled={!hasDetails}
          onClick={(event) => {
            event.stopPropagation()
            setExpanded((value) => !value)
          }}
        >
          <svg
            className="method-badge__chevron"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="m6 9 6 6 6-6" />
          </svg>
        </button>
      </div>

      {expanded && hasDetails && (
        <div className="method-badge__details">
          {fields.Summary && (
            <p>
              <strong>Summary:</strong> {fields.Summary}
            </p>
          )}
          {fields.Input && (
            <p>
              <strong>Input:</strong> {fields.Input}
            </p>
          )}
          {fields.Publication && !publicationUrl && (
            <p>
              <strong>Publication:</strong> {fields.Publication}
            </p>
          )}
          {fields.Documentation && !documentationUrl && (
            <p>
              <strong>Documentation:</strong> {fields.Documentation}
            </p>
          )}

          <div className="method-badge__detail-links">
            {publicationUrl && (
              <a href={publicationUrl} target="_blank" rel="noreferrer">
                Publication
              </a>
            )}
            {documentationUrl && (
              <a href={documentationUrl} target="_blank" rel="noreferrer">
                Documentation
              </a>
            )}
            {dockerUrl && (
              <a href={dockerUrl} target="_blank" rel="noreferrer">
                Container image
              </a>
            )}
          </div>

          {documentationUrl && stripUrl(fields.Documentation || '') && (
            <p className="method-badge__detail-note">{stripUrl(fields.Documentation || '')}</p>
          )}
        </div>
      )}
    </div>
  )
}
