import { Tooltip } from './Tooltip'
import './InfoIcon.css'

interface InfoIconProps {
  tooltip: string
  size?: number
}

export function InfoIcon({ tooltip, size = 14 }: InfoIconProps) {
  return (
    <Tooltip content={tooltip} position="top">
      <span className="info-icon" aria-label="More information">
        <svg
          width={size}
          height={size}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4" />
          <path d="M12 8h.01" />
        </svg>
      </span>
    </Tooltip>
  )
}
