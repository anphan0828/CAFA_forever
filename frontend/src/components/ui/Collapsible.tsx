import { useState, type ReactNode } from 'react'
import './Collapsible.css'

interface CollapsibleProps {
  title: string
  children: ReactNode
  defaultOpen?: boolean
  className?: string
}

export function Collapsible({
  title,
  children,
  defaultOpen = false,
  className = '',
}: CollapsibleProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className={`collapsible ${isOpen ? 'collapsible--open' : ''} ${className}`}>
      <button
        className="collapsible__trigger"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <span className="collapsible__title">{title}</span>
        <svg
          className="collapsible__icon"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      <div className="collapsible__content" hidden={!isOpen}>
        <div className="collapsible__inner">
          {children}
        </div>
      </div>
    </div>
  )
}
