import type { ReactNode } from 'react'
import './Section.css'

interface SectionProps {
  id?: string
  title?: string
  description?: string
  children: ReactNode
  variant?: 'default' | 'alternate' | 'hero'
  className?: string
}

export function Section({
  id,
  title,
  description,
  children,
  variant = 'default',
  className = '',
}: SectionProps) {
  return (
    <section
      id={id}
      className={`section section--${variant} ${className}`}
      aria-labelledby={title && id ? `${id}-title` : undefined}
    >
      <div className="section__inner">
        {title && (
          <header className="section__header">
            <h2 id={id ? `${id}-title` : undefined} className="section__title">
              {title}
            </h2>
            {description && (
              <p className="section__description">{description}</p>
            )}
          </header>
        )}
        <div className="section__content">
          {children}
        </div>
      </div>
    </section>
  )
}
