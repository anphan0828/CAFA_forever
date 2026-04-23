import { useState, useId, cloneElement, isValidElement, type ReactNode, type ReactElement } from 'react'
import './Tooltip.css'

interface TooltipProps {
  content: ReactNode
  children: ReactElement
  position?: 'top' | 'bottom' | 'left' | 'right'
}

export function Tooltip({ content, children, position = 'top' }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const tooltipId = useId()

  const childWithAria = isValidElement(children)
    ? cloneElement(children, {
        'aria-describedby': isVisible ? tooltipId : undefined,
      } as React.HTMLAttributes<HTMLElement>)
    : children

  return (
    <div
      className="tooltip-wrapper"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
      onFocus={() => setIsVisible(true)}
      onBlur={() => setIsVisible(false)}
    >
      {childWithAria}
      {isVisible && (
        <div id={tooltipId} className={`tooltip tooltip--${position}`} role="tooltip">
          {content}
        </div>
      )}
    </div>
  )
}
