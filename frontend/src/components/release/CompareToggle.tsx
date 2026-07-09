import type { ReleaseEntry } from '../../types'
import { TimelineSelector } from './TimelineSelector'
import './CompareToggle.css'

interface CompareToggleProps {
  enabled: boolean
  onToggle: (enabled: boolean) => void
  releases: ReleaseEntry[]
  timepoints: string[]
  secondaryRelease: string | null
  onSelectSecondaryRelease: (releaseId: string | null) => void
}

export function CompareToggle({
  enabled,
  onToggle,
  releases,
  timepoints,
  secondaryRelease,
  onSelectSecondaryRelease,
}: CompareToggleProps) {
  return (
    <div className="compare-toggle">
      <label className="compare-toggle__checkbox">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
        />
        <span className="compare-toggle__label">
          Compare with another window
        </span>
      </label>

      {enabled && (
        <div className="compare-toggle__secondary">
          <TimelineSelector
            releases={releases}
            timepoints={timepoints}
            selectedRelease={secondaryRelease}
            onSelectRelease={onSelectSecondaryRelease}
            label="Comparison Window"
          />
        </div>
      )}
    </div>
  )
}
