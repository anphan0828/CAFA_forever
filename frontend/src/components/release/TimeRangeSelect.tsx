import { useState, useMemo, useEffect } from 'react'
import type { ReleaseEntry } from '../../types'
import { InfoIcon } from '../ui'
import './TimeRangeSelect.css'

interface TimeRangeSelectProps {
  releases: ReleaseEntry[]
  timepoints: string[]
  selectedRelease: string | null
  onSelectRelease: (releaseId: string | null) => void
}

// Format timepoint for display (e.g., "Dec_2025" -> "Dec 2025")
function formatTimepoint(tp: string): string {
  return tp.replace('_', ' ')
}

// Parse timepoint to sortable date
function parseTimepoint(tp: string): Date {
  const [month, year] = tp.split('_')
  const monthMap: Record<string, number> = {
    Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5,
    Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11
  }
  return new Date(parseInt(year), monthMap[month] || 0)
}

export function TimeRangeSelect({
  releases,
  timepoints,
  selectedRelease,
  onSelectRelease,
}: TimeRangeSelectProps) {
  // Parse selected release into start/end timepoints
  const parseReleaseId = (id: string | null): { start: string; end: string } | null => {
    if (!id) return null
    const parts = id.split('_')
    if (parts.length !== 4) return null
    return {
      start: `${parts[0]}_${parts[1]}`,
      end: `${parts[2]}_${parts[3]}`
    }
  }

  const parsed = parseReleaseId(selectedRelease)
  const [startTimepoint, setStartTimepoint] = useState<string>(parsed?.start || '')
  const [endTimepoint, setEndTimepoint] = useState<string>(parsed?.end || '')

  // Sort timepoints chronologically
  const sortedTimepoints = useMemo(() => {
    return [...timepoints].sort((a, b) =>
      parseTimepoint(a).getTime() - parseTimepoint(b).getTime()
    )
  }, [timepoints])

  // Build a set of valid release combinations
  const validReleases = useMemo(() => {
    const map = new Map<string, ReleaseEntry>()
    releases.forEach(r => map.set(r.id, r))
    return map
  }, [releases])

  // Get available end timepoints (must come after start)
  const availableEndTimepoints = useMemo(() => {
    if (!startTimepoint) return []
    const startDate = parseTimepoint(startTimepoint)
    return sortedTimepoints.filter(tp => parseTimepoint(tp) > startDate)
  }, [startTimepoint, sortedTimepoints])

  // Check if current selection is valid
  const currentReleaseId = startTimepoint && endTimepoint
    ? `${startTimepoint}_${endTimepoint}`
    : null
  const currentRelease = currentReleaseId ? validReleases.get(currentReleaseId) : null
  const isValidSelection = currentRelease?.status === 'ready'

  // Sync internal state with prop changes
  useEffect(() => {
    const newParsed = parseReleaseId(selectedRelease)
    if (newParsed) {
      setStartTimepoint(newParsed.start)
      setEndTimepoint(newParsed.end)
    }
  }, [selectedRelease])

  // Update parent when valid
  useEffect(() => {
    if (isValidSelection && currentReleaseId !== selectedRelease) {
      onSelectRelease(currentReleaseId)
    }
  }, [isValidSelection, currentReleaseId, selectedRelease, onSelectRelease])

  // Handle start change
  const handleStartChange = (value: string) => {
    setStartTimepoint(value)
    if (value) {
      const startDate = parseTimepoint(value)
      const endDate = endTimepoint ? parseTimepoint(endTimepoint) : null
      if (!endDate || endDate <= startDate) {
        setEndTimepoint('')
      }
    } else {
      setEndTimepoint('')
    }
  }

  return (
    <div className="time-range-select">
      <h3 className="time-range-select__title">
        Evaluation Window
        <InfoIcon tooltip="Select a time range to evaluate methods. Start = when predictions were made; End = when annotations are used as ground truth." />
      </h3>

      <div className="time-range-select__fields">
        <div className="time-range-select__field">
          <label htmlFor="sidebar-start" className="time-range-select__label">
            Start
          </label>
          <select
            id="sidebar-start"
            className="time-range-select__select"
            value={startTimepoint}
            onChange={(e) => handleStartChange(e.target.value)}
          >
            <option value="">Select...</option>
            {sortedTimepoints.slice(0, -1).map(tp => (
              <option key={tp} value={tp}>
                {formatTimepoint(tp)}
              </option>
            ))}
          </select>
        </div>

        <span className="time-range-select__separator">to</span>

        <div className="time-range-select__field">
          <label htmlFor="sidebar-end" className="time-range-select__label">
            End
          </label>
          <select
            id="sidebar-end"
            className="time-range-select__select"
            value={endTimepoint}
            onChange={(e) => setEndTimepoint(e.target.value)}
            disabled={!startTimepoint}
          >
            <option value="">Select...</option>
            {availableEndTimepoints.map(tp => {
              const releaseId = `${startTimepoint}_${tp}`
              const release = validReleases.get(releaseId)
              const isReady = release?.status === 'ready'
              return (
                <option key={tp} value={tp} disabled={!isReady}>
                  {formatTimepoint(tp)}{isReady ? '' : ' (N/A)'}
                </option>
              )
            })}
          </select>
        </div>
      </div>

      {startTimepoint && endTimepoint && !isValidSelection && (
        <div className="time-range-select__warning">
          No data for this range
        </div>
      )}
    </div>
  )
}
