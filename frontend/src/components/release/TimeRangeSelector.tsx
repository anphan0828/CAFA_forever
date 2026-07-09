import { useState, useMemo, useEffect } from 'react'
import type { ReleaseEntry } from '../../types'
import { Section } from '../layout'
import './TimeRangeSelector.css'

interface TimeRangeSelectorProps {
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

export function TimeRangeSelector({
  releases,
  timepoints,
  selectedRelease,
  onSelectRelease,
}: TimeRangeSelectorProps) {
  // Parse selected release into start/end timepoints
  const parseReleaseId = (id: string | null): { start: string; end: string } | null => {
    if (!id) return null
    // Release ID format: "Dec_2025_Mar_2026" -> start: "Dec_2025", end: "Mar_2026"
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

  // Build a set of valid release combinations for quick lookup
  const validReleases = useMemo(() => {
    const map = new Map<string, ReleaseEntry>()
    releases.forEach(r => {
      map.set(r.id, r)
    })
    return map
  }, [releases])

  // Get available end timepoints (must come after start)
  const availableEndTimepoints = useMemo(() => {
    if (!startTimepoint) return []
    const startDate = parseTimepoint(startTimepoint)
    return sortedTimepoints.filter(tp => {
      const tpDate = parseTimepoint(tp)
      return tpDate > startDate
    })
  }, [startTimepoint, sortedTimepoints])

  // Check if current selection is a valid release
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

  // Update parent when both timepoints are selected and valid
  useEffect(() => {
    if (isValidSelection && currentReleaseId !== selectedRelease) {
      onSelectRelease(currentReleaseId)
    }
  }, [isValidSelection, currentReleaseId, selectedRelease, onSelectRelease])

  // Handle start change - reset end if it's no longer valid
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

  // Count ready releases
  const readyCount = releases.filter(r => r.status === 'ready').length

  return (
    <Section id="releases" title="Select Evaluation Window">
      <div className="time-range-selector">
        <p className="time-range-selector__description">
          Choose the start and end timepoints to define the evaluation window.
          {readyCount > 0 && (
            <span className="time-range-selector__count">
              {' '}{readyCount} evaluation{readyCount !== 1 ? 's' : ''} available
            </span>
          )}
        </p>

        <div className="time-range-selector__controls">
          <div className="time-range-selector__field">
            <label htmlFor="start-timepoint" className="time-range-selector__label">
              Start Timepoint
            </label>
            <select
              id="start-timepoint"
              className="time-range-selector__select"
              value={startTimepoint}
              onChange={(e) => handleStartChange(e.target.value)}
            >
              <option value="">Select start...</option>
              {sortedTimepoints.slice(0, -1).map(tp => (
                <option key={tp} value={tp}>
                  {formatTimepoint(tp)}
                </option>
              ))}
            </select>
          </div>

          <div className="time-range-selector__arrow">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </div>

          <div className="time-range-selector__field">
            <label htmlFor="end-timepoint" className="time-range-selector__label">
              End Timepoint
            </label>
            <select
              id="end-timepoint"
              className="time-range-selector__select"
              value={endTimepoint}
              onChange={(e) => setEndTimepoint(e.target.value)}
              disabled={!startTimepoint}
            >
              <option value="">Select end...</option>
              {availableEndTimepoints.map(tp => {
                const releaseId = `${startTimepoint}_${tp}`
                const release = validReleases.get(releaseId)
                const isReady = release?.status === 'ready'
                return (
                  <option
                    key={tp}
                    value={tp}
                    disabled={!isReady}
                  >
                    {formatTimepoint(tp)}{isReady ? '' : ' (no data)'}
                  </option>
                )
              })}
            </select>
          </div>
        </div>

        {/* Status message */}
        {startTimepoint && endTimepoint && (
          <div className={`time-range-selector__status ${isValidSelection ? 'time-range-selector__status--valid' : 'time-range-selector__status--invalid'}`}>
            {isValidSelection ? (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
                <span>
                  Evaluation: <strong>{formatTimepoint(startTimepoint)}</strong> to <strong>{formatTimepoint(endTimepoint)}</strong>
                </span>
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 8v4M12 16h.01" />
                </svg>
                <span>No evaluation data available for this time range</span>
              </>
            )}
          </div>
        )}

        {/* Quick select presets for recent evaluations */}
        {releases.filter(r => r.status === 'ready').length > 0 && (
          <div className="time-range-selector__presets">
            <span className="time-range-selector__presets-label">Quick select:</span>
            <div className="time-range-selector__presets-list">
              {releases
                .filter(r => r.status === 'ready')
                .slice(0, 3)
                .map(release => (
                  <button
                    key={release.id}
                    type="button"
                    className={`time-range-selector__preset ${selectedRelease === release.id ? 'time-range-selector__preset--active' : ''}`}
                    onClick={() => {
                      const p = parseReleaseId(release.id)
                      if (p) {
                        setStartTimepoint(p.start)
                        setEndTimepoint(p.end)
                        onSelectRelease(release.id)
                      }
                    }}
                  >
                    {formatTimepoint(release.startTimepoint)} → {formatTimepoint(release.endTimepoint)}
                  </button>
                ))}
            </div>
          </div>
        )}
      </div>
    </Section>
  )
}
