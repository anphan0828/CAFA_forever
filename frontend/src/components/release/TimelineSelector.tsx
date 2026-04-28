import { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import { ComposedChart, XAxis, YAxis, ReferenceLine, ReferenceArea, ResponsiveContainer } from 'recharts'
import type { ReleaseEntry } from '../../types'
import { InfoIcon } from '../ui'
import './TimelineSelector.css'

interface TimelineSelectorProps {
  releases: ReleaseEntry[]
  timepoints: string[]
  selectedRelease: string | null
  onSelectRelease: (releaseId: string | null) => void
  label?: string
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

// Parse release ID into start/end timepoints
function parseReleaseId(id: string | null): { start: string; end: string } | null {
  if (!id) return null
  const parts = id.split('_')
  if (parts.length !== 4) return null
  return {
    start: `${parts[0]}_${parts[1]}`,
    end: `${parts[2]}_${parts[3]}`
  }
}

export function TimelineSelector({
  releases,
  timepoints,
  selectedRelease,
  onSelectRelease,
  label = 'Evaluation Window',
}: TimelineSelectorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [activeHandle, setActiveHandle] = useState<'start' | 'end' | null>(null)

  const parsed = parseReleaseId(selectedRelease)
  const [startTimepoint, setStartTimepoint] = useState<string>(parsed?.start || '')
  const [endTimepoint, setEndTimepoint] = useState<string>(parsed?.end || '')

  // Sort timepoints chronologically
  const sortedTimepoints = useMemo(() => {
    return [...timepoints].sort((a, b) =>
      parseTimepoint(a).getTime() - parseTimepoint(b).getTime()
    )
  }, [timepoints])

  // Map timepoints to x-positions (0-100 scale)
  const timepointPositions = useMemo(() => {
    const positions = new Map<string, number>()
    sortedTimepoints.forEach((tp, index) => {
      positions.set(tp, (index / (sortedTimepoints.length - 1)) * 100)
    })
    return positions
  }, [sortedTimepoints])

  // Build a set of valid release combinations
  const validReleases = useMemo(() => {
    const map = new Map<string, ReleaseEntry>()
    releases.forEach(r => map.set(r.id, r))
    return map
  }, [releases])

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

  // Find nearest timepoint to an x position
  const findNearestTimepoint = useCallback((xPosition: number): string => {
    let nearest = sortedTimepoints[0]
    let minDist = Infinity

    sortedTimepoints.forEach((tp) => {
      const tpPos = timepointPositions.get(tp) || 0
      const dist = Math.abs(xPosition - tpPos)
      if (dist < minDist) {
        minDist = dist
        nearest = tp
      }
    })

    return nearest
  }, [sortedTimepoints, timepointPositions])

  // Convert mouse position to x scale position
  const getXPositionFromMouse = useCallback((clientX: number): number => {
    if (!containerRef.current) return 0
    const rect = containerRef.current.getBoundingClientRect()
    const chartLeft = rect.left + 36 // Account for left margin
    const chartWidth = rect.width - 72 // Account for both margins
    const relativeX = clientX - chartLeft
    return Math.max(0, Math.min(100, (relativeX / chartWidth) * 100))
  }, [])

  // Handle mouse events
  const handleMouseDown = useCallback((handle: 'start' | 'end') => {
    setIsDragging(true)
    setActiveHandle(handle)
  }, [])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !activeHandle) return

    const xPosition = getXPositionFromMouse(e.clientX)
    const nearestTp = findNearestTimepoint(xPosition)
    const nearestDate = parseTimepoint(nearestTp)

    if (activeHandle === 'start') {
      const endDate = endTimepoint ? parseTimepoint(endTimepoint) : null
      if (!endDate || nearestDate < endDate) {
        setStartTimepoint(nearestTp)
      }
    } else {
      const startDate = startTimepoint ? parseTimepoint(startTimepoint) : null
      if (!startDate || nearestDate > startDate) {
        setEndTimepoint(nearestTp)
      }
    }
  }, [isDragging, activeHandle, getXPositionFromMouse, findNearestTimepoint, startTimepoint, endTimepoint])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
    setActiveHandle(null)
  }, [])

  // Global mouse up listener
  useEffect(() => {
    if (isDragging) {
      const handleGlobalMouseUp = () => {
        setIsDragging(false)
        setActiveHandle(null)
      }
      window.addEventListener('mouseup', handleGlobalMouseUp)
      return () => window.removeEventListener('mouseup', handleGlobalMouseUp)
    }
  }, [isDragging])

  // Click on timeline to set nearest handle
  const handleTimelineClick = useCallback((e: React.MouseEvent) => {
    if (isDragging) return

    const xPosition = getXPositionFromMouse(e.clientX)
    const nearestTp = findNearestTimepoint(xPosition)
    const nearestDate = parseTimepoint(nearestTp)

    // Determine which handle to move based on proximity
    const startPos = timepointPositions.get(startTimepoint) || 0
    const endPos = timepointPositions.get(endTimepoint) || 100

    if (!startTimepoint && !endTimepoint) {
      setStartTimepoint(nearestTp)
    } else if (!startTimepoint) {
      if (nearestDate < parseTimepoint(endTimepoint)) {
        setStartTimepoint(nearestTp)
      }
    } else if (!endTimepoint) {
      if (nearestDate > parseTimepoint(startTimepoint)) {
        setEndTimepoint(nearestTp)
      }
    } else {
      // Both set - move the closer one
      const distToStart = Math.abs(xPosition - startPos)
      const distToEnd = Math.abs(xPosition - endPos)

      if (distToStart < distToEnd) {
        if (nearestDate < parseTimepoint(endTimepoint)) {
          setStartTimepoint(nearestTp)
        }
      } else {
        if (nearestDate > parseTimepoint(startTimepoint)) {
          setEndTimepoint(nearestTp)
        }
      }
    }
  }, [isDragging, getXPositionFromMouse, findNearestTimepoint, startTimepoint, endTimepoint, timepointPositions])

  // Data for the chart
  const chartData = useMemo(() => {
    return sortedTimepoints.map(tp => ({
      timepoint: tp,
      position: timepointPositions.get(tp) || 0,
    }))
  }, [sortedTimepoints, timepointPositions])

  // Get positions for reference elements
  const startPosition = timepointPositions.get(startTimepoint)
  const endPosition = timepointPositions.get(endTimepoint)

  return (
    <div className="timeline-selector">
      <h3 className="timeline-selector__title">
        {label}
        <InfoIcon tooltip="Select a time range to evaluate methods. Drag the handles or click on timepoints. Start = when predictions were made; End = when annotations are used as ground truth." />
      </h3>

      <div
        ref={containerRef}
        className={`timeline-selector__chart ${isDragging ? 'timeline-selector__chart--dragging' : ''}`}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleTimelineClick}
      >
        <ResponsiveContainer width="100%" height={100}>
          <ComposedChart
            data={chartData}
            margin={{ top: 24, right: 36, left: 36, bottom: 34 }}
          >
            <XAxis
              dataKey="position"
              type="number"
              domain={[0, 100]}
              ticks={chartData.map(d => d.position)}
              tickFormatter={(value) => {
                const tp = sortedTimepoints.find(t => Math.abs((timepointPositions.get(t) || 0) - value) < 0.1)
                return tp ? formatTimepoint(tp) : ''
              }}
              tick={{ fontSize: 14, fill: 'var(--isu-charcoal)', fontWeight: 600 }}
              axisLine={{ stroke: 'var(--isu-border)' }}
              tickLine={{ stroke: 'var(--isu-border)' }}
            />
            <YAxis hide domain={[0, 1]} />

            {/* Timepoint markers */}
            {sortedTimepoints.map((tp) => (
              <ReferenceLine
                key={tp}
                x={timepointPositions.get(tp)}
                stroke="var(--isu-border)"
                strokeWidth={1}
              />
            ))}

            {/* Selected range highlight */}
            {startPosition !== undefined && endPosition !== undefined && (
              <ReferenceArea
                x1={startPosition}
                x2={endPosition}
                fill={isValidSelection ? 'var(--timeline-accent)' : 'var(--isu-caption)'}
                fillOpacity={0.15}
                stroke={isValidSelection ? 'var(--timeline-accent)' : 'var(--isu-caption)'}
                strokeDasharray={isValidSelection ? undefined : '4 4'}
              />
            )}

            {/* Custom handles rendered via reference lines with custom styling */}
            {startPosition !== undefined && (
              <ReferenceLine
                x={startPosition}
                stroke="var(--timeline-accent)"
                strokeWidth={3}
                label={{
                  value: '',
                  position: 'center',
                }}
              />
            )}
            {endPosition !== undefined && (
              <ReferenceLine
                x={endPosition}
                stroke="var(--timeline-accent)"
                strokeWidth={3}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>

        {/* Draggable handles overlay */}
        <div className="timeline-selector__handles">
          {startPosition !== undefined && (
            <div
              className={`timeline-selector__handle timeline-selector__handle--start ${activeHandle === 'start' ? 'timeline-selector__handle--active' : ''}`}
              style={{ left: `calc(36px + ${(startPosition / 100) * (containerRef.current?.offsetWidth ? containerRef.current.offsetWidth - 72 : 0)}px)` }}
              onMouseDown={(e) => {
                e.stopPropagation()
                handleMouseDown('start')
              }}
              title={startTimepoint ? formatTimepoint(startTimepoint) : 'Start'}
            />
          )}
          {endPosition !== undefined && (
            <div
              className={`timeline-selector__handle timeline-selector__handle--end ${activeHandle === 'end' ? 'timeline-selector__handle--active' : ''}`}
              style={{ left: `calc(36px + ${(endPosition / 100) * (containerRef.current?.offsetWidth ? containerRef.current.offsetWidth - 72 : 0)}px)` }}
              onMouseDown={(e) => {
                e.stopPropagation()
                handleMouseDown('end')
              }}
              title={endTimepoint ? formatTimepoint(endTimepoint) : 'End'}
            />
          )}
        </div>
      </div>

      {/* Current selection display */}
      <div className="timeline-selector__selection">
        {startTimepoint && endTimepoint ? (
          <>
            <span className="timeline-selector__range">
              {formatTimepoint(startTimepoint)} — {formatTimepoint(endTimepoint)}
            </span>
            {!isValidSelection && (
              <span className="timeline-selector__warning">No data available</span>
            )}
          </>
        ) : (
          <span className="timeline-selector__hint">Click or drag to select range</span>
        )}
      </div>
    </div>
  )
}
