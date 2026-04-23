import { useMemo, useState } from 'react'
import type { ReleaseMethod, MethodConfig, Subset } from '../../types'
import { useMethodColors } from '../../hooks'
import { MethodBadge } from './MethodBadge'
import { Checkbox } from '../ui'
import './MethodSelector.css'

interface MethodSelectorProps {
  methods: Record<string, ReleaseMethod>
  methodConfigs: Record<string, MethodConfig>
  selectedMethods: string[]
  onSelectionChange: (methods: string[]) => void
  activeSubset?: Subset
  showBaselinesOnly?: boolean
  onShowBaselinesOnlyChange?: (show: boolean) => void
}

export function MethodSelector({
  methods,
  methodConfigs,
  selectedMethods,
  onSelectionChange,
  activeSubset,
  showBaselinesOnly,
  onShowBaselinesOnlyChange,
}: MethodSelectorProps) {
  const [searchQuery, setSearchQuery] = useState('')

  // Get sorted list of method names
  const sortedMethods = useMemo(() => {
    return Object.keys(methods).sort((a, b) => {
      // Baselines first
      const aBaseline = methodConfigs[a]?.isBaseline ?? false
      const bBaseline = methodConfigs[b]?.isBaseline ?? false
      if (aBaseline !== bBaseline) return bBaseline ? 1 : -1
      return a.localeCompare(b)
    })
  }, [methods, methodConfigs])

  // Filter by availability, baseline status, and search query
  const filteredMethods = useMemo(() => {
    const query = searchQuery.toLowerCase().trim()
    return sortedMethods.filter((name) => {
      const method = methods[name]
      const config = methodConfigs[name]

      // Filter by search query
      if (query && !name.toLowerCase().includes(query)) {
        return false
      }

      // Filter by availability if subset specified
      if (activeSubset && method.availability && !method.availability[activeSubset]) {
        return false
      }

      // Filter by baseline status
      if (showBaselinesOnly && !config?.isBaseline) {
        return false
      }

      return true
    })
  }, [sortedMethods, methods, methodConfigs, activeSubset, showBaselinesOnly, searchQuery])

  const { getColor } = useMethodColors(sortedMethods)

  const handleToggle = (method: string) => {
    if (selectedMethods.includes(method)) {
      onSelectionChange(selectedMethods.filter((m) => m !== method))
    } else {
      onSelectionChange([...selectedMethods, method])
    }
  }

  const handleSelectAll = () => {
    onSelectionChange(filteredMethods)
  }

  const handleClearAll = () => {
    onSelectionChange([])
  }

  const baselineCount = filteredMethods.filter((m) => methodConfigs[m]?.isBaseline).length
  const selectedCount = selectedMethods.filter((m) => filteredMethods.includes(m)).length

  return (
    <div className="method-selector">
      <div className="method-selector__header">
        <h3 className="method-selector__title">Methods</h3>
        <div className="method-selector__summary">
          {selectedCount} of {filteredMethods.length} selected
          {baselineCount > 0 && ` (${baselineCount} baselines)`}
        </div>
      </div>

      <div className="method-selector__search">
        <svg
          className="method-selector__search-icon"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <circle cx="11" cy="11" r="8" />
          <path d="M21 21l-4.35-4.35" />
        </svg>
        <input
          type="text"
          className="method-selector__search-input"
          placeholder="Search methods..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <button
            type="button"
            className="method-selector__search-clear"
            onClick={() => setSearchQuery('')}
            aria-label="Clear search"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      <div className="method-selector__controls">
        <div className="method-selector__actions">
          <button
            className="method-selector__action"
            onClick={handleSelectAll}
            disabled={selectedCount === filteredMethods.length}
          >
            Select All
          </button>
          <button
            className="method-selector__action"
            onClick={handleClearAll}
            disabled={selectedCount === 0}
          >
            Clear All
          </button>
        </div>

        {onShowBaselinesOnlyChange && (
          <Checkbox
            id="show-baselines-only"
            label="Show baselines only"
            checked={showBaselinesOnly ?? false}
            onChange={onShowBaselinesOnlyChange}
          />
        )}
      </div>

      <div className="method-selector__list">
        {filteredMethods.map((name) => {
          const config = methodConfigs[name]
          return (
            <MethodBadge
              key={name}
              label={name}
              color={getColor(name)}
              isBaseline={config?.isBaseline}
              description={config?.description}
              dockerUrl={config?.dockerUrl}
              selected={selectedMethods.includes(name)}
              onClick={() => handleToggle(name)}
            />
          )
        })}

        {filteredMethods.length === 0 && (
          <p className="method-selector__empty">
            No methods available for this selection.
          </p>
        )}
      </div>
    </div>
  )
}
