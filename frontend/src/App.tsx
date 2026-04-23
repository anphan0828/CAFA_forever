import { useEffect, useMemo } from 'react'
import { AppProvider, useAppState } from './context'
import { useCatalog, useReleaseData, useMethodsConfig, useComparisonData } from './hooks'
import { Header, Footer, Section } from './components/layout'
import { HeroSection, TimelineSelector, CompareToggle } from './components/release'
import { MethodSelector } from './components/methods'
import { OverallRankingChart, FmaxSmallMultiples, TargetCountChart, PRCurveGrid, ComparisonTab } from './components/charts'
import { SummaryTable } from './components/table'
import { Tabs, Collapsible, type Tab } from './components/ui'
import './App.css'

function AppContent() {
  const { state, dispatch } = useAppState()
  const { catalog, loading: catalogLoading, error: catalogError } = useCatalog()
  const { config: methodsConfig, loading: configLoading } = useMethodsConfig()
  const {
    data: releaseData,
    loading: releaseLoading,
    loadCurves,
    curvesLoading,
  } = useReleaseData(state.primaryRelease)

  // Load secondary release data when in compare mode
  const {
    data: secondaryReleaseData,
    loading: secondaryLoading,
  } = useReleaseData(state.compareMode ? state.secondaryRelease : null)

  // Get method lists for comparison (filtered by selected methods and baseline toggle)
  const primaryMethods = useMemo(() => {
    if (!releaseData.methods) return []
    const available = new Set(Object.keys(releaseData.methods.methods))
    return state.selectedMethods.filter(m => {
      if (!available.has(m)) return false
      if (state.showBaselinesOnly && methodsConfig) {
        const config = methodsConfig.methods[m]
        if (!config?.isBaseline) return false
      }
      return true
    })
  }, [releaseData.methods, state.selectedMethods, state.showBaselinesOnly, methodsConfig])

  const secondaryMethods = useMemo(() => {
    if (!secondaryReleaseData.methods) return []
    const available = new Set(Object.keys(secondaryReleaseData.methods.methods))
    return state.selectedMethods.filter(m => {
      if (!available.has(m)) return false
      if (state.showBaselinesOnly && methodsConfig) {
        const config = methodsConfig.methods[m]
        if (!config?.isBaseline) return false
      }
      return true
    })
  }, [secondaryReleaseData.methods, state.selectedMethods, state.showBaselinesOnly, methodsConfig])

  // Compute comparison data
  const comparisonData = useComparisonData(
    releaseData.best,
    secondaryReleaseData.best,
    primaryMethods,
    secondaryMethods
  )

  // Format release ID to readable label
  const formatReleaseLabel = (releaseId: string | null): string => {
    if (!releaseId) return ''
    const parts = releaseId.split('_')
    if (parts.length !== 4) return releaseId
    return `${parts[0]} ${parts[1]} - ${parts[2]} ${parts[3]}`
  }

  const primaryLabel = formatReleaseLabel(state.primaryRelease)
  const secondaryLabel = formatReleaseLabel(state.secondaryRelease)

  // Auto-select first release
  useEffect(() => {
    if (catalog?.releases.length && !state.primaryRelease) {
      const readyReleases = catalog.releases.filter((r) => r.status === 'ready')
      if (readyReleases.length > 0) {
        dispatch({ type: 'SET_PRIMARY_RELEASE', payload: readyReleases[0].id })
      }
    }
  }, [catalog, state.primaryRelease, dispatch])

  // Auto-select all methods when release data loads
  useEffect(() => {
    if (releaseData.methods && state.selectedMethods.length === 0) {
      const methods = Object.keys(releaseData.methods.methods)
      dispatch({ type: 'SET_SELECTED_METHODS', payload: methods })
    }
  }, [releaseData.methods, state.selectedMethods.length, dispatch])

  // Load curves when switching to curves tab
  useEffect(() => {
    if (state.activeTab === 'curves' && !releaseData.curves && !curvesLoading) {
      loadCurves()
    }
  }, [state.activeTab, releaseData.curves, curvesLoading, loadCurves])

  // Auto-select secondary release when comparison mode is enabled
  useEffect(() => {
    if (state.compareMode && !state.secondaryRelease && catalog?.releases.length) {
      const readyReleases = catalog.releases.filter((r) => r.status === 'ready')
      // Select a different release than primary, or fallback to first ready release
      const secondaryCandidate = readyReleases.find((r) => r.id !== state.primaryRelease)
        || readyReleases[0]
      if (secondaryCandidate) {
        dispatch({ type: 'SET_SECONDARY_RELEASE', payload: secondaryCandidate.id })
      }
    }
  }, [state.compareMode, state.secondaryRelease, state.primaryRelease, catalog, dispatch])

  if (catalogLoading || configLoading) {
    return (
      <div className="app">
        <Header />
        <main className="main-content">
          <div className="loading-container">
            <div className="loading-spinner" />
            <p>Loading CAFA Forever...</p>
          </div>
        </main>
        <Footer />
      </div>
    )
  }

  if (catalogError) {
    return (
      <div className="app">
        <Header />
        <main className="main-content">
          <div className="error-container">
            <h2>Error Loading Data</h2>
            <p>{catalogError}</p>
            <p>Please ensure the data pipeline has been run.</p>
          </div>
        </main>
        <Footer />
      </div>
    )
  }

  const methodColorDomain = releaseData.methods
    ? Object.keys(releaseData.methods.methods)
    : state.selectedMethods

  const tabs: Tab[] = [
    {
      id: 'summary',
      label: 'Summary',
      content: (
        <div className="tab-content">
          {releaseData.best && (
            <OverallRankingChart
              bestMetrics={releaseData.best}
              selectedMethods={state.selectedMethods}
              colorDomain={methodColorDomain}
            />
          )}

          {releaseData.best && (
            <FmaxSmallMultiples
              bestMetrics={releaseData.best}
              selectedMethods={state.selectedMethods}
              colorDomain={methodColorDomain}
            />
          )}

          {releaseData.meta && (
            <TargetCountChart targetCounts={releaseData.meta.targetCounts} />
          )}
        </div>
      ),
    },
    {
      id: 'curves',
      label: 'PR Curves',
      content: (
        <div className="tab-content">
          {curvesLoading ? (
            <div className="loading-container">
              <div className="loading-spinner" />
              <p>Loading PR curves...</p>
            </div>
          ) : releaseData.curves && releaseData.best ? (
            <PRCurveGrid
              curves={releaseData.curves}
              bestMetrics={releaseData.best}
              selectedMethods={state.selectedMethods.slice(0, 5)}
              colorDomain={methodColorDomain}
              totalSelectedCount={state.selectedMethods.length}
            />
          ) : (
            <p>Select a release to view PR curves</p>
          )}
        </div>
      ),
    },
    {
      id: 'table',
      label: 'Data Table',
      content: (
        <div className="tab-content">
          {releaseData.best && state.primaryRelease ? (
            <SummaryTable
              bestMetrics={releaseData.best}
              selectedMethods={state.selectedMethods}
              releaseId={state.primaryRelease}
            />
          ) : (
            <p>Select a release to view data table</p>
          )}
        </div>
      ),
    },
    {
      id: 'comparison',
      label: 'Comparison',
      hidden: !state.compareMode,
      content: (
        <div className="tab-content">
          <ComparisonTab
            comparisonData={comparisonData}
            windowALabel={primaryLabel}
            windowBLabel={secondaryLabel}
            loading={secondaryLoading}
          />
        </div>
      ),
    },
  ]

  return (
    <div className="app">
      <Header />

      <main id="main-content" className="main-content">
        <HeroSection />

        {catalog && (
          <Section id="analysis" title="Results">
            <div className="analysis-layout">
              <aside className="analysis-sidebar">
                <TimelineSelector
                  releases={catalog.releases}
                  timepoints={catalog.timepoints}
                  selectedRelease={state.primaryRelease}
                  onSelectRelease={(id) => dispatch({ type: 'SET_PRIMARY_RELEASE', payload: id })}
                />

                <CompareToggle
                  enabled={state.compareMode}
                  onToggle={(enabled) => dispatch({ type: 'SET_COMPARE_MODE', payload: enabled })}
                  releases={catalog.releases}
                  timepoints={catalog.timepoints}
                  secondaryRelease={state.secondaryRelease}
                  onSelectSecondaryRelease={(id) => dispatch({ type: 'SET_SECONDARY_RELEASE', payload: id })}
                />

                {releaseData.methods && methodsConfig && (
                  <Collapsible title="Methods" defaultOpen={state.configPanelOpen}>
                    <MethodSelector
                      methods={releaseData.methods.methods}
                      methodConfigs={methodsConfig.methods}
                      selectedMethods={state.selectedMethods}
                      onSelectionChange={(methods) =>
                        dispatch({ type: 'SET_SELECTED_METHODS', payload: methods })
                      }
                      activeSubset={state.activeSubset}
                      showBaselinesOnly={state.showBaselinesOnly}
                      onShowBaselinesOnlyChange={(show) =>
                        dispatch({ type: 'SET_SHOW_BASELINES_ONLY', payload: show })
                      }
                    />
                  </Collapsible>
                )}
              </aside>

              <div className="analysis-main">
                {releaseLoading ? (
                  <div className="loading-container">
                    <div className="loading-spinner" />
                    <p>Loading release data...</p>
                  </div>
                ) : state.primaryRelease && releaseData.best ? (
                  <Tabs
                    tabs={tabs}
                    defaultTab={state.activeTab}
                    onChange={(tabId) =>
                      dispatch({ type: 'SET_ACTIVE_TAB', payload: tabId as typeof state.activeTab })
                    }
                  />
                ) : (
                  <div className="analysis-placeholder">
                    <p>Select an evaluation window to view results</p>
                  </div>
                )}
              </div>
            </div>
          </Section>
        )}
      </main>

      <Footer />
    </div>
  )
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  )
}

export default App
