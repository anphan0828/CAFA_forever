import { createContext, useContext, useReducer, type ReactNode, type Dispatch } from 'react'
import type { Subset } from '../types'

export type ActiveTab = 'summary' | 'curves' | 'breakdown' | 'table'

interface AppState {
  primaryRelease: string | null
  secondaryRelease: string | null
  compareMode: boolean
  selectedMethods: string[]
  activeTab: ActiveTab
  activeSubset: Subset
  configPanelOpen: boolean
  showBaselinesOnly: boolean
}

type AppAction =
  | { type: 'SET_PRIMARY_RELEASE'; payload: string | null }
  | { type: 'SET_SECONDARY_RELEASE'; payload: string | null }
  | { type: 'SET_COMPARE_MODE'; payload: boolean }
  | { type: 'SET_SELECTED_METHODS'; payload: string[] }
  | { type: 'TOGGLE_METHOD'; payload: string }
  | { type: 'SET_ACTIVE_TAB'; payload: ActiveTab }
  | { type: 'SET_ACTIVE_SUBSET'; payload: Subset }
  | { type: 'SET_CONFIG_PANEL_OPEN'; payload: boolean }
  | { type: 'SET_SHOW_BASELINES_ONLY'; payload: boolean }
  | { type: 'RESET_STATE' }

const initialState: AppState = {
  primaryRelease: null,
  secondaryRelease: null,
  compareMode: false,
  selectedMethods: [],
  activeTab: 'summary',
  activeSubset: 'NK',
  configPanelOpen: true,
  showBaselinesOnly: false,
}

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_PRIMARY_RELEASE':
      return { ...state, primaryRelease: action.payload, selectedMethods: [] }

    case 'SET_SECONDARY_RELEASE':
      return { ...state, secondaryRelease: action.payload }

    case 'SET_COMPARE_MODE':
      return { ...state, compareMode: action.payload }

    case 'SET_SELECTED_METHODS':
      return { ...state, selectedMethods: action.payload }

    case 'TOGGLE_METHOD':
      return {
        ...state,
        selectedMethods: state.selectedMethods.includes(action.payload)
          ? state.selectedMethods.filter((m) => m !== action.payload)
          : [...state.selectedMethods, action.payload],
      }

    case 'SET_ACTIVE_TAB':
      return { ...state, activeTab: action.payload }

    case 'SET_ACTIVE_SUBSET':
      return { ...state, activeSubset: action.payload }

    case 'SET_CONFIG_PANEL_OPEN':
      return { ...state, configPanelOpen: action.payload }

    case 'SET_SHOW_BASELINES_ONLY':
      return { ...state, showBaselinesOnly: action.payload }

    case 'RESET_STATE':
      return initialState

    default:
      return state
  }
}

interface AppContextValue {
  state: AppState
  dispatch: Dispatch<AppAction>
}

const AppContext = createContext<AppContextValue | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState)

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  )
}

export function useAppState() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error('useAppState must be used within an AppProvider')
  }
  return context
}
