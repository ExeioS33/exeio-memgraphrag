import { useCallback, useSyncExternalStore } from 'react'

export type Theme = 'light' | 'dark'

const STORAGE_KEY = 'memgraphrag.theme'
const DEFAULT_THEME: Theme = 'dark'

/** Read the theme index.html already stamped on <html>, so React and the
 *  pre-render script never disagree about what is showing. */
export function readTheme(): Theme {
  const current = document.documentElement.dataset.theme
  return current === 'light' || current === 'dark' ? current : DEFAULT_THEME
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    /* storage disabled: the choice lasts for the tab, which is still better than nothing */
  }
  listeners.forEach((listener) => listener())
}

const listeners = new Set<() => void>()

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

/** Current theme and a toggle. Backed by useSyncExternalStore so every subscriber
 *  re-renders on a change made anywhere, not only through this hook's own toggle. */
export function useTheme(): [Theme, () => void] {
  const theme = useSyncExternalStore(subscribe, readTheme, () => DEFAULT_THEME)
  const toggle = useCallback(() => applyTheme(readTheme() === 'dark' ? 'light' : 'dark'), [])
  return [theme, toggle]
}
