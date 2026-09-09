import { beforeEach, describe, expect, it } from 'vitest'

import { applyTheme, readTheme } from './useTheme'

describe('theme store', () => {
  beforeEach(() => {
    delete document.documentElement.dataset.theme
    localStorage.clear()
  })

  it('defaults to dark when nothing was stamped on <html>', () => {
    expect(readTheme()).toBe('dark')
  })

  it('reads whatever index.html stamped before the first render', () => {
    document.documentElement.dataset.theme = 'light'
    expect(readTheme()).toBe('light')
  })

  it('ignores a value that is not a theme', () => {
    document.documentElement.dataset.theme = 'sepia'
    expect(readTheme()).toBe('dark')
  })

  it('applies to <html> and persists under the same key index.html reads', () => {
    applyTheme('light')
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(localStorage.getItem('memgraphrag.theme')).toBe('light')
    applyTheme('dark')
    expect(readTheme()).toBe('dark')
    expect(localStorage.getItem('memgraphrag.theme')).toBe('dark')
  })
})
