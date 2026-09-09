import { defineConfig } from 'vitest/config'

// Pure-logic tests only: the reply split, the theme store and the thread filter.
// The DOM-backed ones run under happy-dom; nothing renders React.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'happy-dom',
  },
})
