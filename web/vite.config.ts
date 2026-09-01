import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The bundle is written straight into the Python package so `uv run
// memgraphrag-server` serves it with no extra process and no CORS. Anything under
// memgraphrag/api/static is a build artifact — it is gitignored, and it reaches the
// wheel only because pyproject lists `api/static/**` under package-data.
const API_ROUTES = [
  '/query',
  '/chat',
  '/documents',
  '/graphs',
  '/graph',
  '/models',
  '/login',
  '/health',
  '/metrics',
  '/api',
]

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../memgraphrag/api/static',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    // `npm run dev` talks to a server started separately on 9621. Same-origin in
    // production, proxied in development, so the client never needs a base URL.
    proxy: Object.fromEntries(
      API_ROUTES.map((route) => [
        route,
        { target: 'http://127.0.0.1:9621', changeOrigin: true },
      ]),
    ),
  },
})
