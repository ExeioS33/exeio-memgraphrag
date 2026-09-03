import { lazy, Suspense, useCallback, useEffect, useState } from 'react'

import * as api from './api/client'
import { ApiError } from './api/client'
import type {
  GraphSuggestion,
  HealthResponse,
  LibraryTarget,
  ProviderInfo,
  QuerySettings,
} from './api/types'
import Composer from './components/Composer'
import EmptyState, { SuggestionCards } from './components/EmptyState'
import { CloseIcon, HelpIcon, TranslateIcon } from './components/icons'
import LoginDialog from './components/LoginDialog'
import MessageList from './components/MessageList'
import Sidebar, { type NavKey } from './components/Sidebar'
import TopBar from './components/TopBar'
import { useChat } from './state/useChat'

// Panels are loaded on demand: the chat path is what the page is for, and neither
// the library nor the Cypher console is needed to render the first screen.
const LibraryPanel = lazy(() => import('./components/LibraryPanel'))
const GraphPanel = lazy(() => import('./components/GraphPanel'))
const SettingsPanel = lazy(() => import('./components/SettingsPanel'))

const BASE_SETTINGS: QuerySettings = { mode: 'ppr', provider: null, model: null }

export default function App() {
  const [authed, setAuthed] = useState<boolean | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [settings, setSettings] = useState<QuerySettings>(BASE_SETTINGS)
  const [suggestions, setSuggestions] = useState<GraphSuggestion[]>([])
  const [nav, setNav] = useState<NavKey>('chat')
  const [showSettings, setShowSettings] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  // Where a citation click should land. The library panel is mounted lazily and
  // unmounted on close, so it reads this once at mount — no imperative ref needed.
  const [libraryTarget, setLibraryTarget] = useState<LibraryTarget | null>(null)

  const chat = useChat(settings)

  const bootstrap = useCallback(async () => {
    try {
      const models = await api.listModels()
      setProviders(models.providers)
      setSettings((prev) => ({
        ...prev,
        provider: prev.provider ?? models.default.provider,
        model: prev.model ?? models.default.model,
      }))
      setAuthed(true)
    } catch (exc) {
      if (exc instanceof ApiError && exc.isAuthError) {
        setAuthed(false)
        return
      }
      setToast(exc instanceof Error ? exc.message : String(exc))
      setAuthed(true)
    }
  }, [])

  useEffect(() => {
    void bootstrap()
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [bootstrap])

  // Suggestion cards are derived from the graph, so they name entities that are
  // actually in the corpus. A hardcoded set goes stale the moment the corpus does.
  useEffect(() => {
    if (authed !== true) return
    let alive = true
    api
      .graphHighlights()
      .then((data) => {
        if (alive) setSuggestions(data.suggestions)
      })
      .catch(() => {
        /* the empty state still renders; the cards just stay as skeletons */
      })
    return () => {
      alive = false
    }
  }, [authed])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(null), 6000)
    return () => window.clearTimeout(timer)
  }, [toast])

  const logout = useCallback(() => {
    api.logout()
    setAuthed(false)
  }, [])

  if (authed === null) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-ink-muted">
        Connexion au serveur…
      </div>
    )
  }

  if (!authed) {
    return <LoginDialog onSuccess={() => void bootstrap()} />
  }

  const activeThread = chat.threads.find((t) => t.id === chat.activeId) ?? null
  const account = 'guest'
  const empty = chat.messages.length === 0 && !chat.streaming

  return (
    <div className="flex h-full">
      <Sidebar
        threads={chat.threads}
        activeId={chat.activeId}
        collapsed={collapsed}
        persistent={chat.persistent}
        account={account}
        active={nav}
        onToggleCollapse={() => setCollapsed((v) => !v)}
        onNewThread={() => {
          chat.newThread()
          setNav('chat')
        }}
        onOpenThread={(id) => {
          void chat.openThread(id)
          setNav('chat')
        }}
        onDeleteThread={(id) => void chat.removeThread(id)}
        onNavigate={setNav}
        onLogout={logout}
      />

      <main className="flex min-w-0 flex-1 flex-col py-3 pr-3">
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-panel border border-edge bg-surface">
          <TopBar
            providers={providers}
            provider={settings.provider ?? 'default'}
            model={settings.model ?? null}
            onSelect={(provider, model) => setSettings((prev) => ({ ...prev, provider, model }))}
            mode={settings.mode}
            messages={chat.messages}
            threadTitle={activeThread?.title ?? null}
            onOpenSettings={() => setShowSettings(true)}
          />

          {health?.retrieval_status === 'error' && (
            <p className="mx-5 mb-2 rounded-lg bg-red-50 px-3 py-2 text-[12px] text-red-700">
              Récupération indisponible : {health.retrieval_error}
            </p>
          )}

          <div className="min-h-0 flex-1 overflow-y-auto px-5">
            <div className="mx-auto flex min-h-full w-full max-w-[760px] flex-col">
              {empty ? (
                <div className="flex flex-1 flex-col justify-center">
                  <EmptyState account={account} />
                </div>
              ) : (
                <div className="py-4">
                  <MessageList
                    messages={chat.messages}
                    streaming={chat.streaming}
                    pendingAnswer={chat.pendingAnswer}
                    pendingRefs={chat.pendingRefs}
                    pendingSteps={chat.pendingSteps}
                    onCitationClick={(ref) => {
                      setLibraryTarget({
                        path: ref.source_path || ref.file_path,
                        chunkId: ref.chunk_id ?? null,
                      })
                      setNav('library')
                    }}
                  />
                </div>
              )}
            </div>
          </div>

          <div className="px-5 pb-3">
            <div className="mx-auto w-full max-w-[760px]">
              {chat.error && (
                <div className="mb-2 flex items-start justify-between gap-3 rounded-lg bg-red-50 px-3 py-2 text-[12px] text-red-700">
                  <span>{chat.error}</span>
                  <button onClick={() => chat.setError(null)} className="shrink-0">
                    <CloseIcon size={14} />
                  </button>
                </div>
              )}
              <Composer
                disabled={chat.streaming}
                streaming={chat.streaming}
                onSend={(text) => void chat.send(text)}
                onStop={chat.stop}
                onOpenSettings={() => setShowSettings(true)}
              />
              {empty && (
                <div className="mt-3">
                  <SuggestionCards
                    suggestions={suggestions}
                    onPick={(prompt) => void chat.send(prompt)}
                  />
                </div>
              )}
              <div className="mt-2 flex items-center justify-between text-[11px] text-ink-faint">
                <span>
                  {health
                    ? `MemGraphRAG ${health.core_version} · API ${health.api_version}${
                        health.pipeline_busy ? ' · ingestion en cours' : ''
                      }`
                    : 'Serveur injoignable'}
                </span>
                <span className="flex items-center gap-1">
                  <TranslateIcon size={13} />
                  <HelpIcon size={13} />
                </span>
              </div>
            </div>
          </div>
        </div>
      </main>

      <Suspense fallback={null}>
        {nav === 'library' && (
          <LibraryPanel
            target={libraryTarget}
            onClose={() => {
              // Cleared on close, or the next plain "Bibliothèque" click would jump
              // straight back to the last cited file.
              setLibraryTarget(null)
              setNav('chat')
            }}
          />
        )}
        {nav === 'graph' && <GraphPanel onClose={() => setNav('chat')} />}
        {showSettings && (
          <SettingsPanel
            settings={settings}
            onChange={setSettings}
            onClose={() => setShowSettings(false)}
          />
        )}
      </Suspense>

      {toast && (
        <div
          className="fixed bottom-4 left-1/2 z-50 max-w-[520px] -translate-x-1/2 rounded-full
            border border-edge bg-white px-4 py-2 text-[12.5px] shadow-lg"
        >
          {toast}
        </div>
      )}
    </div>
  )
}
