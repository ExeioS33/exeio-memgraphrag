import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'

import * as api from './api/client'
import { ApiError } from './api/client'
import type {
  AuthUser,
  GraphSuggestion,
  HealthResponse,
  LibraryTarget,
  ProviderInfo,
  QuerySettings,
} from './api/types'
import Composer from './components/Composer'
import EmptyState, { SuggestionList } from './components/EmptyState'
import { CloseIcon } from './components/icons'
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
const AdminPanel = lazy(() => import('./components/AdminPanel'))

const BASE_SETTINGS: QuerySettings = { mode: 'ppr', provider: null, model: null }

/** What the UI shows before /auth/me answers, or when it cannot (guest token). */
const GUEST: AuthUser = { id: null, name: 'guest', email: null, role: 'guest', auth_source: 'env' }

const MODE_LABEL: Record<QuerySettings['mode'], string> = {
  ppr: 'Graphe',
  naive: 'Dense',
  context: 'Contexte',
  bypass: 'Direct',
  agent: 'Agent',
}

/** Together AI's refusal for a model the account has not enabled. It arrives as a
 *  400 with an opaque sentence; the UI turns it into a marked entry in the picker. */
const NON_SERVERLESS = /non-serverless model/i

export default function App() {
  const [authed, setAuthed] = useState<boolean | null>(null)
  const [user, setUser] = useState<AuthUser>(GUEST)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [settings, setSettings] = useState<QuerySettings>(BASE_SETTINGS)
  const [suggestions, setSuggestions] = useState<GraphSuggestion[]>([])
  const [nav, setNav] = useState<NavKey>('chat')
  const [showSettings, setShowSettings] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [unavailableModels, setUnavailableModels] = useState<ReadonlySet<string>>(() => new Set())
  // Where a citation click should land. The library panel is mounted lazily and
  // unmounted on close, so it reads this once at mount — no imperative ref needed.
  const [libraryTarget, setLibraryTarget] = useState<LibraryTarget | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const chat = useChat(settings, authed === true)

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
    // Who we are, for the sidebar. A guest token has no /auth/me worth showing.
    try {
      const me = await api.me()
      setUser({ ...me, name: me.name || me.email || 'guest', role: me.role || 'guest' })
    } catch {
      setUser(GUEST)
    }
  }, [])

  useEffect(() => {
    void bootstrap()
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [bootstrap])

  // Suggestions are derived from the graph, so they name entities that are
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
        /* the empty state still renders; the list just stays as skeletons */
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

  // A provider refusal names the model; remember it so the picker can say so.
  useEffect(() => {
    if (chat.error && NON_SERVERLESS.test(chat.error) && settings.model) {
      const refused = settings.model
      setUnavailableModels((prev) => (prev.has(refused) ? prev : new Set(prev).add(refused)))
    }
  }, [chat.error, settings.model])

  const logout = useCallback(() => {
    api.logout()
    setUser(GUEST)
    setNav('chat')
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
    return (
      <LoginDialog
        signupEnabled={Boolean(health?.signup_enabled)}
        onSuccess={() => void bootstrap()}
      />
    )
  }

  const activeThread = chat.threads.find((t) => t.id === chat.activeId) ?? null
  const empty = chat.messages.length === 0 && !chat.streaming
  const modelError =
    chat.error && NON_SERVERLESS.test(chat.error)
      ? `Le fournisseur refuse « ${settings.model} » : ce modèle n’est pas activé sur ce compte. Choisissez-en un autre dans la liste.`
      : chat.error

  return (
    <div className="flex h-full">
      <Sidebar
        threads={chat.threads}
        activeId={chat.activeId}
        collapsed={collapsed}
        persistent={chat.persistent}
        user={user}
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
        onRenameThread={(id, title) => void chat.renameThread(id, title)}
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
            unavailableModels={unavailableModels}
            onSelect={(provider, model) => setSettings((prev) => ({ ...prev, provider, model }))}
            messages={chat.messages}
            threadTitle={activeThread?.title ?? null}
            onOpenSettings={() => setShowSettings(true)}
          />

          {health?.retrieval_status === 'error' && (
            <p className="mx-5 mb-2 rounded-lg bg-red-500/10 px-3 py-2 text-[12px] text-red-500">
              Récupération indisponible : {health.retrieval_error}
            </p>
          )}

          <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-5">
            <div className="mx-auto flex min-h-full w-full max-w-[760px] flex-col">
              {empty ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-6">
                  <EmptyState model={settings.model ?? null} account={user.name} />
                  <div className="mt-7 w-full">
                    <Composer
                      hero
                      disabled={chat.streaming}
                      streaming={chat.streaming}
                      mode={MODE_LABEL[settings.mode]}
                      onSend={(text) => void chat.send(text)}
                      onStop={chat.stop}
                      onOpenSettings={() => setShowSettings(true)}
                    />
                  </div>
                  <SuggestionList
                    suggestions={suggestions}
                    onPick={(prompt) => void chat.send(prompt)}
                  />
                </div>
              ) : (
                <div className="py-4">
                  <MessageList
                    messages={chat.messages}
                    streaming={chat.streaming}
                    pendingAnswer={chat.pendingAnswer}
                    pendingRefs={chat.pendingRefs}
                    pendingSteps={chat.pendingSteps}
                    pendingThinkingSeconds={chat.pendingThinkingSeconds}
                    thinkingSecondsById={chat.thinkingSecondsById}
                    scrollRef={scrollRef}
                    onCitationClick={(ref) => {
                      setLibraryTarget({
                        path: ref.source_path || ref.file_path,
                        chunkId: ref.chunk_id ?? null,
                      })
                      setNav('library')
                    }}
                    onRegenerate={() => void chat.regenerate()}
                  />
                </div>
              )}
            </div>
          </div>

          <div className="px-5 pb-3">
            <div className="mx-auto w-full max-w-[760px]">
              {modelError && (
                <div className="mb-2 flex items-start justify-between gap-3 rounded-lg bg-red-500/10 px-3 py-2 text-[12px] text-red-500">
                  <span>{modelError}</span>
                  <button onClick={() => chat.setError(null)} className="shrink-0" aria-label="Fermer">
                    <CloseIcon size={14} />
                  </button>
                </div>
              )}
              {!empty && (
                <Composer
                  disabled={chat.streaming}
                  streaming={chat.streaming}
                  mode={MODE_LABEL[settings.mode]}
                  onSend={(text) => void chat.send(text)}
                  onStop={chat.stop}
                  onOpenSettings={() => setShowSettings(true)}
                />
              )}
              <p className="mt-2 text-center text-[11px] text-ink-faint">
                {health
                  ? `MemGraphRAG ${health.core_version} · API ${health.api_version}${
                      health.pipeline_busy ? ' · ingestion en cours' : ''
                    } · les réponses citent le corpus, vérifiez les sources`
                  : 'Serveur injoignable'}
              </p>
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
        {nav === 'admin' && user.role === 'admin' && (
          <AdminPanel selfId={user.id} onClose={() => setNav('chat')} />
        )}
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
            border border-edge bg-surface-raised px-4 py-2 text-[12.5px] shadow-lg"
        >
          {toast}
        </div>
      )}
    </div>
  )
}
