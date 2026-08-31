import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react'

import * as api from './api/client'
import { ApiError } from './api/client'
import type { HealthResponse, QuerySettings } from './api/types'
import Composer from './components/Composer'
import EmptyState, { SuggestionCards } from './components/EmptyState'
import { CloseIcon, HelpIcon, TranslateIcon } from './components/icons'
import LoginDialog from './components/LoginDialog'
import MessageList from './components/MessageList'
import Sidebar, { type NavKey } from './components/Sidebar'
import TopBar from './components/TopBar'
import { useChat } from './state/useChat'

// Panels are loaded on demand: the chat path is what the page is for, and neither
// the library nor the graph explorer is needed to render the first screen.
const LibraryPanel = lazy(() => import('./components/LibraryPanel'))
const GraphPanel = lazy(() => import('./components/GraphPanel'))
const SettingsPanel = lazy(() => import('./components/SettingsPanel'))

const BASE_SETTINGS: QuerySettings = { mode: 'ppr', model: null }

/** "Recherche approfondie" in the composer. Widens fact linking and re-enables the
 *  LLM fact rerank, which is what actually deepens a MemGraphRAG answer. */
const DEEP_SETTINGS: Partial<QuerySettings> = {
  linking_top_k: 90,
  top_k: 18,
  skip_fact_rerank: false,
  schema_top_k: 12,
}

export default function App() {
  const [authed, setAuthed] = useState<boolean | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [extensions, setExtensions] = useState<string[]>([])
  const [settings, setSettings] = useState<QuerySettings>(BASE_SETTINGS)
  const [deep, setDeep] = useState(false)
  const [nav, setNav] = useState<NavKey>('chat')
  const [showSettings, setShowSettings] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const effective = useMemo<QuerySettings>(
    () => (deep ? { ...settings, ...DEEP_SETTINGS } : settings),
    [deep, settings],
  )

  const chat = useChat(effective)

  const bootstrap = useCallback(async () => {
    try {
      const [modelsResponse, params] = await Promise.all([api.listModels(), api.queryParams()])
      setModels(modelsResponse.models)
      setExtensions(params.supported_extensions)
      setSettings((prev) => ({ ...prev, model: prev.model ?? modelsResponse.default }))
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

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(null), 6000)
    return () => window.clearTimeout(timer)
  }, [toast])

  const attach = useCallback(async (files: FileList) => {
    const names: string[] = []
    for (const file of Array.from(files)) {
      try {
        await api.uploadFile(file)
        names.push(file.name)
      } catch (exc) {
        setToast(`${file.name} : ${exc instanceof Error ? exc.message : String(exc)}`)
        return
      }
    }
    setToast(
      `${names.length} document(s) mis en file. L'indexation est asynchrone — suivez-la dans la bibliothèque.`,
    )
  }, [])

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
            models={models}
            model={settings.model ?? null}
            onModelChange={(model) => setSettings((prev) => ({ ...prev, model }))}
            mode={effective.mode}
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
                deepMode={deep}
                extensions={extensions}
                onToggleDeep={() => setDeep((v) => !v)}
                onSend={(text) => void chat.send(text)}
                onStop={chat.stop}
                onAttach={(files) => void attach(files)}
                onOpenSettings={() => setShowSettings(true)}
              />
              {empty && (
                <div className="mt-3">
                  <SuggestionCards onPick={(prompt) => void chat.send(prompt)} />
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
        {nav === 'library' && <LibraryPanel onClose={() => setNav('chat')} />}
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
