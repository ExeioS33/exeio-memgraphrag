import { useMemo, useState } from 'react'

import type { ChatThread } from '../api/types'
import {
  BookIcon,
  BrandMark,
  GlobeIcon,
  HistoryIcon,
  LogoutIcon,
  PanelIcon,
  PlusIcon,
  SearchIcon,
  TrashIcon,
  TrayIcon,
} from './icons'

export type NavKey = 'chat' | 'library' | 'graph' | 'history'

const DAY = 86_400

/** The mockup groups the thread list by age; these are the buckets it shows. */
function bucketFor(updatedAt: number, now: number): string {
  const age = now - updatedAt
  if (age < DAY) return "Aujourd'hui"
  if (age < 2 * DAY) return 'Hier'
  if (age < 7 * DAY) return '7 jours'
  if (age < 30 * DAY) return '30 jours'
  return 'Plus ancien'
}

const BUCKET_ORDER = ["Aujourd'hui", 'Hier', '7 jours', '30 jours', 'Plus ancien']

interface Props {
  threads: ChatThread[]
  activeId: string | null
  collapsed: boolean
  persistent: boolean
  account: string
  onToggleCollapse: () => void
  onNewThread: () => void
  onOpenThread: (id: string) => void
  onDeleteThread: (id: string) => void
  onNavigate: (key: NavKey) => void
  onLogout: () => void
  active: NavKey
}

export default function Sidebar({
  threads,
  activeId,
  collapsed,
  persistent,
  account,
  onToggleCollapse,
  onNewThread,
  onOpenThread,
  onDeleteThread,
  onNavigate,
  onLogout,
  active,
}: Props) {
  const [filter, setFilter] = useState('')

  const grouped = useMemo(() => {
    const now = Math.floor(Date.now() / 1000)
    const needle = filter.trim().toLowerCase()
    const buckets = new Map<string, ChatThread[]>()
    for (const thread of threads) {
      if (needle && !thread.title.toLowerCase().includes(needle)) continue
      const key = bucketFor(thread.updated_at, now)
      const list = buckets.get(key)
      if (list) list.push(thread)
      else buckets.set(key, [thread])
    }
    return BUCKET_ORDER.filter((k) => buckets.has(k)).map((k) => [k, buckets.get(k)!] as const)
  }, [filter, threads])

  if (collapsed) {
    return (
      <aside className="flex w-[64px] shrink-0 flex-col items-center gap-3 py-4">
        <BrandMark />
        <button className="icon-btn" onClick={onToggleCollapse} title="Déplier le panneau">
          <PanelIcon />
        </button>
        <button className="icon-btn" onClick={onNewThread} title="Nouvelle discussion">
          <PlusIcon />
        </button>
      </aside>
    )
  }

  return (
    <aside className="flex w-[248px] shrink-0 flex-col gap-3 px-3 py-4">
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <BrandMark />
          <span className="text-[15px] font-semibold tracking-tight">MemGraphRAG</span>
        </div>
        <button className="icon-btn" onClick={onToggleCollapse} title="Replier le panneau">
          <PanelIcon />
        </button>
      </div>

      <button className="btn-dark w-full py-2.5" onClick={onNewThread}>
        <PlusIcon size={16} />
        Nouvelle discussion
      </button>

      <label className="relative block">
        <SearchIcon
          size={15}
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint"
        />
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Rechercher"
          className="w-full rounded-full border border-edge bg-white py-2 pl-9 pr-3 text-sm
            outline-none transition placeholder:text-ink-faint focus:border-violet-300"
        />
      </label>

      <nav className="flex flex-col gap-0.5">
        <button
          className={`nav-item ${active === 'library' ? 'nav-item-active' : ''}`}
          onClick={() => onNavigate('library')}
        >
          <BookIcon size={17} />
          Bibliothèque
        </button>
        <button
          className={`nav-item ${active === 'graph' ? 'nav-item-active' : ''}`}
          onClick={() => onNavigate('graph')}
        >
          <GlobeIcon size={17} />
          Explorer le graphe
        </button>
        <button
          className={`nav-item ${active === 'chat' ? 'nav-item-active' : ''}`}
          onClick={() => onNavigate('chat')}
        >
          <TrayIcon size={17} />
          Discussion
        </button>
      </nav>

      <div className="mt-1 border-t border-edge pt-3" />

      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {!persistent && (
          <p className="mb-3 rounded-lg bg-violet-50 px-2.5 py-2 text-[11px] leading-snug text-ink-muted">
            Persistance indisponible : les discussions restent dans cet onglet. Démarrez le
            service <code className="font-mono">postgres-app</code> pour les conserver.
          </p>
        )}
        {grouped.length === 0 ? (
          <p className="px-2 text-xs text-ink-faint">
            {filter ? 'Aucune discussion ne correspond.' : 'Aucune discussion pour le moment.'}
          </p>
        ) : (
          grouped.map(([bucket, list]) => (
            <div key={bucket} className="mb-3">
              <p className="px-2 pb-1 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
                {bucket}
              </p>
              <ul className="flex flex-col gap-0.5">
                {list.map((thread) => (
                  <li key={thread.id} className="group relative">
                    <button
                      onClick={() => onOpenThread(thread.id)}
                      title={thread.title}
                      className={`w-full truncate rounded-lg py-1.5 pl-2 pr-7 text-left text-[13px]
                        transition ${
                          thread.id === activeId
                            ? 'bg-white text-ink shadow-[0_1px_2px_rgba(0,0,0,0.04)]'
                            : 'text-ink-muted hover:bg-white hover:text-ink'
                        }`}
                    >
                      {thread.title}
                    </button>
                    <button
                      onClick={() => onDeleteThread(thread.id)}
                      title="Supprimer"
                      className="absolute right-1 top-1/2 hidden -translate-y-1/2 rounded p-1
                        text-ink-faint transition hover:text-ink group-hover:block"
                    >
                      <TrashIcon size={14} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))
        )}
      </div>

      <div className="flex items-center gap-2 rounded-card border border-edge bg-white p-2">
        <div
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full
            bg-violet-100 text-[12px] font-semibold text-violet-700"
        >
          {account.slice(0, 2).toUpperCase()}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13px] font-medium leading-tight">{account}</p>
          <p className="truncate text-[11px] text-ink-faint">
            {persistent ? 'Session authentifiée' : 'Session locale'}
          </p>
        </div>
        <button className="icon-btn h-7 w-7" onClick={onLogout} title="Se déconnecter">
          <LogoutIcon size={15} />
        </button>
      </div>

      <button
        className="nav-item justify-start text-[12px]"
        onClick={() => onNavigate('history')}
      >
        <HistoryIcon size={15} />
        Historique complet
      </button>
    </aside>
  )
}
