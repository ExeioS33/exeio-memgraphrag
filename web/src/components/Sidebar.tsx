import { useEffect, useMemo, useRef, useState } from 'react'

import type { AuthUser, ChatThread } from '../api/types'
import { groupThreads } from '../lib/threads'
import AccountMenu from './AccountMenu'
import {
  BookIcon,
  BrandMark,
  EditIcon,
  GlobeIcon,
  PanelIcon,
  SearchIcon,
  ShieldIcon,
  TrashIcon,
} from './icons'

export type NavKey = 'chat' | 'library' | 'graph' | 'admin'

interface Props {
  threads: ChatThread[]
  activeId: string | null
  collapsed: boolean
  persistent: boolean
  user: AuthUser
  active: NavKey
  onToggleCollapse: () => void
  onNewThread: () => void
  onOpenThread: (id: string) => void
  onRenameThread: (id: string, title: string) => void
  onDeleteThread: (id: string) => void
  onNavigate: (key: NavKey) => void
  onLogout: () => void
}

/** One conversation row: click to open, hover for rename and delete, rename in place. */
function ThreadRow({
  thread,
  active,
  onOpen,
  onRename,
  onDelete,
}: {
  thread: ChatThread
  active: boolean
  onOpen: () => void
  onRename: (title: string) => void
  onDelete: () => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(thread.title)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) input.current?.select()
  }, [editing])

  const commit = () => {
    const title = draft.trim()
    setEditing(false)
    if (title && title !== thread.title) onRename(title)
    else setDraft(thread.title)
  }

  if (editing) {
    return (
      <li>
        <input
          ref={input}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commit()
            if (e.key === 'Escape') {
              setDraft(thread.title)
              setEditing(false)
            }
          }}
          aria-label="Renommer la discussion"
          className="w-full rounded-lg border border-violet-300 bg-surface-raised px-2 py-1.5
            text-[13px] outline-none"
        />
      </li>
    )
  }

  return (
    <li className="group relative">
      <button
        type="button"
        onClick={onOpen}
        onDoubleClick={() => setEditing(true)}
        title={thread.title}
        className={`w-full truncate rounded-lg py-1.5 pl-2.5 pr-14 text-left text-[13.5px] transition
          ${active ? 'bg-surface-raised text-ink' : 'text-ink-muted hover:bg-surface-raised hover:text-ink'}`}
      >
        {thread.title}
      </button>
      <span
        className="absolute right-1 top-1/2 hidden -translate-y-1/2 items-center gap-0.5
          group-hover:flex group-focus-within:flex"
      >
        <button
          type="button"
          onClick={() => setEditing(true)}
          title="Renommer"
          aria-label="Renommer"
          className="rounded p-1 text-ink-faint transition hover:text-ink"
        >
          <EditIcon size={13} />
        </button>
        <button
          type="button"
          onClick={onDelete}
          title="Supprimer"
          aria-label="Supprimer"
          className="rounded p-1 text-ink-faint transition hover:text-ink"
        >
          <TrashIcon size={13} />
        </button>
      </span>
    </li>
  )
}

/**
 * Open WebUI's sidebar shape: brand and collapse on top, generous icon+label
 * rows, quiet section labels, the account pinned to the bottom. Features that do
 * not exist here — Notes, Workspace, Channels, Folders — are not shown empty.
 */
export default function Sidebar({
  threads,
  activeId,
  collapsed,
  persistent,
  user,
  active,
  onToggleCollapse,
  onNewThread,
  onOpenThread,
  onRenameThread,
  onDeleteThread,
  onNavigate,
  onLogout,
}: Props) {
  const [searching, setSearching] = useState(false)
  const [filter, setFilter] = useState('')
  const searchInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (searching) searchInput.current?.focus()
    else setFilter('')
  }, [searching])

  const grouped = useMemo(() => groupThreads(threads, filter), [filter, threads])

  if (collapsed) {
    return (
      <aside className="flex w-[60px] shrink-0 flex-col items-center gap-2 py-3">
        <button className="icon-btn" onClick={onToggleCollapse} title="Déplier le panneau">
          <PanelIcon />
        </button>
        <button className="icon-btn" onClick={onNewThread} title="Nouvelle discussion">
          <EditIcon size={17} />
        </button>
        <button className="icon-btn" onClick={() => onNavigate('library')} title="Bibliothèque">
          <BookIcon size={17} />
        </button>
        <button className="icon-btn" onClick={() => onNavigate('graph')} title="Explorer le graphe">
          <GlobeIcon size={17} />
        </button>
      </aside>
    )
  }

  return (
    <aside className="flex w-[270px] shrink-0 flex-col px-2.5 py-3">
      <div className="mb-3 flex items-center justify-between px-1.5">
        <div className="flex items-center gap-2">
          <BrandMark size={26} />
          <span className="text-[15px] font-semibold tracking-tight">MemGraphRAG</span>
        </div>
        <button className="icon-btn" onClick={onToggleCollapse} title="Replier le panneau">
          <PanelIcon />
        </button>
      </div>

      <nav className="flex flex-col gap-0.5">
        <button className="nav-item py-2.5" onClick={onNewThread}>
          <EditIcon size={17} />
          Nouvelle discussion
        </button>
        <button
          className={`nav-item py-2.5 ${searching ? 'nav-item-active' : ''}`}
          onClick={() => setSearching((v) => !v)}
          aria-expanded={searching}
        >
          <SearchIcon size={17} />
          Rechercher
        </button>
        {searching && (
          <input
            ref={searchInput}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') setSearching(false)
            }}
            placeholder="Filtrer les discussions"
            aria-label="Filtrer les discussions"
            className="mx-1 mb-1 rounded-lg border border-edge bg-surface-raised px-3 py-1.5 text-[13px]
              outline-none placeholder:text-ink-faint focus:border-violet-300"
          />
        )}
        <button
          className={`nav-item py-2.5 ${active === 'library' ? 'nav-item-active' : ''}`}
          onClick={() => onNavigate('library')}
        >
          <BookIcon size={17} />
          Bibliothèque
        </button>
        <button
          className={`nav-item py-2.5 ${active === 'graph' ? 'nav-item-active' : ''}`}
          onClick={() => onNavigate('graph')}
        >
          <GlobeIcon size={17} />
          Explorer le graphe
        </button>
        {user.role === 'admin' && (
          <button
            className={`nav-item py-2.5 ${active === 'admin' ? 'nav-item-active' : ''}`}
            onClick={() => onNavigate('admin')}
          >
            <ShieldIcon size={17} />
            Administration
          </button>
        )}
      </nav>

      <div className="mt-4 min-h-0 flex-1 overflow-y-auto pr-0.5">
        <p className="px-2.5 pb-1.5 text-[12px] font-medium text-ink-faint">Conversations</p>
        {!persistent && (
          <p className="mx-1 mb-3 rounded-lg bg-violet-50 px-2.5 py-2 text-[11px] leading-snug text-ink-muted">
            Persistance indisponible : les discussions restent dans cet onglet. Démarrez le
            service <code className="font-mono">postgres-app</code> pour les conserver.
          </p>
        )}
        {grouped.length === 0 ? (
          <p className="px-2.5 text-xs text-ink-faint">
            {filter ? 'Aucune discussion ne correspond.' : 'Aucune discussion pour le moment.'}
          </p>
        ) : (
          grouped.map(([bucket, list]) => (
            <div key={bucket} className="mb-3">
              <p className="px-2.5 pb-1 text-[11px] text-ink-faint">{bucket}</p>
              <ul className="flex flex-col gap-0.5">
                {list.map((thread) => (
                  <ThreadRow
                    key={thread.id}
                    thread={thread}
                    active={thread.id === activeId}
                    onOpen={() => onOpenThread(thread.id)}
                    onRename={(title) => onRenameThread(thread.id, title)}
                    onDelete={() => onDeleteThread(thread.id)}
                  />
                ))}
              </ul>
            </div>
          ))
        )}
      </div>

      <div className="mt-2 border-t border-edge pt-2">
        <AccountMenu
          user={user}
          persistent={persistent}
          onLogout={onLogout}
          onOpenAdmin={user.role === 'admin' ? () => onNavigate('admin') : undefined}
        />
      </div>
    </aside>
  )
}
