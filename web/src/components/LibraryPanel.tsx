/**
 * Document library: what the pipeline has ingested, and the passages it produced.
 *
 * Filtering is client-side even though /documents/ accepts a status: the panel polls
 * while anything is in flight, and a server-side filter of "processed" would hide the
 * very rows that decide whether to keep polling.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  ApiError,
  deleteDocument,
  documentChunks,
  insertText,
  listDocuments,
  requeueDocument,
  scanInputDir,
  uploadFile,
} from '../api/client'
import type { DocStatus, DocumentChunksResponse, DocumentRecord } from '../api/types'
import { CloseIcon, FileIcon, RefreshIcon, TrashIcon, TrayIcon } from './icons'

type Row = { id: string; record: DocumentRecord }
type Filter = 'tous' | DocStatus

type ChunkState = {
  docId: string
  loading: boolean
  error: string | null
  data: DocumentChunksResponse | null
}

const POLL_MS = 4000
const PAGE_SIZE = 200

const STATUS_META: Record<DocStatus, { label: string; dot: string; badge: string }> = {
  pending: { label: 'En attente', dot: 'bg-amber-500', badge: 'border-amber-200 bg-amber-50 text-amber-700' },
  parsing: { label: 'Analyse', dot: 'bg-amber-500', badge: 'border-amber-200 bg-amber-50 text-amber-700' },
  processing: { label: 'Traitement', dot: 'bg-amber-500', badge: 'border-amber-200 bg-amber-50 text-amber-700' },
  processed: { label: 'Traité', dot: 'bg-emerald-500', badge: 'border-emerald-200 bg-emerald-50 text-emerald-700' },
  failed: { label: 'Échec', dot: 'bg-red-500', badge: 'border-red-200 bg-red-50 text-red-700' },
}

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'tous', label: 'Tous' },
  { key: 'pending', label: 'En attente' },
  { key: 'parsing', label: 'Analyse' },
  { key: 'processing', label: 'Traitement' },
  { key: 'processed', label: 'Traités' },
  { key: 'failed', label: 'Échecs' },
]

function statusMeta(status: DocStatus) {
  return STATUS_META[status] ?? STATUS_META.pending
}

function isInFlight(status: DocStatus): boolean {
  return status === 'pending' || status === 'parsing' || status === 'processing'
}

function basename(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean)
  return parts.length ? parts[parts.length - 1] : path
}

/** doc_status stores epoch seconds; the millisecond branch is defensive only. */
function relativeDate(stamp: number): string {
  if (!stamp) return 'date inconnue'
  const ms = stamp < 1e12 ? stamp * 1000 : stamp
  const minutes = Math.floor((Date.now() - ms) / 60000)
  if (minutes < 1) return "à l'instant"
  if (minutes < 60) return `il y a ${minutes} min`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `il y a ${hours} h`
  const days = Math.floor(hours / 24)
  if (days < 7) return `il y a ${days} j`
  return new Intl.DateTimeFormat('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' }).format(new Date(ms))
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.requestId ? `${error.message} (requête ${error.requestId})` : error.message
  }
  return error instanceof Error ? error.message : 'Erreur inconnue'
}

export default function LibraryPanel({ onClose }: { onClose: () => void }) {
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>('tous')
  const [expanded, setExpanded] = useState<string | null>(null)
  const [chunks, setChunks] = useState<ChunkState | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [textDraft, setTextDraft] = useState('')
  const [textOpen, setTextOpen] = useState(false)

  const alive = useRef(true)
  const fileInput = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const response = await listDocuments(PAGE_SIZE, 0)
      if (!alive.current) return
      const next = Object.entries(response.statuses)
        .map(([id, record]) => ({ id, record }))
        .sort((a, b) => b.record.created_at - a.record.created_at)
      setRows(next)
      setError(null)
    } catch (err) {
      if (alive.current) setError(describeError(err))
    } finally {
      if (alive.current && !silent) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const pending = useMemo(() => rows.some((row) => isInFlight(row.record.status)), [rows])

  useEffect(() => {
    if (!pending) return
    const timer = window.setInterval(() => {
      void refresh(true)
    }, POLL_MS)
    return () => window.clearInterval(timer)
  }, [pending, refresh])

  const visible = useMemo(
    () => (filter === 'tous' ? rows : rows.filter((row) => row.record.status === filter)),
    [rows, filter],
  )

  const counts = useMemo(() => {
    const tally = new Map<Filter, number>()
    for (const row of rows) tally.set(row.record.status, (tally.get(row.record.status) ?? 0) + 1)
    tally.set('tous', rows.length)
    return tally
  }, [rows])

  const toggleRow = useCallback(
    async (docId: string) => {
      setConfirmDelete(null)
      if (expanded === docId) {
        setExpanded(null)
        return
      }
      setExpanded(docId)
      setChunks({ docId, loading: true, error: null, data: null })
      try {
        const data = await documentChunks(docId)
        if (alive.current) setChunks({ docId, loading: false, error: null, data })
      } catch (err) {
        if (alive.current) setChunks({ docId, loading: false, error: describeError(err), data: null })
      }
    },
    [expanded],
  )

  const runAction = useCallback(
    async (action: () => Promise<unknown>, message: string) => {
      setBusy(true)
      setNotice(null)
      try {
        await action()
        if (!alive.current) return
        setNotice(message)
        await refresh(true)
      } catch (err) {
        if (alive.current) setError(describeError(err))
      } finally {
        if (alive.current) setBusy(false)
      }
    },
    [refresh],
  )

  const onFiles = useCallback(
    async (list: FileList | null) => {
      if (!list || list.length === 0) return
      const files = Array.from(list)
      await runAction(async () => {
        for (const file of files) await uploadFile(file)
      }, `${files.length} fichier(s) envoyé(s).`)
    },
    [runAction],
  )

  const submitText = useCallback(async () => {
    const text = textDraft.trim()
    if (!text) return
    await runAction(() => insertText(text), 'Texte ajouté à la file.')
    if (alive.current) {
      setTextDraft('')
      setTextOpen(false)
    }
  }, [runAction, textDraft])

  return (
    <div className="fixed inset-0 z-40 flex">
      <button
        type="button"
        aria-label="Fermer la bibliothèque"
        onClick={onClose}
        className="flex-1 bg-ink/10 backdrop-blur-[1px]"
      />
      <aside className="flex h-full w-full max-w-3xl flex-col border-l border-edge bg-surface shadow-[0_0_60px_-20px_rgba(0,0,0,0.3)]">
        <header className="flex items-center gap-3 border-b border-edge px-6 py-4">
          <BookMark />
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-ink">Bibliothèque</h2>
            <p className="truncate text-xs text-ink-muted">
              {rows.length} document(s) indexé(s){pending ? ' · actualisation automatique' : ''}
            </p>
          </div>
          <button type="button" className="icon-btn" onClick={() => void refresh()} title="Actualiser">
            <RefreshIcon size={16} />
          </button>
          <button type="button" className="icon-btn" onClick={onClose} title="Fermer">
            <CloseIcon size={16} />
          </button>
        </header>

        <section className="border-b border-edge bg-surface-sunken px-6 py-4">
          <div className="flex flex-wrap items-center gap-2">
            <input
              ref={fileInput}
              type="file"
              multiple
              className="hidden"
              onChange={(event) => {
                void onFiles(event.target.files)
                event.target.value = ''
              }}
            />
            <button
              type="button"
              className="btn-dark"
              disabled={busy}
              onClick={() => fileInput.current?.click()}
            >
              <FileIcon size={15} />
              Téléverser
            </button>
            <button type="button" className="btn-ghost" disabled={busy} onClick={() => setTextOpen((open) => !open)}>
              Texte brut
            </button>
            <button
              type="button"
              className="btn-ghost"
              disabled={busy}
              onClick={() => void runAction(() => scanInputDir(), "Dossier d'entrée scanné.")}
            >
              <TrayIcon size={15} />
              Scanner le dossier
            </button>
            {busy ? <span className="text-xs text-ink-faint">Envoi en cours…</span> : null}
            {notice ? <span className="text-xs text-violet-600">{notice}</span> : null}
          </div>

          {textOpen ? (
            <div className="mt-3 animate-fade-up">
              <textarea
                value={textDraft}
                onChange={(event) => setTextDraft(event.target.value)}
                rows={4}
                placeholder="Collez ici le texte à indexer…"
                className="w-full resize-y rounded-card border border-edge-strong bg-white px-3 py-2 text-sm
                  text-ink outline-none placeholder:text-ink-faint focus:border-violet-300"
              />
              <div className="mt-2 flex justify-end gap-2">
                <button type="button" className="btn-ghost" onClick={() => setTextOpen(false)}>
                  Annuler
                </button>
                <button type="button" className="btn-dark" disabled={busy || !textDraft.trim()} onClick={() => void submitText()}>
                  Indexer le texte
                </button>
              </div>
            </div>
          ) : null}

          <div className="mt-4 flex flex-wrap gap-1.5">
            {FILTERS.map((entry) => {
              const active = entry.key === filter
              return (
                <button
                  key={entry.key}
                  type="button"
                  onClick={() => setFilter(entry.key)}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition ${
                    active
                      ? 'border-violet-300 bg-violet-50 text-violet-600'
                      : 'border-edge bg-white text-ink-muted hover:text-ink'
                  }`}
                >
                  {entry.key !== 'tous' ? (
                    <span className={`h-1.5 w-1.5 rounded-full ${statusMeta(entry.key).dot}`} />
                  ) : null}
                  {entry.label}
                  <span className="text-ink-faint">{counts.get(entry.key) ?? 0}</span>
                </button>
              )
            })}
          </div>
        </section>

        <div className="flex-1 overflow-y-auto px-6 py-4">
          {error ? (
            <div className="mb-4 rounded-card border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          ) : null}

          {loading ? (
            <p className="py-10 text-center text-sm text-ink-faint">Chargement des documents…</p>
          ) : visible.length === 0 ? (
            <div className="py-14 text-center">
              <p className="text-sm text-ink-muted">Aucun document pour ce filtre.</p>
              <p className="mt-1 text-xs text-ink-faint">
                Téléversez un fichier ou scannez le dossier d&apos;entrée pour alimenter la mémoire.
              </p>
            </div>
          ) : (
            <ul className="space-y-2">
              {visible.map(({ id, record }) => {
                const meta = statusMeta(record.status)
                const open = expanded === id
                return (
                  <li key={id} className="overflow-hidden rounded-card border border-edge bg-white">
                    <div className="flex items-center gap-3 px-4 py-3">
                      <button
                        type="button"
                        onClick={() => void toggleRow(id)}
                        className="flex min-w-0 flex-1 items-center gap-3 text-left"
                      >
                        <span className={`h-2 w-2 shrink-0 rounded-full ${meta.dot}`} />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-medium text-ink">
                            {basename(record.file_path)}
                          </span>
                          <span className="mt-0.5 block truncate text-xs text-ink-faint">
                            {relativeDate(record.created_at)}
                            {typeof record.chunk_count === 'number' ? ` · ${record.chunk_count} passages` : ''}
                            {record.metadata?.memory_sub_stage ? ` · ${record.metadata.memory_sub_stage}` : ''}
                          </span>
                        </span>
                      </button>
                      <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[11px] ${meta.badge}`}>
                        {meta.label}
                      </span>
                      <button
                        type="button"
                        className="icon-btn"
                        title="Relancer"
                        disabled={busy}
                        onClick={() => void runAction(() => requeueDocument(id), 'Document remis en file.')}
                      >
                        <RefreshIcon size={15} />
                      </button>
                      <button
                        type="button"
                        className="icon-btn hover:text-red-600"
                        title="Supprimer"
                        disabled={busy}
                        onClick={() => setConfirmDelete(confirmDelete === id ? null : id)}
                      >
                        <TrashIcon size={15} />
                      </button>
                    </div>

                    {confirmDelete === id ? (
                      <div className="flex items-center justify-between gap-3 border-t border-edge bg-red-50 px-4 py-2">
                        <span className="text-xs text-red-700">
                          Supprimer ce document et ses passages exclusifs ?
                        </span>
                        <span className="flex gap-2">
                          <button type="button" className="btn-ghost" onClick={() => setConfirmDelete(null)}>
                            Annuler
                          </button>
                          <button
                            type="button"
                            className="btn-dark"
                            disabled={busy}
                            onClick={() => {
                              setConfirmDelete(null)
                              if (expanded === id) setExpanded(null)
                              void runAction(() => deleteDocument(id), 'Document supprimé.')
                            }}
                          >
                            Supprimer
                          </button>
                        </span>
                      </div>
                    ) : null}

                    {open ? (
                      <div className="border-t border-edge bg-surface-sunken px-4 py-3">
                        {record.metadata?.error ? (
                          <p className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                            {record.metadata.error}
                          </p>
                        ) : null}
                        <p className="mb-2 text-[11px] uppercase tracking-wide text-ink-faint">
                          {record.file_path}
                        </p>
                        {chunks && chunks.docId === id ? (
                          chunks.loading ? (
                            <p className="py-4 text-center text-xs text-ink-faint">Chargement des passages…</p>
                          ) : chunks.error ? (
                            <p className="text-xs text-red-700">{chunks.error}</p>
                          ) : chunks.data && chunks.data.chunks.length > 0 ? (
                            <>
                              <p className="mb-2 text-xs text-ink-muted">
                                {chunks.data.returned} passage(s) affiché(s) sur {chunks.data.total}
                              </p>
                              <div className="max-h-80 space-y-2 overflow-y-auto pr-1">
                                {chunks.data.chunks.map((chunk) => (
                                  <article
                                    key={chunk.chunk_id}
                                    className="rounded-lg border border-edge bg-white px-3 py-2"
                                  >
                                    <header className="mb-1 flex items-center justify-between text-[11px] text-ink-faint">
                                      <span>#{chunk.order}</span>
                                      <span className="truncate font-mono">{chunk.chunk_id}</span>
                                    </header>
                                    <p className="whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-ink">
                                      {chunk.content}
                                    </p>
                                  </article>
                                ))}
                              </div>
                            </>
                          ) : (
                            <p className="text-xs text-ink-faint">
                              Aucun passage disponible pour ce document.
                            </p>
                          )
                        ) : null}
                      </div>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </aside>
    </div>
  )
}

function BookMark() {
  return (
    <span className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-violet-50 text-violet-600">
      <FileIcon size={17} />
    </span>
  )
}
