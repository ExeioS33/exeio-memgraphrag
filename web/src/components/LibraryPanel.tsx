/**
 * Library browser: the source corpus on disk, not the ingestion queue.
 *
 * This panel deliberately reads /library/* rather than /documents/: doc-status is
 * empty on this deployment, so a queue view showed nothing while 23 PDFs sat in the
 * configured root. Three panes — tree, document, provenance — over the same root the
 * server exposes.
 *
 * Two shapes are worth knowing before editing:
 *  - the PDF itself is rendered by the browser through an <iframe>, which cannot send
 *    an Authorization header; `libraryFileUrl` puts the token in the query string and
 *    the server accepts it for that one read-only route.
 *  - /library/passages answers `linked: false` when no Passage node carries a
 *    file_path at all. That is a missing provenance backfill, not an empty document,
 *    and the right pane says so instead of rendering a bare "aucun résultat".
 */
import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  ApiError,
  libraryFileUrl,
  libraryPassages,
  libraryPreview,
  libraryTree,
} from '../api/client'
import type { LibraryEntry, LibraryPassages, LibraryPreview, LibraryTree } from '../api/types'
import {
  BookIcon,
  ChevronDownIcon,
  CloseIcon,
  DownloadIcon,
  FileIcon,
  LayersIcon,
  LinkIcon,
  RefreshIcon,
  SearchIcon,
} from './icons'

const PASSAGE_LIMIT = 30

/* ------------------------------------------------------------------ format --- */

function humanSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 o'
  const units = ['o', 'Ko', 'Mo', 'Go', 'To']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  const digits = unit === 0 || value >= 100 ? 0 : 1
  return `${value.toFixed(digits).replace('.', ',')} ${units[unit]}`
}

/**
 * The tree endpoint sends `modified` as an ISO 8601 string; the numeric branches are
 * defensive so a server switched to epoch seconds/ms does not print "date inconnue".
 */
function formatDate(value: number | string): string {
  const ms =
    typeof value === 'number' ? (value < 1e12 ? value * 1000 : value) : Date.parse(String(value))
  if (!ms || Number.isNaN(ms)) return 'date inconnue'
  return new Intl.DateTimeFormat('fr-FR', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  }).format(new Date(ms))
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.requestId ? `${error.message} (requête ${error.requestId})` : error.message
  }
  return error instanceof Error ? error.message : 'Erreur inconnue'
}

/* -------------------------------------------------------------------- tree --- */

/** Directories first, then by name — the order the server already uses, re-applied
 *  because the filter rebuilds the arrays. */
function sortEntries(entries: LibraryEntry[]): LibraryEntry[] {
  const ordered = [...entries].sort((a, b) => {
    if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1
    return a.name.localeCompare(b.name, 'fr', { numeric: true, sensitivity: 'base' })
  })
  return ordered.map((entry) =>
    entry.is_dir ? { ...entry, children: sortEntries(entry.children ?? []) } : entry,
  )
}

/** Keep files whose name matches, and directories that match or still hold a match. */
function pruneTree(entries: LibraryEntry[], needle: string): LibraryEntry[] {
  const out: LibraryEntry[] = []
  for (const entry of entries) {
    const hit = entry.name.toLowerCase().includes(needle)
    if (entry.is_dir) {
      if (hit) {
        out.push(entry)
        continue
      }
      const children = pruneTree(entry.children ?? [], needle)
      if (children.length > 0) out.push({ ...entry, children })
    } else if (hit) {
      out.push(entry)
    }
  }
  return out
}

function countFiles(entry: LibraryEntry): number {
  if (!entry.is_dir) return 1
  return (entry.children ?? []).reduce((sum, child) => sum + countFiles(child), 0)
}

function collectDirs(entries: LibraryEntry[], into: Set<string>): Set<string> {
  for (const entry of entries) {
    if (!entry.is_dir) continue
    into.add(entry.path)
    collectDirs(entry.children ?? [], into)
  }
  return into
}

function firstFile(entries: LibraryEntry[]): LibraryEntry | null {
  for (const entry of entries) {
    if (!entry.is_dir) return entry
    const nested = firstFile(entry.children ?? [])
    if (nested) return nested
  }
  return null
}

type TreeNodeProps = {
  entry: LibraryEntry
  depth: number
  selectedPath: string | null
  openDirs: ReadonlySet<string>
  /** While a filter is active every surviving directory stays open. */
  forceOpen: boolean
  onToggleDir: (path: string) => void
  onSelectFile: (entry: LibraryEntry) => void
}

function TreeNode({
  entry,
  depth,
  selectedPath,
  openDirs,
  forceOpen,
  onToggleDir,
  onSelectFile,
}: TreeNodeProps) {
  const indent = { paddingLeft: 10 + depth * 14 }

  if (entry.is_dir) {
    const open = forceOpen || openDirs.has(entry.path)
    const children = entry.children ?? []
    return (
      <li>
        <button
          type="button"
          onClick={() => onToggleDir(entry.path)}
          aria-expanded={open}
          style={indent}
          className="flex w-full items-center gap-1.5 rounded-lg py-1.5 pr-2 text-left text-sm
            text-ink transition hover:bg-white"
        >
          <ChevronDownIcon
            size={14}
            className={`shrink-0 text-ink-faint transition-transform ${open ? '' : '-rotate-90'}`}
          />
          <span className="min-w-0 flex-1 truncate font-medium">{entry.name}</span>
          <span className="shrink-0 text-[11px] text-ink-faint">{countFiles(entry)}</span>
        </button>
        {open && children.length > 0 ? (
          <ul>
            {children.map((child) => (
              <TreeNode
                key={child.path}
                entry={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                openDirs={openDirs}
                forceOpen={forceOpen}
                onToggleDir={onToggleDir}
                onSelectFile={onSelectFile}
              />
            ))}
          </ul>
        ) : null}
      </li>
    )
  }

  const active = entry.path === selectedPath
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelectFile(entry)}
        style={indent}
        className={`flex w-full items-start gap-2 rounded-lg py-1.5 pr-2 text-left transition ${
          active ? 'bg-violet-50 text-violet-700' : 'text-ink hover:bg-white'
        }`}
      >
        <FileIcon
          size={14}
          className={`mt-0.5 shrink-0 ${active ? 'text-violet-600' : 'text-ink-faint'}`}
        />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm">{entry.name}</span>
          <span
            className={`mt-0.5 block truncate text-[11px] ${
              active ? 'text-violet-600' : 'text-ink-faint'
            }`}
          >
            {humanSize(entry.size)} · {formatDate(entry.modified)}
          </span>
        </span>
      </button>
    </li>
  )
}

/* ------------------------------------------------------------------- panel --- */

export default function LibraryPanel({ onClose }: { onClose: () => void }) {
  const [tree, setTree] = useState<LibraryTree | null>(null)
  const [treeLoading, setTreeLoading] = useState(true)
  const [treeError, setTreeError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [openDirs, setOpenDirs] = useState<Set<string>>(() => new Set())

  const [selected, setSelected] = useState<LibraryEntry | null>(null)
  const [page, setPage] = useState(1)
  const [showText, setShowText] = useState(false)

  const [preview, setPreview] = useState<LibraryPreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  const [passages, setPassages] = useState<LibraryPassages | null>(null)
  const [passagesLoading, setPassagesLoading] = useState(false)
  const [passagesError, setPassagesError] = useState<string | null>(null)

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let cancelled = false
    setTreeLoading(true)
    libraryTree()
      .then((data) => {
        if (cancelled) return
        setTree(data)
        setTreeError(null)
        setOpenDirs(collectDirs(data.entries, new Set<string>()))
        // Opening on an empty centre pane would read as a broken panel; the first
        // file is as good a default as any and the corpus is small.
        setSelected((current) => current ?? firstFile(data.entries))
      })
      .catch((error: unknown) => {
        if (!cancelled) setTreeError(describeError(error))
      })
      .finally(() => {
        if (!cancelled) setTreeLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [reloadToken])

  const selectedPath = selected?.path ?? null

  // One page at a time: the window exists to bound the JSON body, and the reader
  // moves page by page.
  useEffect(() => {
    if (!selectedPath) {
      setPreview(null)
      setPreviewError(null)
      return
    }
    let cancelled = false
    setPreviewLoading(true)
    libraryPreview(selectedPath, page, 1)
      .then((data) => {
        if (cancelled) return
        setPreview(data)
        setPreviewError(null)
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setPreview(null)
        setPreviewError(describeError(error))
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedPath, page])

  useEffect(() => {
    if (!selectedPath) {
      setPassages(null)
      setPassagesError(null)
      return
    }
    let cancelled = false
    setPassagesLoading(true)
    libraryPassages(selectedPath, PASSAGE_LIMIT)
      .then((data) => {
        if (cancelled) return
        setPassages(data)
        setPassagesError(null)
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setPassages(null)
        setPassagesError(describeError(error))
      })
      .finally(() => {
        if (!cancelled) setPassagesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedPath])

  const needle = filter.trim().toLowerCase()

  const visible = useMemo(() => {
    if (!tree) return []
    const pruned = needle ? pruneTree(tree.entries, needle) : tree.entries
    return sortEntries(pruned)
  }, [tree, needle])

  const visibleFiles = useMemo(
    () => visible.reduce((sum, entry) => sum + countFiles(entry), 0),
    [visible],
  )

  const toggleDir = useCallback((path: string) => {
    setOpenDirs((current) => {
      const next = new Set(current)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }, [])

  const selectFile = useCallback((entry: LibraryEntry) => {
    setSelected(entry)
    setPage(1)
    setShowText(false)
  }, [])

  const pageCount = preview?.page_count ?? null
  const pageText = preview?.pages[0]?.text ?? ''
  const fileUrl = selected ? libraryFileUrl(selected.path) : null
  const isPdf = selected?.ext === 'pdf'

  return (
    <div className="fixed inset-0 z-40 p-3 sm:p-5">
      <button
        type="button"
        aria-label="Fermer la bibliothèque"
        onClick={onClose}
        className="absolute inset-0 bg-ink/25 backdrop-blur-[1px]"
      />

      <section
        className="relative flex h-full w-full flex-col overflow-hidden rounded-panel border
          border-edge bg-surface shadow-[0_30px_80px_-30px_rgba(0,0,0,0.45)]"
      >
        <header className="flex items-center gap-3 border-b border-edge px-5 py-3.5">
          <span
            className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-violet-50
              text-violet-600"
          >
            <BookIcon size={17} />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-ink">Bibliothèque</h2>
            <p className="truncate text-xs text-ink-muted">
              {tree ? `${tree.total_files} fichier(s) · ${tree.root}` : 'Dossier source'}
            </p>
          </div>
          <button
            type="button"
            className="icon-btn"
            title="Actualiser l'arborescence"
            onClick={() => setReloadToken((value) => value + 1)}
          >
            <RefreshIcon size={16} />
          </button>
          <button type="button" className="icon-btn" title="Fermer" onClick={onClose}>
            <CloseIcon size={16} />
          </button>
        </header>

        <div className="flex min-h-0 flex-1">
          {/* ------------------------------------------------------------ left --- */}
          <aside className="flex w-72 shrink-0 flex-col border-r border-edge bg-surface-sunken">
            <div className="border-b border-edge px-3 py-3">
              <label className="relative block">
                <span className="sr-only">Filtrer les fichiers</span>
                <SearchIcon
                  size={15}
                  className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2
                    text-ink-faint"
                />
                <input
                  type="search"
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                  placeholder="Filtrer par nom…"
                  className="w-full rounded-full border border-edge bg-white py-1.5 pl-9 pr-3
                    text-sm text-ink outline-none placeholder:text-ink-faint
                    focus:border-violet-300"
                />
              </label>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
              {treeLoading ? (
                <p className="px-2 py-8 text-center text-xs text-ink-faint">
                  Chargement de l&apos;arborescence…
                </p>
              ) : treeError ? (
                <p
                  className="mx-1 rounded-card border border-red-200 bg-red-50 px-3 py-2 text-xs
                    text-red-700"
                >
                  {treeError}
                </p>
              ) : visible.length === 0 ? (
                <p className="px-2 py-8 text-center text-xs text-ink-faint">
                  {needle
                    ? `Aucun fichier ne correspond à « ${filter.trim()} ».`
                    : 'La bibliothèque est vide. Vérifiez LIBRARY_ROOT côté serveur.'}
                </p>
              ) : (
                <ul>
                  {visible.map((entry) => (
                    <TreeNode
                      key={entry.path}
                      entry={entry}
                      depth={0}
                      selectedPath={selectedPath}
                      openDirs={openDirs}
                      forceOpen={needle.length > 0}
                      onToggleDir={toggleDir}
                      onSelectFile={selectFile}
                    />
                  ))}
                </ul>
              )}
            </div>

            {tree && !treeLoading && !treeError ? (
              <p className="border-t border-edge px-4 py-2 text-[11px] text-ink-faint">
                {needle
                  ? `${visibleFiles} fichier(s) affiché(s) sur ${tree.total_files}`
                  : `${tree.total_files} fichier(s)`}
              </p>
            ) : null}
          </aside>

          {/* ---------------------------------------------------------- centre --- */}
          <div className="flex min-w-0 flex-1 flex-col">
            {!selected ? (
              <div className="flex flex-1 items-center justify-center px-8 text-center">
                <p className="max-w-sm text-sm text-ink-muted">
                  Sélectionnez un document dans l&apos;arborescence pour l&apos;afficher.
                </p>
              </div>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-3 border-b border-edge px-5 py-3">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-ink">{selected.name}</p>
                    <p className="truncate text-xs text-ink-faint">
                      {humanSize(selected.size)} ·{' '}
                      {previewLoading && pageCount === null
                        ? 'pages…'
                        : pageCount !== null
                          ? `${pageCount} page(s)`
                          : 'pagination indisponible'}{' '}
                      · {formatDate(selected.modified)}
                    </p>
                  </div>
                  <a
                    className="btn-dark"
                    href={fileUrl ?? '#'}
                    target="_blank"
                    rel="noreferrer noopener"
                  >
                    <LinkIcon size={15} />
                    {isPdf ? 'Ouvrir le PDF' : 'Ouvrir le fichier'}
                  </a>
                  <a className="btn-ghost" href={fileUrl ?? '#'} download={selected.name}>
                    <DownloadIcon size={15} />
                    Télécharger
                  </a>
                </div>

                <div
                  className="flex flex-wrap items-center gap-2 border-b border-edge
                    bg-surface-sunken px-5 py-2"
                >
                  <button
                    type="button"
                    onClick={() => setShowText(false)}
                    className={`rounded-full border px-3 py-1 text-xs transition ${
                      showText
                        ? 'border-edge bg-white text-ink-muted hover:text-ink'
                        : 'border-violet-300 bg-violet-50 text-violet-600'
                    }`}
                  >
                    Document
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowText(true)}
                    className={`rounded-full border px-3 py-1 text-xs transition ${
                      showText
                        ? 'border-violet-300 bg-violet-50 text-violet-600'
                        : 'border-edge bg-white text-ink-muted hover:text-ink'
                    }`}
                  >
                    Texte extrait
                  </button>

                  {showText ? (
                    <span className="ml-auto flex items-center gap-2">
                      <button
                        type="button"
                        className="btn-ghost"
                        disabled={page <= 1 || previewLoading}
                        onClick={() => setPage((value) => Math.max(1, value - 1))}
                      >
                        Page précédente
                      </button>
                      <span className="text-xs text-ink-muted">
                        Page {page}
                        {pageCount !== null ? ` sur ${pageCount}` : ''}
                      </span>
                      <button
                        type="button"
                        className="btn-ghost"
                        disabled={previewLoading || (pageCount !== null && page >= pageCount)}
                        onClick={() => setPage((value) => value + 1)}
                      >
                        Page suivante
                      </button>
                    </span>
                  ) : null}
                </div>

                <div className="min-h-0 flex-1 overflow-hidden bg-surface-sunken">
                  {showText ? (
                    <div className="h-full overflow-y-auto px-5 py-4">
                      {previewLoading ? (
                        <p className="py-10 text-center text-sm text-ink-faint">
                          Extraction du texte…
                        </p>
                      ) : previewError ? (
                        <p
                          className="rounded-card border border-red-200 bg-red-50 px-4 py-3 text-sm
                            text-red-700"
                        >
                          Texte indisponible : {previewError}
                        </p>
                      ) : pageText.trim() ? (
                        <article className="rounded-card border border-edge bg-white px-5 py-4">
                          <p
                            className="whitespace-pre-wrap font-mono text-[12.5px] leading-relaxed
                              text-ink"
                          >
                            {pageText}
                          </p>
                        </article>
                      ) : (
                        <p className="py-10 text-center text-sm text-ink-muted">
                          Cette page ne contient aucun texte extractible — elle est probablement
                          numérisée en image. Utilisez l&apos;onglet « Document ».
                        </p>
                      )}
                    </div>
                  ) : fileUrl ? (
                    <iframe
                      key={selected.path}
                      src={fileUrl}
                      title={`Aperçu de ${selected.name}`}
                      className="h-full w-full border-0 bg-white"
                    />
                  ) : null}
                </div>
              </>
            )}
          </div>

          {/* ----------------------------------------------------------- right --- */}
          <aside className="flex w-80 shrink-0 flex-col border-l border-edge bg-surface">
            <div className="flex items-center gap-2 border-b border-edge px-4 py-3">
              <LayersIcon size={16} className="text-ink-faint" />
              <h3 className="flex-1 text-sm font-medium text-ink">Passages liés</h3>
              {passages && passages.linked ? (
                <span className="text-[11px] text-ink-faint">{passages.total}</span>
              ) : null}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
              {!selected ? (
                <p className="py-8 text-center text-xs text-ink-faint">
                  Aucun document sélectionné.
                </p>
              ) : passagesLoading ? (
                <p className="py-8 text-center text-xs text-ink-faint">
                  Recherche des passages…
                </p>
              ) : passagesError ? (
                <p
                  className="rounded-card border border-red-200 bg-red-50 px-3 py-2 text-xs
                    text-red-700"
                >
                  {passagesError}
                </p>
              ) : passages && passages.passages.length > 0 ? (
                <ul className="space-y-2">
                  {passages.passages.map((passage) => (
                    <li
                      key={passage.chunk_id}
                      className="rounded-card border border-edge bg-white px-3 py-2"
                    >
                      <p className="mb-1 truncate font-mono text-[11px] text-ink-faint">
                        {passage.chunk_id}
                      </p>
                      <p className="whitespace-pre-wrap text-[12.5px] leading-relaxed text-ink">
                        {passage.content}
                      </p>
                    </li>
                  ))}
                </ul>
              ) : passages && !passages.linked ? (
                <div
                  className="rounded-card border border-amber-200 bg-amber-50 px-3 py-3 text-xs
                    leading-relaxed text-amber-800"
                >
                  <p className="font-medium">Provenance non renseignée</p>
                  <p className="mt-1">
                    Aucun passage du graphe ne porte de chemin de fichier : la provenance
                    n&apos;a pas encore été réinjectée pour ce corpus. Le graphe ne peut donc
                    pas rattacher ses passages à ce document.
                  </p>
                  <p className="mt-1">
                    Il ne s&apos;agit pas d&apos;un document vide : relancez la réinjection de
                    provenance pour que ce panneau se remplisse.
                  </p>
                </div>
              ) : (
                <div
                  className="rounded-card border border-edge bg-surface-sunken px-3 py-3 text-xs
                    leading-relaxed text-ink-muted"
                >
                  <p className="font-medium text-ink">Aucun passage pour ce fichier</p>
                  <p className="mt-1">
                    Le graphe porte bien la provenance d&apos;autres documents, mais aucun
                    passage ne référence celui-ci : il n&apos;a probablement jamais été ingéré.
                  </p>
                </div>
              )}
            </div>
          </aside>
        </div>
      </section>
    </div>
  )
}
