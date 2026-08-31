/**
 * Memory-graph explorer.
 *
 * Layout is a deterministic sunflower spiral rather than a force simulation: no extra
 * dependency, stable across renders, and readable up to the 500-node cap.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, exploreGraph, graphLabels } from '../api/client'
import type { GraphEdge, GraphNode, GraphResponse } from '../api/types'
import { CloseIcon, GraphIcon, RefreshIcon } from './icons'

const VIEW_W = 720
const VIEW_H = 520
const RADIUS = 220
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))

const GROUP_COLORS = ['#8B5CF6', '#10B981', '#F59E0B', '#3B82F6', '#EF4444', '#14B8A6', '#EC4899', '#6B7280']

type Placed = { node: GraphNode; x: number; y: number; group: string }
type DrawnEdge = { edge: GraphEdge; from: Placed; to: Placed }

function groupOf(node: GraphNode): string {
  return node.layer ?? node.label ?? 'autre'
}

function isPassage(node: GraphNode): boolean {
  return groupOf(node).toLowerCase().includes('passage')
}

function displayLabel(node: GraphNode): string {
  if (isPassage(node)) {
    const length = typeof node.content_length === 'number' ? node.content_length : null
    return length === null ? 'Passage (masqué)' : `Passage · ${length} car.`
  }
  return node.label ?? node.id
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.requestId ? `${error.message} (requête ${error.requestId})` : error.message
  }
  return error instanceof Error ? error.message : 'Erreur inconnue'
}

function place(nodes: GraphNode[]): { placed: Placed[]; groups: string[] } {
  const groups: string[] = []
  for (const node of nodes) {
    const group = groupOf(node)
    if (!groups.includes(group)) groups.push(group)
  }
  const total = Math.max(nodes.length, 1)
  const placed = nodes.map((node, index) => {
    const radius = RADIUS * Math.sqrt((index + 0.5) / total)
    const angle = index * GOLDEN_ANGLE
    return {
      node,
      x: VIEW_W / 2 + radius * Math.cos(angle),
      y: VIEW_H / 2 + radius * Math.sin(angle),
      group: groupOf(node),
    }
  })
  return { placed, groups }
}

function colorFor(groups: string[], group: string): string {
  const index = groups.indexOf(group)
  return GROUP_COLORS[(index < 0 ? 0 : index) % GROUP_COLORS.length]
}

export default function GraphPanel({ onClose }: { onClose: () => void }) {
  const [labels, setLabels] = useState<string[]>([])
  const [label, setLabel] = useState('')
  const [limit, setLimit] = useState(120)
  const [graph, setGraph] = useState<GraphResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<GraphNode | null>(null)

  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    graphLabels()
      .then((response) => {
        if (!cancelled) setLabels(response.labels)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(describeError(err))
      })
    return () => {
      cancelled = true
    }
  }, [])

  const explore = useCallback(async () => {
    setLoading(true)
    setError(null)
    setSelected(null)
    try {
      const response = await exploreGraph(label || null, limit)
      if (alive.current) setGraph(response)
    } catch (err) {
      if (alive.current) setError(describeError(err))
    } finally {
      if (alive.current) setLoading(false)
    }
  }, [label, limit])

  const { placed, groups } = useMemo(() => place(graph?.nodes ?? []), [graph])

  const positions = useMemo(() => {
    const map = new Map<string, Placed>()
    for (const item of placed) map.set(item.node.id, item)
    return map
  }, [placed])

  const edges = useMemo<DrawnEdge[]>(() => {
    if (!graph) return []
    const drawn: DrawnEdge[] = []
    for (const edge of graph.edges) {
      const from = positions.get(edge.source)
      const to = positions.get(edge.target)
      if (from && to) drawn.push({ edge, from, to })
    }
    return drawn
  }, [graph, positions])

  return (
    <div className="fixed inset-0 z-40 flex">
      <button type="button" aria-label="Fermer le graphe" onClick={onClose} className="flex-1 bg-ink/10 backdrop-blur-[1px]" />
      <aside className="flex h-full w-full max-w-5xl flex-col border-l border-edge bg-surface shadow-[0_0_60px_-20px_rgba(0,0,0,0.3)]">
        <header className="flex items-center gap-3 border-b border-edge px-6 py-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-violet-50 text-violet-600">
            <GraphIcon size={17} />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-ink">Graphe mémoire</h2>
            <p className="truncate text-xs text-ink-muted">Schémas, faits et passages installés dans le graphe</p>
          </div>
          <button type="button" className="icon-btn" onClick={onClose} title="Fermer">
            <CloseIcon size={16} />
          </button>
        </header>

        <section className="flex flex-wrap items-end gap-3 border-b border-edge bg-surface-sunken px-6 py-4">
          <label className="flex flex-col gap-1 text-xs text-ink-muted">
            Libellé
            <select
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              className="min-w-[200px] rounded-full border border-edge-strong bg-white px-3 py-1.5 text-sm text-ink outline-none focus:border-violet-300"
            >
              <option value="">(tous)</option>
              {labels.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-xs text-ink-muted">
            Nœuds max ({limit})
            <input
              type="number"
              min={10}
              max={500}
              step={10}
              value={limit}
              onChange={(event) => {
                const parsed = Number(event.target.value)
                setLimit(Number.isFinite(parsed) ? Math.min(500, Math.max(10, Math.round(parsed))) : 120)
              }}
              className="w-28 rounded-full border border-edge-strong bg-white px-3 py-1.5 text-sm text-ink outline-none focus:border-violet-300"
            />
          </label>

          <button type="button" className="btn-dark" disabled={loading} onClick={() => void explore()}>
            <RefreshIcon size={15} />
            {loading ? 'Exploration…' : 'Explorer'}
          </button>

          {graph ? (
            <p className="ml-auto text-xs text-ink-muted">
              {graph.returned_nodes} nœuds affichés sur {graph.total_nodes} · {graph.returned_edges} arêtes
            </p>
          ) : null}
        </section>

        {error ? (
          <p className="mx-6 mt-4 rounded-card border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>
        ) : null}

        {graph?.truncated ? (
          <p className="mx-6 mt-4 rounded-card border border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-700">
            Résultat tronqué : augmentez la limite pour voir davantage de nœuds.
          </p>
        ) : null}

        <div className="flex min-h-0 flex-1 gap-4 px-6 py-4">
          <div className="flex min-w-0 flex-1 flex-col rounded-panel border border-edge bg-white p-3">
            {!graph ? (
              <p className="m-auto text-sm text-ink-faint">
                Choisissez un libellé puis lancez « Explorer » pour dessiner le graphe.
              </p>
            ) : placed.length === 0 ? (
              <p className="m-auto text-sm text-ink-faint">Aucun nœud pour ce libellé.</p>
            ) : (
              <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="h-full w-full" role="img" aria-label="Graphe mémoire">
                <g stroke="#E2E2E2" strokeWidth={1}>
                  {edges.map(({ edge, from, to }, index) => (
                    <line key={`${edge.source}-${edge.target}-${index}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} />
                  ))}
                </g>
                <g>
                  {placed.map((item) => {
                    const active = selected?.id === item.node.id
                    return (
                      <circle
                        key={item.node.id}
                        cx={item.x}
                        cy={item.y}
                        r={active ? 9 : 6}
                        fill={colorFor(groups, item.group)}
                        stroke={active ? '#111111' : '#FFFFFF'}
                        strokeWidth={active ? 2 : 1.5}
                        className="cursor-pointer"
                        onClick={() => setSelected(item.node)}
                      >
                        <title>{displayLabel(item.node)}</title>
                      </circle>
                    )
                  })}
                </g>
              </svg>
            )}

            <div className="mt-2 flex flex-wrap items-center gap-3 border-t border-edge pt-2">
              {groups.map((group) => (
                <span key={group} className="inline-flex items-center gap-1.5 text-[11px] text-ink-muted">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: colorFor(groups, group) }} />
                  {group}
                </span>
              ))}
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">
              Le serveur masque le texte des nœuds Passage : ils n&apos;ont donc aucun libellé lisible et seule leur
              longueur (<code>content_length</code>) est affichée.
            </p>
          </div>

          <div className="flex w-72 shrink-0 flex-col overflow-y-auto rounded-panel border border-edge bg-surface-sunken p-4">
            <h3 className="text-xs uppercase tracking-wide text-ink-faint">Détail du nœud</h3>
            {!selected ? (
              <p className="mt-3 text-sm text-ink-faint">Cliquez sur un nœud pour voir ses propriétés.</p>
            ) : (
              <>
                <p className="mt-3 break-all font-mono text-xs text-ink">{selected.id}</p>
                <p className="mt-1 text-sm font-medium text-ink">{displayLabel(selected)}</p>
                <dl className="mt-4 space-y-2">
                  {Object.entries(selected)
                    .filter(([key]) => key !== 'id')
                    .map(([key, value]) => (
                      <div key={key} className="rounded-lg border border-edge bg-white px-3 py-2">
                        <dt className="text-[11px] uppercase tracking-wide text-ink-faint">{key}</dt>
                        <dd className="mt-0.5 break-words text-xs text-ink">{formatValue(value)}</dd>
                      </div>
                    ))}
                </dl>
              </>
            )}
          </div>
        </div>
      </aside>
    </div>
  )
}
