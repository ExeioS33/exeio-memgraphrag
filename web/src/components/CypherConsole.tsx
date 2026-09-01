/**
 * Cypher console — the Neo4j Browser's information architecture, in our design language.
 *
 * Three columns, like the reference: the database inventory on the left (counts, labels,
 * relationship types, property keys), the editor and its result in the middle, the
 * result summary on the right.
 *
 * The graph view lays nodes out on a deterministic golden-angle spiral rather than a
 * force simulation: no extra dependency (the project has none and adds none), stable
 * across re-renders, and readable up to the 300-node cap. It is a *readable arrangement*,
 * not a layout that means anything — distance between two nodes carries no information.
 *
 * The console is read-only: /graph/cypher refuses writes server-side, and a refusal is
 * surfaced as such instead of as a generic 400.
 */
import type { ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, graphNeighborhood, graphSchema, runCypher } from '../api/client'
import type { CypherEdge, CypherNode, CypherResponse, GraphSchema } from '../api/types'
import { CloseIcon, FileIcon, GraphIcon, LayersIcon, RefreshIcon, SendIcon } from './icons'

const VIEW_W = 960
const VIEW_H = 640
const RADIUS = 285
const NODE_RADIUS = 7.5
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))

/** Beyond this the spiral turns into a grey disc, so rendering stops and says so. */
const MAX_DRAWN_NODES = 300
/** Edge type captions only stay legible while the drawing is sparse. */
const MAX_EDGE_CAPTIONS = 70
/** Node captions cost more room than edge captions. */
const MAX_NODE_CAPTIONS = 70

const DEFAULT_LIMIT = 200
const MAX_LIMIT = 5000

const NODE_COLORS = [
  '#8B5CF6',
  '#10B981',
  '#F59E0B',
  '#3B82F6',
  '#EF4444',
  '#14B8A6',
  '#EC4899',
  '#6366F1',
  '#84CC16',
  '#0EA5E9',
]

const INITIAL_QUERY = 'MATCH p=()-[:ENTITY_TO_TYPE]->()\nRETURN p\nLIMIT 25'

/**
 * Presets are written against the memory graph's own labels (Entity / Fact / Schema /
 * Type / Passage) and edge types (ENTITY_RELATION, ENTITY_TO_TYPE, FACT_PASSAGE,
 * PASSAGE_ENTITY, TYPE_RELATION), which is what `_install_memory_graph` writes.
 */
const PRESETS: { label: string; query: string }[] = [
  {
    label: 'Entités → types',
    query: 'MATCH p=()-[:ENTITY_TO_TYPE]->()\nRETURN p\nLIMIT 25',
  },
  {
    label: 'Entités par degré',
    query:
      'MATCH (e:Entity)-[r:ENTITY_RELATION]-()\n' +
      'RETURN e.content AS entite, count(r) AS degre\n' +
      'ORDER BY degre DESC\n' +
      'LIMIT 25',
  },
  {
    label: 'Faits ↔ passages',
    query: 'MATCH p=(:Fact)-[:FACT_PASSAGE]->(:Passage)\nRETURN p\nLIMIT 25',
  },
  {
    label: "Passages d'un document",
    query:
      'MATCH (p:Passage)<-[:PASSAGE_ENTITY]-(e:Entity)\n' +
      "WHERE p.entity_id STARTS WITH 'chunk-'\n" +
      'RETURN p.entity_id AS passage, collect(e.content)[..6] AS entites\n' +
      'LIMIT 25',
  },
  {
    label: 'Schémas fréquents',
    query:
      'MATCH (s:Schema)\n' +
      'RETURN s.content AS schema, s.frequency AS frequence\n' +
      'ORDER BY frequence DESC\n' +
      'LIMIT 25',
  },
  {
    label: 'Échantillon du graphe',
    query: 'MATCH p=()-[]->()\nRETURN p\nLIMIT 50',
  },
]

type Tab = 'graph' | 'table' | 'raw'

type Placed = { node: CypherNode; x: number; y: number; group: string }

type DrawnEdge = {
  edge: CypherEdge
  x1: number
  y1: number
  x2: number
  y2: number
  loop: boolean
}

const numberFormat = new Intl.NumberFormat('fr-FR')

function formatCount(value: number): string {
  return numberFormat.format(value)
}

/** Hashed rather than index-based so a label keeps its colour between two results. */
function colorForGroup(group: string): string {
  let hash = 0
  for (let index = 0; index < group.length; index += 1) {
    hash = (hash * 31 + group.charCodeAt(index)) % 100003
  }
  return NODE_COLORS[hash % NODE_COLORS.length]
}

/** The server sorts the typed label first and the workspace label last. */
function groupOf(node: CypherNode, workspace: string | null): string {
  const typed = node.labels.filter((label) => label !== workspace)
  return typed[0] ?? node.labels[0] ?? 'Nœud'
}

function captionOf(node: CypherNode): string {
  const content = node.properties.content
  if (typeof content === 'string' && content.trim()) return content.trim()
  const entityId = node.properties.entity_id
  if (typeof entityId === 'string' && entityId) return entityId
  return node.id
}

function entityIdOf(node: CypherNode): string {
  const entityId = node.properties.entity_id
  return typeof entityId === 'string' && entityId ? entityId : node.id
}

function clip(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, Math.max(1, max - 1))}…` : text
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.requestId ? `${error.message} (requête ${error.requestId})` : error.message
  }
  return error instanceof Error ? error.message : 'Erreur inconnue'
}

/** The server's own wording for a refused write, matched so the hint can be added. */
function isReadOnlyRejection(message: string): boolean {
  return /lecture seule|refus/i.test(message)
}

function place(nodes: CypherNode[], workspace: string | null): Placed[] {
  const total = Math.max(nodes.length, 1)
  return nodes.map((node, index) => {
    const radius = RADIUS * Math.sqrt((index + 0.5) / total)
    const angle = index * GOLDEN_ANGLE
    return {
      node,
      x: VIEW_W / 2 + radius * Math.cos(angle),
      y: VIEW_H / 2 + radius * Math.sin(angle),
      group: groupOf(node, workspace),
    }
  })
}

function mergeGraph(base: CypherResponse, extra: CypherResponse): CypherResponse {
  const nodes = [...base.nodes]
  const seenNodes = new Set(nodes.map((node) => node.id))
  for (const node of extra.nodes) {
    if (seenNodes.has(node.id)) continue
    seenNodes.add(node.id)
    nodes.push(node)
  }
  const edges = [...base.edges]
  const seenEdges = new Set(edges.map((edge) => edge.id))
  for (const edge of extra.edges) {
    if (seenEdges.has(edge.id)) continue
    seenEdges.add(edge.id)
    edges.push(edge)
  }
  return { ...base, nodes, edges }
}

function countBy<T>(items: T[], key: (item: T) => string): [string, number][] {
  const counts = new Map<string, number>()
  for (const item of items) {
    const bucket = key(item)
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
}

/* --------------------------------------------------------------- pieces --- */

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="border-t border-edge px-4 py-3">
      <h4 className="mb-2 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
        {title}
      </h4>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </section>
  )
}

function CountPill({
  name,
  count,
  color,
  onClick,
  title,
}: {
  name: string
  count?: number
  color?: string
  onClick?: () => void
  title?: string
}) {
  const body = (
    <>
      {color ? (
        <span
          className="h-2 w-2 shrink-0 rounded-full"
          style={{ backgroundColor: color }}
          aria-hidden="true"
        />
      ) : null}
      <span className="truncate">{name}</span>
      {count === undefined ? null : (
        <span className="shrink-0 rounded-full bg-white px-1.5 text-[10px] text-ink-muted">
          {formatCount(count)}
        </span>
      )}
    </>
  )
  const shared =
    'inline-flex max-w-full items-center gap-1.5 rounded-full border border-edge ' +
    'bg-surface-sunken px-2 py-1 text-[11px] text-ink'
  if (!onClick) {
    return (
      <span className={shared} title={title ?? name}>
        {body}
      </span>
    )
  }
  return (
    <button
      type="button"
      onClick={onClick}
      title={title ?? name}
      className={`${shared} transition hover:border-violet-300 hover:bg-violet-50`}
    >
      {body}
    </button>
  )
}

/* ------------------------------------------------------------ component --- */

export default function CypherConsole() {
  const [schema, setSchema] = useState<GraphSchema | null>(null)
  const [schemaError, setSchemaError] = useState<string | null>(null)
  const [schemaLoading, setSchemaLoading] = useState(false)

  const [query, setQuery] = useState(INITIAL_QUERY)
  const [limit, setLimit] = useState(DEFAULT_LIMIT)
  const [result, setResult] = useState<CypherResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('graph')
  const [selected, setSelected] = useState<CypherNode | null>(null)

  const alive = useRef(true)
  const resultRef = useRef<CypherResponse | null>(null)
  const expanding = useRef(false)
  const editorRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    resultRef.current = result
  }, [result])

  const loadSchema = useCallback(async () => {
    setSchemaLoading(true)
    setSchemaError(null)
    try {
      const response = await graphSchema()
      if (alive.current) setSchema(response)
    } catch (err) {
      if (alive.current) setSchemaError(describeError(err))
    } finally {
      if (alive.current) setSchemaLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadSchema()
  }, [loadSchema])

  const execute = useCallback(
    async (statement?: string) => {
      const text = (statement ?? query).trim()
      if (!text) {
        setError('Saisissez une requête Cypher.')
        return
      }
      setLoading(true)
      setError(null)
      setNotice(null)
      setSelected(null)
      try {
        const response = await runCypher(text, limit)
        if (!alive.current) return
        setResult(response)
        setTab(response.nodes.length > 0 ? 'graph' : 'table')
      } catch (err) {
        if (!alive.current) return
        setError(describeError(err))
        setResult(null)
      } finally {
        if (alive.current) setLoading(false)
      }
    },
    [limit, query],
  )

  const expand = useCallback(async (node: CypherNode) => {
    if (expanding.current) return
    expanding.current = true
    const name = clip(captionOf(node), 40)
    setNotice(`Expansion de « ${name} »…`)
    setError(null)
    try {
      const extra = await graphNeighborhood(entityIdOf(node), 1, 100)
      if (!alive.current) return
      const base = resultRef.current
      const merged = base ? mergeGraph(base, extra) : extra
      const added = merged.nodes.length - (base?.nodes.length ?? 0)
      setResult(merged)
      setNotice(
        added > 0
          ? `${formatCount(added)} nœud${added > 1 ? 's' : ''} ajouté${added > 1 ? 's' : ''} autour de « ${name} ».`
          : `Aucun nouveau voisin autour de « ${name} ».`,
      )
    } catch (err) {
      if (alive.current) {
        setNotice(null)
        setError(describeError(err))
      }
    } finally {
      expanding.current = false
    }
  }, [])

  const insert = useCallback((text: string) => {
    setQuery(text)
    setError(null)
    editorRef.current?.focus()
  }, [])

  const workspace = schema?.workspace ?? null

  const drawnNodes = useMemo(
    () => (result?.nodes ?? []).slice(0, MAX_DRAWN_NODES),
    [result],
  )

  const placed = useMemo(() => place(drawnNodes, workspace), [drawnNodes, workspace])

  const positions = useMemo(() => {
    const map = new Map<string, Placed>()
    for (const item of placed) map.set(item.node.id, item)
    return map
  }, [placed])

  /** Endpoints are trimmed back to the node rim so the arrowhead stays visible. */
  const drawnEdges = useMemo(() => {
    const drawn: DrawnEdge[] = []
    for (const edge of result?.edges ?? []) {
      const from = positions.get(edge.source)
      const to = positions.get(edge.target)
      if (!from || !to) continue
      if (from.node.id === to.node.id) {
        drawn.push({ edge, x1: from.x, y1: from.y, x2: from.x, y2: from.y, loop: true })
        continue
      }
      const dx = to.x - from.x
      const dy = to.y - from.y
      const length = Math.hypot(dx, dy) || 1
      const trim = NODE_RADIUS + 4
      drawn.push({
        edge,
        x1: from.x + (dx / length) * NODE_RADIUS,
        y1: from.y + (dy / length) * NODE_RADIUS,
        x2: to.x - (dx / length) * trim,
        y2: to.y - (dy / length) * trim,
        loop: false,
      })
    }
    return drawn
  }, [result, positions])

  const groups = useMemo(
    () => countBy(placed, (item) => item.group),
    [placed],
  )

  const resultLabels = useMemo(
    () => countBy(result?.nodes ?? [], (node) => groupOf(node, workspace)),
    [result, workspace],
  )

  const resultTypes = useMemo(
    () => countBy(result?.edges ?? [], (edge) => edge.type),
    [result],
  )

  const stats = result?.stats ?? null
  const truncatedRender = (result?.nodes.length ?? 0) > MAX_DRAWN_NODES
  const showEdgeCaptions = drawnEdges.length <= MAX_EDGE_CAPTIONS
  const showNodeCaptions = placed.length <= MAX_NODE_CAPTIONS

  return (
    <div className="flex min-h-0 flex-1">
      {/* ------------------------------------------------- base inventory --- */}
      <aside className="hidden w-64 shrink-0 flex-col overflow-y-auto border-r border-edge bg-surface-sunken lg:flex">
        <div className="flex items-center gap-2 px-4 py-3">
          <h3 className="flex-1 text-xs font-semibold uppercase tracking-wide text-ink-faint">
            Informations base
          </h3>
          <button
            type="button"
            className="icon-btn"
            onClick={() => void loadSchema()}
            disabled={schemaLoading}
            title="Recharger le schéma"
          >
            <RefreshIcon size={14} />
          </button>
        </div>

        {schemaError ? (
          <p className="mx-4 mb-3 rounded-card border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700">
            {schemaError}
          </p>
        ) : null}

        <div className="grid grid-cols-2 gap-2 px-4 pb-3">
          <div className="rounded-card border border-edge bg-white px-3 py-2">
            <p className="text-lg font-semibold leading-tight text-ink">
              {schema ? formatCount(schema.node_count) : '—'}
            </p>
            <p className="text-[11px] text-ink-muted">Nœuds</p>
          </div>
          <div className="rounded-card border border-edge bg-white px-3 py-2">
            <p className="text-lg font-semibold leading-tight text-ink">
              {schema ? formatCount(schema.relationship_count) : '—'}
            </p>
            <p className="text-[11px] text-ink-muted">Relations</p>
          </div>
        </div>

        {schema ? (
          <p className="px-4 pb-3 text-[11px] text-ink-faint">
            Espace de travail <span className="font-mono text-ink-muted">{schema.workspace}</span>
          </p>
        ) : null}

        <Section title="Labels">
          {schema && schema.labels.length > 0 ? (
            schema.labels.map((item) => (
              <CountPill
                key={item.label}
                name={item.label}
                count={item.count}
                color={colorForGroup(item.label)}
                title={`Insérer une requête sur ${item.label}`}
                onClick={() =>
                  insert(`MATCH (n:\`${item.label}\`)\nRETURN n\nLIMIT 25`)
                }
              />
            ))
          ) : (
            <p className="text-[11px] text-ink-faint">{schemaLoading ? 'Chargement…' : 'Aucun'}</p>
          )}
        </Section>

        <Section title="Types de relations">
          {schema && schema.relationship_types.length > 0 ? (
            schema.relationship_types.map((item) => (
              <CountPill
                key={item.type}
                name={item.type}
                count={item.count}
                title={`Insérer une requête sur ${item.type}`}
                onClick={() =>
                  insert(`MATCH p=()-[r:\`${item.type}\`]->()\nRETURN p\nLIMIT 25`)
                }
              />
            ))
          ) : (
            <p className="text-[11px] text-ink-faint">{schemaLoading ? 'Chargement…' : 'Aucun'}</p>
          )}
        </Section>

        <Section title="Clés de propriétés">
          {schema && schema.property_keys.length > 0 ? (
            schema.property_keys.map((key) => (
              <CountPill
                key={key}
                name={key}
                title={`Insérer une requête sur la propriété ${key}`}
                onClick={() =>
                  insert(
                    `MATCH (n)\nWHERE n.\`${key}\` IS NOT NULL\n` +
                      `RETURN n.\`${key}\` AS \`${key}\`, count(*) AS total\n` +
                      'ORDER BY total DESC\nLIMIT 25',
                  )
                }
              />
            ))
          ) : (
            <p className="text-[11px] text-ink-faint">{schemaLoading ? 'Chargement…' : 'Aucune'}</p>
          )}
        </Section>

        <p className="mt-auto border-t border-edge px-4 py-3 text-[11px] leading-relaxed text-ink-faint">
          Console en lecture seule : les écritures (CREATE, MERGE, DELETE…) sont refusées
          par le serveur.
        </p>
      </aside>

      {/* --------------------------------------------------- editor + result --- */}
      <div className="flex min-w-0 flex-1 flex-col">
        <section className="border-b border-edge px-5 py-4">
          <div className="flex items-start gap-2 rounded-panel border border-edge bg-white px-3 py-2 focus-within:border-violet-300">
            <span className="select-none pt-2 font-mono text-xs text-violet-600">neo4j$</span>
            <textarea
              ref={editorRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                  event.preventDefault()
                  void execute()
                }
              }}
              rows={4}
              spellCheck={false}
              aria-label="Requête Cypher"
              placeholder="MATCH (n) RETURN n LIMIT 25"
              className="min-h-[76px] flex-1 resize-y bg-transparent py-1.5 font-mono text-[13px]
                leading-relaxed text-ink outline-none placeholder:text-ink-faint"
            />
            <div className="flex flex-col items-end gap-2 pt-1">
              <button
                type="button"
                className="btn-dark"
                disabled={loading}
                onClick={() => void execute()}
                title="Exécuter (Ctrl/Cmd + Entrée)"
              >
                <SendIcon size={15} className="rotate-90" />
                {loading ? 'Exécution…' : 'Exécuter'}
              </button>
              <label className="flex items-center gap-1.5 text-[11px] text-ink-muted">
                Limite
                <input
                  type="number"
                  min={1}
                  max={MAX_LIMIT}
                  step={25}
                  value={limit}
                  onChange={(event) => {
                    const parsed = Number(event.target.value)
                    setLimit(
                      Number.isFinite(parsed)
                        ? Math.min(MAX_LIMIT, Math.max(1, Math.round(parsed)))
                        : DEFAULT_LIMIT,
                    )
                  }}
                  className="w-20 rounded-full border border-edge-strong bg-white px-2 py-1
                    text-right text-xs text-ink outline-none focus:border-violet-300"
                />
              </label>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => insert(preset.query)}
                title={preset.query}
                className="rounded-full border border-edge bg-white px-3 py-1 text-[11px] text-ink-muted
                  transition hover:border-violet-300 hover:bg-violet-50 hover:text-ink"
              >
                {preset.label}
              </button>
            ))}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-muted">
            {stats ? (
              <span>
                {formatCount(stats.records)} enregistrement{stats.records > 1 ? 's' : ''} en{' '}
                {numberFormat.format(stats.elapsed_ms)} ms
              </span>
            ) : (
              <span className="text-ink-faint">
                Ctrl/Cmd + Entrée pour exécuter · double-clic sur un nœud pour l&apos;étendre
              </span>
            )}
            {stats?.truncated ? (
              <span className="rounded-full bg-amber-50 px-2 py-0.5 text-amber-700">
                Résultat tronqué à {formatCount(stats.limit_applied ?? limit)} enregistrements
              </span>
            ) : null}
            {stats && !stats.truncated && stats.limit_applied !== null ? (
              <span className="text-ink-faint">limite {formatCount(stats.limit_applied)}</span>
            ) : null}
            {notice ? <span className="text-violet-600">{notice}</span> : null}
          </div>

          {error ? (
            <div className="mt-3 rounded-card border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <p className="font-mono text-[12px] leading-relaxed">{error}</p>
              {isReadOnlyRejection(error) ? (
                <p className="mt-1.5 text-[12px]">
                  La console est en lecture seule : seules les lectures (MATCH, OPTIONAL MATCH,
                  WITH, RETURN…) sont autorisées. La requête a été refusée avant d&apos;atteindre
                  la base.
                </p>
              ) : null}
            </div>
          ) : null}
        </section>

        <div className="flex min-h-0 flex-1">
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex items-center gap-1 border-b border-edge px-5 py-2">
              {(
                [
                  { id: 'graph' as const, label: 'Graphe', icon: <GraphIcon size={14} /> },
                  { id: 'table' as const, label: 'Table', icon: <LayersIcon size={14} /> },
                  { id: 'raw' as const, label: 'Brut', icon: <FileIcon size={14} /> },
                ] satisfies { id: Tab; label: string; icon: ReactNode }[]
              ).map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setTab(item.id)}
                  className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs transition ${
                    tab === item.id
                      ? 'bg-ink text-white'
                      : 'text-ink-muted hover:bg-surface-sunken hover:text-ink'
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              ))}
              {result ? (
                <span className="ml-auto text-[11px] text-ink-faint">
                  {formatCount(result.nodes.length)} nœuds · {formatCount(result.edges.length)}{' '}
                  relations
                </span>
              ) : null}
            </div>

            <div className="relative min-h-0 flex-1 overflow-hidden p-5">
              {!result ? (
                <p className="flex h-full items-center justify-center text-sm text-ink-faint">
                  Exécutez une requête pour voir son résultat.
                </p>
              ) : tab === 'graph' ? (
                <div className="flex h-full flex-col">
                  {truncatedRender ? (
                    <p className="mb-2 rounded-card border border-amber-200 bg-amber-50 px-3 py-1.5 text-[11px] text-amber-700">
                      Affichage limité aux {MAX_DRAWN_NODES} premiers nœuds sur{' '}
                      {formatCount(result.nodes.length)}. Affinez la requête pour tout voir.
                    </p>
                  ) : null}

                  <div className="relative min-h-0 flex-1 rounded-panel border border-edge bg-white">
                    {placed.length === 0 ? (
                      <p className="flex h-full items-center justify-center px-6 text-center text-sm text-ink-faint">
                        Cette requête ne renvoie aucun nœud — regardez l&apos;onglet Table.
                      </p>
                    ) : (
                      <svg
                        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
                        preserveAspectRatio="xMidYMid meet"
                        className="h-full w-full"
                        aria-label="Résultat sous forme de graphe"
                      >
                        <defs>
                          <marker
                            id="cypher-arrow"
                            viewBox="0 0 10 10"
                            refX="9"
                            refY="5"
                            markerWidth="5"
                            markerHeight="5"
                            orient="auto"
                          >
                            <path d="M0 0 L10 5 L0 10 z" fill="#D4D4D4" />
                          </marker>
                        </defs>

                        <g stroke="#DDDDDD" strokeWidth={1} fill="none">
                          {drawnEdges.map((item) =>
                            item.loop ? (
                              <path
                                key={item.edge.id}
                                d={`M${item.x1 - 6},${item.y1 - 4} a 9,9 0 1,1 12,0`}
                              />
                            ) : (
                              <line
                                key={item.edge.id}
                                x1={item.x1}
                                y1={item.y1}
                                x2={item.x2}
                                y2={item.y2}
                                markerEnd="url(#cypher-arrow)"
                              />
                            ),
                          )}
                        </g>

                        {showEdgeCaptions ? (
                          <g
                            fontSize={8}
                            textAnchor="middle"
                            fill="#9A9A9A"
                            stroke="#FFFFFF"
                            strokeWidth={2.5}
                            paintOrder="stroke"
                            className="pointer-events-none"
                          >
                            {drawnEdges.map((item) => (
                              <text
                                key={`${item.edge.id}-caption`}
                                x={(item.x1 + item.x2) / 2}
                                y={(item.y1 + item.y2) / 2 - 3}
                              >
                                {item.edge.type}
                              </text>
                            ))}
                          </g>
                        ) : null}

                        <g>
                          {placed.map((item) => {
                            const active = selected?.id === item.node.id
                            return (
                              <g key={item.node.id}>
                                <circle
                                  cx={item.x}
                                  cy={item.y}
                                  r={active ? NODE_RADIUS + 3.5 : NODE_RADIUS}
                                  fill={colorForGroup(item.group)}
                                  stroke={active ? '#111111' : '#FFFFFF'}
                                  strokeWidth={active ? 2.5 : 1.5}
                                  className="cursor-pointer"
                                  onClick={() => setSelected(item.node)}
                                  onDoubleClick={() => void expand(item.node)}
                                >
                                  <title>
                                    {`${item.group} · ${clip(captionOf(item.node), 120)}`}
                                  </title>
                                </circle>
                                {showNodeCaptions ? (
                                  <text
                                    x={item.x}
                                    y={item.y + 20}
                                    fontSize={9}
                                    textAnchor="middle"
                                    fill="#6B6B6B"
                                    stroke="#FFFFFF"
                                    strokeWidth={2.5}
                                    paintOrder="stroke"
                                    className="pointer-events-none"
                                  >
                                    {clip(captionOf(item.node), 22)}
                                  </text>
                                ) : null}
                              </g>
                            )
                          })}
                        </g>
                      </svg>
                    )}

                    {selected ? (
                      <div className="absolute right-3 top-3 flex max-h-[80%] w-72 flex-col rounded-card border border-edge bg-white/95 shadow-[0_12px_32px_-18px_rgba(0,0,0,0.35)] backdrop-blur-sm">
                        <div className="flex items-center gap-2 border-b border-edge px-3 py-2">
                          <span
                            className="h-2.5 w-2.5 shrink-0 rounded-full"
                            style={{ backgroundColor: colorForGroup(groupOf(selected, workspace)) }}
                          />
                          <p className="min-w-0 flex-1 truncate text-xs font-medium text-ink">
                            {groupOf(selected, workspace)}
                          </p>
                          <button
                            type="button"
                            className="icon-btn h-6 w-6"
                            onClick={() => setSelected(null)}
                            title="Fermer le détail"
                          >
                            <CloseIcon size={13} />
                          </button>
                        </div>
                        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
                          <p className="break-all font-mono text-[11px] text-ink-muted">
                            {selected.id}
                          </p>
                          <p className="mt-1 text-[11px] text-ink-faint">
                            {selected.labels.join(' · ')}
                          </p>
                          <dl className="mt-3 space-y-2">
                            {Object.entries(selected.properties).map(([key, value]) => (
                              <div
                                key={key}
                                className="rounded-lg border border-edge bg-surface-sunken px-2.5 py-1.5"
                              >
                                <dt className="text-[10px] uppercase tracking-wide text-ink-faint">
                                  {key}
                                </dt>
                                <dd className="mt-0.5 break-words text-[11px] text-ink">
                                  {cellText(value)}
                                </dd>
                              </div>
                            ))}
                          </dl>
                        </div>
                        <button
                          type="button"
                          className="btn-ghost m-3 justify-center text-xs"
                          onClick={() => void expand(selected)}
                        >
                          <GraphIcon size={13} />
                          Étendre le voisinage
                        </button>
                      </div>
                    ) : null}
                  </div>

                  <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
                    {groups.map(([group, count]) => (
                      <span
                        key={group}
                        className="inline-flex items-center gap-1.5 text-[11px] text-ink-muted"
                      >
                        <span
                          className="h-2 w-2 rounded-full"
                          style={{ backgroundColor: colorForGroup(group) }}
                        />
                        {group} ({formatCount(count)})
                      </span>
                    ))}
                  </div>
                </div>
              ) : tab === 'table' ? (
                <div className="h-full overflow-auto rounded-panel border border-edge bg-white">
                  {result.rows.length === 0 ? (
                    <p className="flex h-full items-center justify-center text-sm text-ink-faint">
                      Aucun enregistrement.
                    </p>
                  ) : (
                    <table className="w-full min-w-max border-collapse text-left text-xs">
                      <thead className="sticky top-0 bg-surface-sunken">
                        <tr>
                          {result.columns.map((column) => (
                            <th
                              key={column}
                              className="whitespace-nowrap border-b border-edge px-3 py-2 font-mono
                                text-[11px] font-medium text-ink-muted"
                            >
                              {column}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.rows.map((row, index) => (
                          // Rows have no stable id — the server returns plain records.
                          <tr key={index} className="odd:bg-white even:bg-surface-sunken/50">
                            {result.columns.map((column) => {
                              const full = cellText(row[column])
                              return (
                                <td
                                  key={column}
                                  title={full}
                                  className="max-w-[380px] truncate border-b border-edge px-3 py-1.5
                                    font-mono text-[11px] text-ink"
                                >
                                  {clip(full, 160)}
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              ) : (
                <pre className="h-full overflow-auto rounded-panel border border-edge bg-white p-4
                  font-mono text-[11px] leading-relaxed text-ink">
                  {JSON.stringify(result, null, 2)}
                </pre>
              )}
            </div>
          </div>

          {/* ---------------------------------------------- result summary --- */}
          <aside className="hidden w-56 shrink-0 flex-col overflow-y-auto border-l border-edge bg-surface-sunken xl:flex">
            <h3 className="px-4 py-3 text-xs font-semibold uppercase tracking-wide text-ink-faint">
              Résultats
            </h3>
            {!result ? (
              <p className="px-4 text-[11px] text-ink-faint">
                Le résumé du résultat s&apos;affiche ici après exécution.
              </p>
            ) : (
              <>
                <Section title="Nœuds">
                  {resultLabels.length > 0 ? (
                    resultLabels.map(([label, count]) => (
                      <CountPill
                        key={label}
                        name={label}
                        count={count}
                        color={colorForGroup(label)}
                      />
                    ))
                  ) : (
                    <p className="text-[11px] text-ink-faint">Aucun</p>
                  )}
                </Section>
                <Section title="Relations">
                  {resultTypes.length > 0 ? (
                    resultTypes.map(([type, count]) => (
                      <CountPill key={type} name={type} count={count} />
                    ))
                  ) : (
                    <p className="text-[11px] text-ink-faint">Aucune</p>
                  )}
                </Section>
              </>
            )}
          </aside>
        </div>
      </div>
    </div>
  )
}
