import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { CypherEdge, CypherNode } from '../api/types'

export type Placed = { node: CypherNode; x: number; y: number; group: string }

export type DrawnEdge = {
  edge: CypherEdge
  x1: number
  y1: number
  x2: number
  y2: number
  loop: boolean
}

type Viewport = { x: number; y: number; w: number; h: number }

const MIN_SPAN_RATIO = 0.08 // ~12× zoom in
const MAX_SPAN_RATIO = 3 // 3× zoom out
const HIT_STROKE = 12 // invisible companion line width, in viewBox units

interface Props {
  placed: Placed[]
  edges: DrawnEdge[]
  width: number
  height: number
  radius: number
  selected: CypherNode | null
  showNodeCaptions: boolean
  showEdgeCaptions: boolean
  colorForGroup: (group: string) => string
  captionOf: (node: CypherNode) => string
  clip: (text: string, max: number) => string
  onSelect: (node: CypherNode) => void
  onExpand: (node: CypherNode) => void
}

/**
 * The graph canvas: zoom, pan, and hover on nodes *and* edges.
 *
 * Extracted from `CypherConsole` rather than added to it, for one measurable
 * reason: hover state changes on every mouse move, and holding it in a 986-line
 * component re-rendered the sidebar, the legend and the result table along with
 * the circle under the cursor. Memoised here, a hover repaints the canvas only.
 *
 * Zoom is a state-driven `viewBox` rather than a `<g transform>`. Both work; the
 * viewBox keeps stroke widths and font sizes constant on screen, so a zoomed-in
 * graph does not turn into fat lines and giant labels.
 */
function GraphCanvas({
  placed,
  edges,
  width,
  height,
  radius,
  selected,
  showNodeCaptions,
  showEdgeCaptions,
  colorForGroup,
  captionOf,
  clip,
  onSelect,
  onExpand,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [view, setView] = useState<Viewport>({ x: 0, y: 0, w: width, h: height })
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null)
  const dragRef = useRef<{ x: number; y: number; view: Viewport } | null>(null)

  // A new result means new coordinates; keeping the old window would open on empty
  // space with no indication that anything was drawn.
  useEffect(() => {
    setView({ x: 0, y: 0, w: width, h: height })
  }, [width, height, placed])

  const clampSpan = useCallback(
    (span: number) => Math.min(width * MAX_SPAN_RATIO, Math.max(width * MIN_SPAN_RATIO, span)),
    [width],
  )

  const zoomAt = useCallback(
    (factor: number, originX: number, originY: number) => {
      setView((current) => {
        const w = clampSpan(current.w * factor)
        const scale = w / current.w
        const h = current.h * scale
        return {
          w,
          h,
          // Keep the point under the cursor fixed, which is what makes wheel zoom
          // feel like zooming rather than like scrolling a picture.
          x: originX - (originX - current.x) * scale,
          y: originY - (originY - current.y) * scale,
        }
      })
    },
    [clampSpan],
  )

  const toViewBox = useCallback(
    (clientX: number, clientY: number) => {
      const svg = svgRef.current
      if (!svg) return { x: view.x + view.w / 2, y: view.y + view.h / 2 }
      const rect = svg.getBoundingClientRect()
      const ratio = Math.min(rect.width / view.w, rect.height / view.h) || 1
      // preserveAspectRatio="xMidYMid meet" letterboxes; the offsets undo that.
      const offsetX = (rect.width - view.w * ratio) / 2
      const offsetY = (rect.height - view.h * ratio) / 2
      return {
        x: view.x + (clientX - rect.left - offsetX) / ratio,
        y: view.y + (clientY - rect.top - offsetY) / ratio,
      }
    },
    [view],
  )

  // React's onWheel is registered as a passive listener, where preventDefault does
  // nothing at all. Without this the page scrolls under the cursor while the graph
  // zooms, which is worse than either behaviour alone.
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const origin = toViewBox(event.clientX, event.clientY)
      zoomAt(event.deltaY > 0 ? 1.12 : 1 / 1.12, origin.x, origin.y)
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [toViewBox, zoomAt])

  const onPointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return
    dragRef.current = { x: event.clientX, y: event.clientY, view }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current
    if (!drag) return
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    const ratio = Math.min(rect.width / drag.view.w, rect.height / drag.view.h) || 1
    setView({
      ...drag.view,
      x: drag.view.x - (event.clientX - drag.x) / ratio,
      y: drag.view.y - (event.clientY - drag.y) / ratio,
    })
  }

  const endDrag = (event: React.PointerEvent<SVGSVGElement>) => {
    if (dragRef.current) {
      dragRef.current = null
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  const fit = useCallback(() => {
    if (placed.length === 0) {
      setView({ x: 0, y: 0, w: width, h: height })
      return
    }
    const xs = placed.map((p) => p.x)
    const ys = placed.map((p) => p.y)
    const pad = radius * 6
    const minX = Math.min(...xs) - pad
    const minY = Math.min(...ys) - pad
    const spanX = Math.max(...xs) + pad - minX
    const spanY = Math.max(...ys) + pad - minY
    // Match the canvas aspect ratio, or "fit" would crop on one axis.
    const aspect = width / height
    const w = Math.max(spanX, spanY * aspect)
    setView({ x: minX - (w - spanX) / 2, y: minY - (w / aspect - spanY) / 2, w, h: w / aspect })
  }, [placed, width, height, radius])

  const centre = useMemo(() => ({ x: view.x + view.w / 2, y: view.y + view.h / 2 }), [view])
  const hovered = useMemo(
    () => placed.find((item) => item.node.id === hoveredNode) ?? null,
    [placed, hoveredNode],
  )
  const hoveredEdgeItem = useMemo(
    () => edges.find((item) => item.edge.id === hoveredEdge) ?? null,
    [edges, hoveredEdge],
  )

  return (
    <>
      <svg
        ref={svgRef}
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        preserveAspectRatio="xMidYMid meet"
        className="h-full w-full touch-none select-none"
        style={{ cursor: dragRef.current ? 'grabbing' : 'grab' }}
        aria-label="Résultat sous forme de graphe"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
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
          {/* A second marker: markerEnd points at one shared id, so a hovered edge
              cannot recolour its own arrowhead without one. */}
          <marker
            id="cypher-arrow-hot"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path d="M0 0 L10 5 L0 10 z" fill="#7C3AED" />
          </marker>
        </defs>

        <g fill="none">
          {edges.map((item) => {
            const hot = item.edge.id === hoveredEdge
            const stroke = hot ? '#7C3AED' : '#DDDDDD'
            return (
              <g key={item.edge.id}>
                {/* The visible line is 1px wide in viewBox units and sits in a group
                    with no pointer-events, so it is literally unreachable by a
                    cursor. This transparent companion is the hit target. */}
                {item.loop ? (
                  <>
                    <path
                      d={`M${item.x1 - 6},${item.y1 - 4} a 9,9 0 1,1 12,0`}
                      stroke={stroke}
                      strokeWidth={hot ? 2 : 1}
                    />
                    <path
                      d={`M${item.x1 - 6},${item.y1 - 4} a 9,9 0 1,1 12,0`}
                      stroke="transparent"
                      strokeWidth={HIT_STROKE}
                      pointerEvents="stroke"
                      onPointerEnter={() => setHoveredEdge(item.edge.id)}
                      onPointerLeave={() => setHoveredEdge(null)}
                    />
                  </>
                ) : (
                  <>
                    <line
                      x1={item.x1}
                      y1={item.y1}
                      x2={item.x2}
                      y2={item.y2}
                      stroke={stroke}
                      strokeWidth={hot ? 2 : 1}
                      markerEnd={hot ? 'url(#cypher-arrow-hot)' : 'url(#cypher-arrow)'}
                      className="transition-[stroke-width] duration-150 motion-reduce:transition-none"
                    />
                    <line
                      x1={item.x1}
                      y1={item.y1}
                      x2={item.x2}
                      y2={item.y2}
                      stroke="transparent"
                      strokeWidth={HIT_STROKE}
                      pointerEvents="stroke"
                      onPointerEnter={() => setHoveredEdge(item.edge.id)}
                      onPointerLeave={() => setHoveredEdge(null)}
                    />
                  </>
                )}
              </g>
            )
          })}
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
            {edges.map((item) => (
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
            const hot = hoveredNode === item.node.id
            return (
              <g key={item.node.id}>
                <circle
                  cx={item.x}
                  cy={item.y}
                  r={active ? radius + 3.5 : hot ? radius + 2 : radius}
                  fill={colorForGroup(item.group)}
                  stroke={active ? '#111111' : hot ? '#7C3AED' : '#FFFFFF'}
                  strokeWidth={active ? 2.5 : hot ? 2 : 1.5}
                  // `r` and `stroke-width` are CSS-animatable on SVG in every
                  // browser this ships to; `motion-reduce` is honoured because
                  // hover animation is exactly what that preference is about.
                  className="cursor-pointer transition-all duration-150 motion-reduce:transition-none"
                  onClick={() => onSelect(item.node)}
                  onDoubleClick={() => void onExpand(item.node)}
                  onPointerEnter={() => setHoveredNode(item.node.id)}
                  onPointerLeave={() => setHoveredNode(null)}
                />
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

      {/* Hover cards. Bottom-left, because the selection card owns right-3 top-3. */}
      {hovered ? (
        <div
          className="pointer-events-none absolute bottom-14 left-3 max-w-[320px] rounded-card
            border border-edge bg-white/95 px-3 py-2 shadow-[0_12px_32px_-18px_rgba(0,0,0,0.35)]
            backdrop-blur-sm"
        >
          <p className="mb-0.5 flex items-center gap-1.5 text-[11px] font-medium text-ink">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: colorForGroup(hovered.group) }}
            />
            {hovered.group}
          </p>
          <p className="text-[11.5px] leading-relaxed text-ink-muted">
            {clip(captionOf(hovered.node), 160)}
          </p>
        </div>
      ) : hoveredEdgeItem ? (
        <div
          className="pointer-events-none absolute bottom-14 left-3 max-w-[320px] rounded-card
            border border-edge bg-white/95 px-3 py-2 shadow-[0_12px_32px_-18px_rgba(0,0,0,0.35)]
            backdrop-blur-sm"
        >
          <p className="mb-0.5 font-mono text-[11px] font-medium text-violet-700">
            {hoveredEdgeItem.edge.type}
          </p>
          {/* Edge properties are carried by the API and rendered nowhere else. */}
          {Object.keys(hoveredEdgeItem.edge.properties ?? {}).length > 0 ? (
            <dl className="text-[11px] leading-relaxed text-ink-muted">
              {Object.entries(hoveredEdgeItem.edge.properties).map(([key, value]) => (
                <div key={key} className="flex gap-1.5">
                  <dt className="shrink-0 text-ink-faint">{key}</dt>
                  <dd className="truncate">{String(value)}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="text-[11px] text-ink-faint">Aucune propriété</p>
          )}
        </div>
      ) : null}

      {/* Controls, bottom-right like the Neo4j Browser reference. No Escape
          shortcut: GraphPanel binds a global keydown that closes the panel, and a
          second meaning for the same key would be a coin toss. */}
      <div
        className="absolute bottom-3 right-3 flex flex-col overflow-hidden rounded-card border
          border-edge bg-white/95 shadow-[0_12px_32px_-18px_rgba(0,0,0,0.35)] backdrop-blur-sm"
      >
        <ControlButton label="Zoom avant" onClick={() => zoomAt(1 / 1.3, centre.x, centre.y)}>
          +
        </ControlButton>
        <ControlButton label="Zoom arrière" onClick={() => zoomAt(1.3, centre.x, centre.y)}>
          −
        </ControlButton>
        <ControlButton label="Ajuster à l'écran" onClick={fit}>
          ⤢
        </ControlButton>
        <ControlButton
          label="Réinitialiser la vue"
          onClick={() => setView({ x: 0, y: 0, w: width, h: height })}
        >
          ⌖
        </ControlButton>
      </div>
    </>
  )
}

function ControlButton({
  label,
  onClick,
  children,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className="h-8 w-8 text-[13px] leading-none text-ink-muted transition-colors
        hover:bg-violet-50 hover:text-violet-700 focus:outline-none focus-visible:ring-2
        focus-visible:ring-inset focus-visible:ring-violet-400 motion-reduce:transition-none
        [&+button]:border-t [&+button]:border-edge"
    >
      {children}
    </button>
  )
}

export default memo(GraphCanvas)
