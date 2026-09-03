import { useEffect, useRef } from 'react'

import type { ChatMessage, Reference, ToolCall } from '../api/types'
import { FileIcon } from './icons'

function basename(path: string): string {
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] || path
}

/** Group references by document, keeping every citation number.
 *
 *  There is one reference per retrieved passage, so ten passages from three files
 *  would otherwise render ten near-identical pills. Grouping shows three, each
 *  listing the numbers the answer used — the information is the same, the noise
 *  is not.
 */
function groupByDocument(references: Reference[]) {
  const groups = new Map<string, { label: string; refs: Reference[] }>()
  for (const ref of references) {
    const key = ref.source_path || ref.file_path
    const existing = groups.get(key)
    if (existing) existing.refs.push(ref)
    else groups.set(key, { label: basename(ref.file_path), refs: [ref] })
  }
  return [...groups.entries()].map(([key, value]) => ({ key, ...value }))
}

function Citations({
  references,
  onOpen,
}: {
  references: Reference[]
  onOpen?: (ref: Reference) => void
}) {
  if (references.length === 0) return null
  const groups = groupByDocument(references)
  return (
    <div className="mt-3 border-t border-edge pt-2.5">
      <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
        Sources
      </p>
      <ul className="flex flex-wrap gap-1.5">
        {groups.map((group) => (
          <li key={group.key}>
            {/* A button, not a span: the pill is now an action, and a button keeps
                keyboard focus and Enter/Space for free. */}
            <button
              type="button"
              onClick={() => onOpen?.(group.refs[0])}
              title={`${group.key} — ouvrir dans la bibliothèque`}
              className="inline-flex max-w-[280px] items-center gap-1.5 rounded-full border
                border-edge bg-surface-sunken px-2.5 py-1 text-[12px] text-ink-muted
                transition-colors hover:border-violet-400 hover:bg-violet-50 hover:text-ink
                focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-400
                motion-reduce:transition-none"
            >
              <FileIcon size={13} className="shrink-0 text-violet-600" />
              <span className="shrink-0 font-medium text-ink">
                {group.refs.map((r) => `[${r.reference_id}]`).join('')}
              </span>
              <span className="truncate">{group.label}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** Agent-mode step list, shown while the loop runs.
 *
 *  A forty-second turn with no visible progress is indistinguishable from a hang,
 *  and the loop is the one mode where that duration is normal.
 */
function AgentSteps({ steps }: { steps: ToolCall[] }) {
  if (steps.length === 0) return null
  return (
    <ul className="mb-2 flex flex-col gap-1 text-[12px] text-ink-muted">
      {steps.map((step, index) => (
        <li key={`${step.step}-${index}`} className="flex items-start gap-1.5">
          <span className="mt-[3px] h-1.5 w-1.5 shrink-0 rounded-full bg-violet-500" />
          <span className="truncate">
            <span className="font-medium text-ink">{step.name}</span>{' '}
            {readableArguments(step.arguments)}
          </span>
        </li>
      ))}
    </ul>
  )
}

/** Show the reformulated search, not the raw JSON the model emitted. */
function readableArguments(raw: string): string {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>
    const query = parsed.query ?? parsed.term
    if (typeof query === 'string') return `« ${query} »`
  } catch {
    /* a partial or malformed argument string is shown as-is */
  }
  return raw.slice(0, 120)
}

/** Minimal block renderer: paragraphs, bullet lists and fenced code.
 *  The answers are prose with `[n]` citations, not rich documents — a full Markdown
 *  dependency would cost more than it returns here. */
function Answer({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/)
  return (
    <div className="flex flex-col gap-2.5">
      {blocks.map((block, index) => {
        const lines = block.split('\n')
        if (block.startsWith('```')) {
          return (
            <pre
              key={index}
              className="overflow-x-auto rounded-lg bg-surface-sunken p-3 text-[12.5px] leading-relaxed"
            >
              <code>{lines.filter((l) => !l.startsWith('```')).join('\n')}</code>
            </pre>
          )
        }
        if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) {
          return (
            <ul key={index} className="ml-4 list-disc space-y-1">
              {lines.map((line, i) => (
                <li key={i}>{line.replace(/^\s*[-*•]\s+/, '')}</li>
              ))}
            </ul>
          )
        }
        return (
          <p key={index} className="whitespace-pre-wrap">
            {block}
          </p>
        )
      })}
    </div>
  )
}

interface Props {
  messages: ChatMessage[]
  streaming: boolean
  pendingAnswer: string
  pendingRefs: Reference[]
  pendingSteps?: ToolCall[]
  onCitationClick?: (ref: Reference) => void
}

export default function MessageList({
  messages,
  streaming,
  pendingAnswer,
  pendingRefs,
  pendingSteps = [],
  onCitationClick,
}: Props) {
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length, pendingAnswer, streaming])

  return (
    <div className="flex flex-col gap-5">
      {messages.map((message) =>
        message.role === 'user' ? (
          <div key={message.id} className="flex justify-end">
            <div
              className="max-w-[78%] rounded-card rounded-br-md bg-ink px-4 py-2.5
                text-[14.5px] leading-relaxed text-white"
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
            </div>
          </div>
        ) : (
          <div key={message.id} className="flex justify-start">
            <div
              className="max-w-[86%] rounded-card rounded-bl-md border border-edge bg-white
                px-4 py-3 text-[14.5px] leading-relaxed"
            >
              <Answer text={message.content} />
              <Citations references={message.references} onOpen={onCitationClick} />
            </div>
          </div>
        ),
      )}

      {streaming && (
        <div className="flex justify-start">
          <div
            className="max-w-[86%] rounded-card rounded-bl-md border border-edge bg-white
              px-4 py-3 text-[14.5px] leading-relaxed"
          >
            <AgentSteps steps={pendingSteps} />
            {pendingAnswer ? (
              <Answer text={pendingAnswer} />
            ) : (
              // Retrieval runs before the first token, so this covers a real wait
              // rather than being decorative.
              <span className="dot-pulse inline-flex items-center gap-1 text-ink-faint">
                <span className="h-1.5 w-1.5 rounded-full bg-violet-600" />
                <span className="h-1.5 w-1.5 rounded-full bg-violet-600" />
                <span className="h-1.5 w-1.5 rounded-full bg-violet-600" />
                <span className="ml-2 text-[12.5px]">Récupération en cours…</span>
              </span>
            )}
            <Citations references={pendingRefs} onOpen={onCitationClick} />
          </div>
        </div>
      )}
      <div ref={bottom} />
    </div>
  )
}
