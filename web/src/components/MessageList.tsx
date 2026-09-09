import { useCallback, useEffect, useRef, type RefObject } from 'react'

import type { ChatMessage, Reference, ToolCall } from '../api/types'
import { answerOnly, splitThought } from '../lib/split-thought'
import { FileIcon } from './icons'
import MessageActions from './MessageActions'
import ThinkingBlock from './ThinkingBlock'

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

/** Agent-mode step list, shown while the loop runs. A forty-second turn with no
 *  visible progress is indistinguishable from a hang. */
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

/** Inline markdown: **bold**, *italic*, `code`. The answers are prose with `[n]`
 *  citations, so this is the whole inline vocabulary they use; a full markdown
 *  engine would bring HTML rendering of model output along with it. */
function Inline({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|(?<![\w*])\*[^*\n]+\*(?![\w*]))/g)
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
          return <strong key={i}>{part.slice(2, -2)}</strong>
        }
        if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
          return (
            <code key={i} className="rounded bg-surface-sunken px-1 py-0.5 font-mono text-[12.5px]">
              {part.slice(1, -1)}
            </code>
          )
        }
        if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
          return <em key={i}>{part.slice(1, -1)}</em>
        }
        return part
      })}
    </>
  )
}

const BULLET = /^\s*[-*•]\s+/
const NUMBERED = /^\s*\d+[.)]\s+/
const HEADING = /^(#{1,4})\s+(.+)$/

/** Block renderer: paragraphs, headings, bullet and numbered lists, fenced code. */
function Answer({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/)
  return (
    <div className="flex flex-col gap-2.5">
      {blocks.map((block, index) => {
        const lines = block.split('\n').filter((l) => l.trim() !== '')
        if (lines.length === 0) return null
        if (block.trimStart().startsWith('```')) {
          return (
            <pre
              key={index}
              className="overflow-x-auto rounded-lg bg-surface-sunken p-3 text-[12.5px] leading-relaxed"
            >
              <code>{lines.filter((l) => !l.trimStart().startsWith('```')).join('\n')}</code>
            </pre>
          )
        }
        if (lines.every((l) => BULLET.test(l))) {
          return (
            <ul key={index} className="ml-4 list-disc space-y-1">
              {lines.map((line, i) => (
                <li key={i}>
                  <Inline text={line.replace(BULLET, '')} />
                </li>
              ))}
            </ul>
          )
        }
        if (lines.every((l) => NUMBERED.test(l))) {
          return (
            <ol key={index} className="ml-5 list-decimal space-y-1">
              {lines.map((line, i) => (
                <li key={i}>
                  <Inline text={line.replace(NUMBERED, '')} />
                </li>
              ))}
            </ol>
          )
        }
        const heading = lines.length === 1 ? HEADING.exec(lines[0]) : null
        if (heading) {
          return (
            <p key={index} className="text-[15px] font-semibold">
              <Inline text={heading[2]} />
            </p>
          )
        }
        return (
          <p key={index} className="whitespace-pre-wrap">
            <Inline text={lines.join('\n')} />
          </p>
        )
      })}
    </div>
  )
}

/** Open WebUI's in-progress mark: a blinking caret glued to the last character, not
 *  a spinner. Rendered only once there is content to glue it to. */
function Caret() {
  return (
    <span
      aria-hidden
      className="ml-0.5 inline-block h-3.5 w-[0.125rem] animate-pulse rounded-sm bg-ink-faint
        align-text-bottom"
    />
  )
}

/** One reply: reasoning folded above, answer, sources, actions. */
function Assistant({
  text,
  references,
  live,
  thinkingSeconds,
  onCitationClick,
  onRegenerate,
  regenerateDisabled,
}: {
  text: string
  references: Reference[]
  live: boolean
  thinkingSeconds?: number | null
  onCitationClick?: (ref: Reference) => void
  onRegenerate?: () => void
  regenerateDisabled?: boolean
}) {
  const split = splitThought(text)
  return (
    <div
      className="group max-w-[86%] rounded-card rounded-bl-md border border-edge bg-surface-raised
        px-4 py-3 text-[14.5px] leading-relaxed"
    >
      {split.thought !== null && (
        <ThinkingBlock text={split.thought} live={live && !split.answered} seconds={thinkingSeconds} />
      )}
      {(split.answer || !live) && (
        <div>
          <Answer text={split.answer} />
          {live && <Caret />}
        </div>
      )}
      <Citations references={references} onOpen={onCitationClick} />
      {!live && (
        <MessageActions
          copyText={answerOnly(text)}
          onRegenerate={onRegenerate}
          disabled={regenerateDisabled}
        />
      )}
    </div>
  )
}

interface Props {
  messages: ChatMessage[]
  streaming: boolean
  pendingAnswer: string
  pendingRefs: Reference[]
  pendingSteps?: ToolCall[]
  /** Seconds the in-flight reply spent reasoning before its answer began. */
  pendingThinkingSeconds?: number | null
  thinkingSecondsById?: ReadonlyMap<string, number>
  /** The scrolling element. Owned by the parent because the empty state shares it. */
  scrollRef: RefObject<HTMLDivElement>
  onCitationClick?: (ref: Reference) => void
  onRegenerate?: () => void
}

export default function MessageList({
  messages,
  streaming,
  pendingAnswer,
  pendingRefs,
  pendingSteps = [],
  pendingThinkingSeconds,
  thinkingSecondsById,
  scrollRef,
  onCitationClick,
  onRegenerate,
}: Props) {
  // Imperative, as in Open WebUI. `scrollIntoView` in an effect measured the DOM
  // before it had grown and lost the bottom on a fast stream; and CSS scroll
  // anchoring cannot know that the user scrolled up to read. `autoScroll` is that
  // knowledge: released when they scroll away, re-armed when they return.
  const autoScroll = useRef(true)

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const go = () => el.scrollTo({ top: el.scrollHeight })
    go()
    // Two frames: layout after the first token, then again after fonts and any
    // late block (a fenced code, a citation list) have settled.
    requestAnimationFrame(() => requestAnimationFrame(go))
  }, [scrollRef])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight
      autoScroll.current = distance < 48
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => el.removeEventListener('scroll', onScroll)
  }, [scrollRef])

  useEffect(() => {
    if (autoScroll.current) scrollToBottom()
  }, [messages.length, pendingAnswer, pendingSteps.length, streaming, scrollToBottom])

  // A new question always brings the bottom back into view, whatever the reader
  // had been doing — they asked it.
  useEffect(() => {
    autoScroll.current = true
    scrollToBottom()
  }, [messages.length, scrollToBottom])

  const lastAssistantIndex = messages.map((m) => m.role).lastIndexOf('assistant')

  return (
    <div className="flex flex-col gap-5">
      {messages.map((message, index) =>
        message.role === 'user' ? (
          <div key={message.id} className="flex justify-end">
            <div
              className="max-w-[78%] rounded-card rounded-br-md bg-ink px-4 py-2.5
                text-[14.5px] leading-relaxed text-ink-inverse"
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
            </div>
          </div>
        ) : (
          <div key={message.id} className="flex justify-start">
            <Assistant
              text={message.content}
              references={message.references}
              live={false}
              thinkingSeconds={thinkingSecondsById?.get(message.id) ?? null}
              onCitationClick={onCitationClick}
              onRegenerate={index === lastAssistantIndex && !streaming ? onRegenerate : undefined}
            />
          </div>
        ),
      )}

      {streaming && (
        <div className="flex justify-start">
          {pendingAnswer ? (
            <Assistant
              text={pendingAnswer}
              references={pendingRefs}
              live
              thinkingSeconds={pendingThinkingSeconds}
              onCitationClick={onCitationClick}
            />
          ) : (
            <div
              className="max-w-[86%] rounded-card rounded-bl-md border border-edge bg-surface-raised
                px-4 py-3 text-[14.5px] leading-relaxed"
            >
              <AgentSteps steps={pendingSteps} />
              {/* Status, not caret: there is nothing to glue a caret to yet. Retrieval
                  runs before the first token, so this covers a real wait. */}
              <span className="dot-pulse inline-flex items-center gap-1 text-ink-faint">
                <span className="h-1.5 w-1.5 rounded-full bg-violet-600" />
                <span className="h-1.5 w-1.5 rounded-full bg-violet-600" />
                <span className="h-1.5 w-1.5 rounded-full bg-violet-600" />
                <span className="ml-2 text-[12.5px]">
                  {pendingSteps.length ? 'Recherche…' : 'Récupération en cours…'}
                </span>
              </span>
              <Citations references={pendingRefs} onOpen={onCitationClick} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
