import { useEffect, useRef } from 'react'

import type { ChatMessage, Reference } from '../api/types'
import { FileIcon } from './icons'

function basename(path: string): string {
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] || path
}

function Citations({ references }: { references: Reference[] }) {
  if (references.length === 0) return null
  return (
    <div className="mt-3 border-t border-edge pt-2.5">
      <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
        Sources
      </p>
      <ul className="flex flex-wrap gap-1.5">
        {references.map((ref) => (
          <li key={`${ref.reference_id}-${ref.file_path}`}>
            <span
              title={ref.file_path}
              className="inline-flex max-w-[280px] items-center gap-1.5 rounded-full border
                border-edge bg-surface-sunken px-2.5 py-1 text-[12px] text-ink-muted"
            >
              <FileIcon size={13} className="shrink-0 text-violet-600" />
              <span className="shrink-0 font-medium text-ink">[{ref.reference_id}]</span>
              <span className="truncate">{basename(ref.file_path)}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
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
}

export default function MessageList({ messages, streaming, pendingAnswer, pendingRefs }: Props) {
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
              <Citations references={message.references} />
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
            <Citations references={pendingRefs} />
          </div>
        </div>
      )}
      <div ref={bottom} />
    </div>
  )
}
