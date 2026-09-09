import { useEffect, useState } from 'react'

import { ChevronRightIcon } from './icons'

interface Props {
  text: string
  /** Still being generated: the block stays open and streams. */
  live: boolean
  /** Seconds the reasoning took, when known. Not persisted across reloads. */
  seconds?: number | null
}

/**
 * The model's reasoning, folded away once the answer starts.
 *
 * Open WebUI's pattern for reasoning models, and the shape a reader wants: while
 * nothing else has arrived the reasoning *is* the progress indicator, so it stays
 * open and streams; the moment the answer begins it folds, and the summary line
 * keeps the elapsed time so the wait is accounted for rather than hidden.
 *
 * A native <details>: keyboard and screen-reader behaviour for free, and the
 * open/closed state is the element's own, so a reader who reopens a block keeps it
 * open until they close it.
 */
export default function ThinkingBlock({ text, live, seconds }: Props) {
  // Open while live; folded on the render where `live` turns false. After that
  // the user owns the toggle — the effect only fires on the transition.
  const [open, setOpen] = useState(live)
  useEffect(() => {
    setOpen(live)
  }, [live])

  const label = live
    ? 'Réflexion…'
    : seconds != null
      ? `Réflexion · ${seconds < 1 ? '<1' : Math.round(seconds)} s`
      : 'Réflexion'

  return (
    <details
      open={open}
      onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
      className="group mb-2.5 rounded-lg border border-edge bg-surface-sunken/60 text-[13px]"
    >
      <summary
        className="flex cursor-pointer select-none items-center gap-1.5 px-3 py-1.5 text-ink-muted
          transition hover:text-ink [&::-webkit-details-marker]:hidden"
      >
        <ChevronRightIcon
          size={14}
          className="shrink-0 transition-transform group-open:rotate-90 motion-reduce:transition-none"
        />
        <span className="font-medium">{label}</span>
        {live && (
          <span className="ml-1 inline-block h-3 w-[2px] animate-pulse rounded-sm bg-ink-faint" />
        )}
      </summary>
      <div
        className="max-h-[320px] overflow-y-auto whitespace-pre-wrap border-t border-edge px-3 py-2
          leading-relaxed text-ink-muted"
      >
        {text}
      </div>
    </details>
  )
}
