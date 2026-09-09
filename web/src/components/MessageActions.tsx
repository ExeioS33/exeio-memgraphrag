import { useEffect, useState } from 'react'

import { CheckIcon, CopyIcon, RefreshIcon } from './icons'

interface Props {
  /** What goes on the clipboard: the answer alone, without any reasoning. */
  copyText: string
  /** Only the last answer can be regenerated; earlier ones get copy only. */
  onRegenerate?: () => void
  disabled?: boolean
}

/**
 * The two actions Open WebUI attaches to every reply. Copy never includes the
 * reasoning block — the clipboard has no use for "Thought:" — and regenerate
 * replaces the last turn rather than appending a second answer to the same
 * question.
 */
export default function MessageActions({ copyText, onRegenerate, disabled }: Props) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(false), 1600)
    return () => window.clearTimeout(timer)
  }, [copied])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(copyText)
      setCopied(true)
    } catch {
      /* clipboard denied (insecure context): nothing to recover, the button just does not confirm */
    }
  }

  return (
    <div
      className="mt-2 flex items-center gap-0.5 text-ink-faint opacity-0 transition-opacity
        group-hover:opacity-100 focus-within:opacity-100 motion-reduce:transition-none"
    >
      <button
        type="button"
        onClick={() => void copy()}
        title={copied ? 'Copié' : 'Copier la réponse'}
        aria-label="Copier la réponse"
        className="icon-btn h-7 w-7"
      >
        {copied ? <CheckIcon size={14} className="text-violet-600" /> : <CopyIcon size={14} />}
      </button>
      {onRegenerate && (
        <button
          type="button"
          onClick={onRegenerate}
          disabled={disabled}
          title="Régénérer la réponse"
          aria-label="Régénérer la réponse"
          className="icon-btn h-7 w-7 disabled:opacity-40"
        >
          <RefreshIcon size={14} />
        </button>
      )}
    </div>
  )
}
