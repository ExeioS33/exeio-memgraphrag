import { useEffect, useRef, useState } from 'react'

import { PlusIcon, SendIcon, StopIcon } from './icons'

interface Props {
  disabled: boolean
  streaming: boolean
  mode: string
  onSend: (text: string) => void
  onStop: () => void
  onOpenSettings: () => void
  /** Larger, centred variant for the empty screen. */
  hero?: boolean
}

const MAX_ROWS_PX = 190

/**
 * Open WebUI's composer: one rounded field, with the actions on a second row
 * *inside* it rather than beside it — `+` for the retrieval settings on the left,
 * the send / stop control on the right.
 */
export default function Composer({
  disabled,
  streaming,
  mode,
  onSend,
  onStop,
  onOpenSettings,
  hero = false,
}: Props) {
  const [value, setValue] = useState('')
  const textarea = useRef<HTMLTextAreaElement>(null)

  // Grow with the content up to a cap, then scroll inside.
  useEffect(() => {
    const node = textarea.current
    if (!node) return
    node.style.height = 'auto'
    node.style.height = `${Math.min(node.scrollHeight, MAX_ROWS_PX)}px`
  }, [value])

  const submit = () => {
    const text = value.trim()
    if (!text || disabled) return
    onSend(text)
    setValue('')
  }

  return (
    <div
      className={`rounded-[20px] border border-edge bg-surface-raised transition-colors
        focus-within:border-edge-strong ${hero ? 'shadow-[0_8px_30px_-18px_rgba(0,0,0,0.35)]' : ''}`}
    >
      <textarea
        ref={textarea}
        rows={1}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            submit()
          }
        }}
        placeholder="Posez votre question…"
        title="Entrée pour envoyer · Maj+Entrée pour un saut de ligne"
        aria-label="Votre question"
        className={`w-full resize-none bg-transparent px-4 pt-3.5 outline-none
          placeholder:text-ink-faint ${hero ? 'text-[16px] pb-1' : 'text-[15px] pb-1'}`}
      />

      <div className="flex items-center justify-between gap-2 px-2.5 pb-2.5 pt-1">
        <div className="flex items-center gap-1">
          <button
            type="button"
            className="icon-btn"
            onClick={onOpenSettings}
            title="Réglages de récupération"
            aria-label="Réglages de récupération"
          >
            <PlusIcon size={18} />
          </button>
          <span className="rounded-full bg-violet-50 px-2.5 py-1 text-[11px] font-medium text-violet-700">
            {mode}
          </span>
        </div>

        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            title="Arrêter"
            aria-label="Arrêter la génération"
            className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-ink
              text-ink-inverse transition hover:opacity-90"
          >
            <StopIcon size={14} />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={disabled || !value.trim()}
            title="Envoyer"
            aria-label="Envoyer"
            className="inline-flex h-9 w-9 items-center justify-center rounded-full
              bg-[linear-gradient(140deg,#CCB3FC_0%,#8B5CF6_100%)] text-white transition
              hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-35"
          >
            <SendIcon size={17} />
          </button>
        )}
      </div>
    </div>
  )
}
