import { useEffect, useRef, useState } from 'react'

import { BulbIcon, LayersIcon, SendIcon, SparkIcon } from './icons'

interface Props {
  disabled: boolean
  streaming: boolean
  onSend: (text: string) => void
  onStop: () => void
  onOpenSettings: () => void
}

const MAX_ROWS_PX = 190

export default function Composer({
  disabled,
  streaming,
  onSend,
  onStop,
  onOpenSettings,
}: Props) {
  const [value, setValue] = useState('')
  const textarea = useRef<HTMLTextAreaElement>(null)

  // Grow with the content up to a cap, then scroll inside — a fixed-height box
  // hides the start of a long question while it is being written.
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
    <div className="rounded-panel border border-edge bg-white shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
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
        className="w-full resize-none bg-transparent px-5 pb-2 pt-4 text-[15px] outline-none
          placeholder:text-ink-faint"
      />

      <div className="flex items-center justify-between gap-2 px-3 pb-2.5">
        <div className="flex items-center gap-1.5">
          <button className="icon-btn" onClick={onOpenSettings} title="Réglages de récupération">
            <LayersIcon size={16} />
          </button>
          <button
            className="icon-btn"
            title="Astuce : /naive, /context et /bypass changent le mode de récupération"
          >
            <BulbIcon size={16} />
          </button>
        </div>

        <div className="flex items-center gap-1.5">
          {streaming ? (
            <button
              onClick={onStop}
              className="inline-flex h-9 items-center rounded-full bg-ink px-4 text-sm
                font-medium text-white transition hover:bg-black"
            >
              Arrêter
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={disabled || !value.trim()}
              title="Envoyer"
              className="inline-flex h-9 w-9 items-center justify-center rounded-full
                bg-[linear-gradient(140deg,#CCB3FC_0%,#8B5CF6_100%)] text-white transition
                hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-35"
            >
              <SendIcon size={17} />
            </button>
          )}
        </div>
      </div>

      <div className="flex items-center justify-center rounded-b-panel border-t border-edge bg-violet-50 px-4 py-2">
        <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-muted">
          <SparkIcon size={13} className="text-violet-600" />
          Entrée pour envoyer · Maj+Entrée pour un saut de ligne
        </span>
      </div>
    </div>
  )
}
