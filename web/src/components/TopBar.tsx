import { useEffect, useRef, useState } from 'react'

import type { ChatMessage } from '../api/types'
import { BrandMark, ChevronDownIcon, DotsIcon, DownloadIcon, LinkIcon, SlidersIcon } from './icons'

interface Props {
  models: string[]
  model: string | null
  onModelChange: (model: string) => void
  mode: string
  messages: ChatMessage[]
  threadTitle: string | null
  onOpenSettings: () => void
}

/** Export the visible conversation as Markdown, entirely client-side — the API has
 *  no export endpoint and does not need one for this. */
function exportChat(title: string, messages: ChatMessage[]): void {
  const lines = [`# ${title}`, '']
  for (const message of messages) {
    lines.push(message.role === 'user' ? '## Question' : '## Réponse')
    lines.push('', message.content, '')
    if (message.references.length) {
      lines.push('**Sources**', '')
      for (const ref of message.references) {
        lines.push(`- [${ref.reference_id}] ${ref.file_path}`)
      }
      lines.push('')
    }
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${title.replace(/[^\w\-À-ÿ ]+/g, '').trim() || 'discussion'}.md`
  anchor.click()
  URL.revokeObjectURL(url)
}

export default function TopBar({
  models,
  model,
  onModelChange,
  mode,
  messages,
  threadTitle,
  onOpenSettings,
}: Props) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onClick = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  useEffect(() => {
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(false), 1800)
    return () => window.clearTimeout(timer)
  }, [copied])

  return (
    <header className="flex items-center justify-between gap-3 px-5 py-3">
      <div className="relative" ref={menuRef}>
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 rounded-full border border-edge bg-white
            py-1.5 pl-1.5 pr-3 text-sm transition hover:border-edge-strong"
        >
          <BrandMark size={22} />
          <span className="max-w-[220px] truncate font-medium">{model ?? 'Modèle par défaut'}</span>
          <ChevronDownIcon size={15} className="text-ink-faint" />
        </button>
        {open && (
          <div
            className="absolute left-0 top-[calc(100%+6px)] z-30 w-[290px] overflow-hidden
              rounded-card border border-edge bg-white py-1 shadow-lg"
          >
            {models.length === 0 ? (
              <p className="px-3 py-2 text-xs text-ink-faint">Aucun modèle exposé par le serveur.</p>
            ) : (
              models.map((name) => (
                <button
                  key={name}
                  onClick={() => {
                    onModelChange(name)
                    setOpen(false)
                  }}
                  className={`block w-full truncate px-3 py-2 text-left text-sm transition
                    hover:bg-surface-sunken ${name === model ? 'text-violet-700' : 'text-ink'}`}
                >
                  {name}
                </button>
              ))
            )}
            <p className="border-t border-edge px-3 pb-1 pt-2 text-[11px] leading-snug text-ink-faint">
              Liste définie par <code className="font-mono">LLM_MODELS</code>. Un modèle absent
              de cette liste est refusé côté serveur.
            </p>
          </div>
        )}
      </div>

      <div className="flex items-center gap-1.5">
        <span className="mr-1 rounded-full bg-violet-50 px-2.5 py-1 text-[11px] font-medium text-violet-700">
          mode {mode}
        </span>
        <button className="icon-btn" onClick={onOpenSettings} title="Réglages de récupération">
          <SlidersIcon size={17} />
        </button>
        <button
          className="icon-btn"
          title={copied ? 'Lien copié' : 'Copier le lien'}
          onClick={() => {
            void navigator.clipboard?.writeText(window.location.href).then(() => setCopied(true))
          }}
        >
          <LinkIcon size={17} />
        </button>
        <button
          className="btn-ghost"
          disabled={messages.length === 0}
          onClick={() => exportChat(threadTitle ?? 'Discussion', messages)}
        >
          <DownloadIcon size={15} />
          Exporter
        </button>
        <button className="icon-btn" title="Plus d'actions">
          <DotsIcon size={17} />
        </button>
      </div>
    </header>
  )
}
