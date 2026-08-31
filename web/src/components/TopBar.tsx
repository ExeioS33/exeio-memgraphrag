import { useEffect, useRef, useState } from 'react'

import type { ChatMessage, ProviderInfo } from '../api/types'
import { BrandMark, ChevronDownIcon, DownloadIcon, SlidersIcon } from './icons'

interface Props {
  providers: ProviderInfo[]
  provider: string
  model: string | null
  onSelect: (provider: string, model: string) => void
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
  providers,
  provider,
  model,
  onSelect,
  mode,
  messages,
  threadTitle,
  onOpenSettings,
}: Props) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onClick = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  const current = providers.find((p) => p.id === provider)
  const label = current?.label ?? provider
  const pill = model ? `${label} · ${model}` : 'Modèle par défaut'

  return (
    <header className="flex items-center justify-between gap-3 px-5 py-3">
      <div className="relative" ref={menuRef}>
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 rounded-full border border-edge bg-white
            py-1.5 pl-1.5 pr-3 text-sm transition hover:border-edge-strong"
        >
          <BrandMark size={22} />
          <span className="max-w-[260px] truncate font-medium">{pill}</span>
          <ChevronDownIcon size={15} className="text-ink-faint" />
        </button>
        {open && (
          <div
            className="absolute left-0 top-[calc(100%+6px)] z-30 max-h-[70vh] w-[320px]
              overflow-y-auto rounded-card border border-edge bg-white py-1 shadow-lg"
          >
            {providers.length === 0 ? (
              <p className="px-3 py-2 text-xs text-ink-faint">
                Aucun fournisseur exposé par le serveur.
              </p>
            ) : (
              providers.map((p) => (
                <div key={p.id} className="py-1">
                  <p
                    className={`px-3 pb-1 pt-1 text-[11px] font-semibold uppercase tracking-wide
                      ${p.available ? 'text-ink-muted' : 'text-ink-faint'}`}
                  >
                    {p.label}
                  </p>
                  {!p.available ? (
                    <p className="px-3 pb-1 text-[11px] leading-snug text-ink-faint">
                      Indisponible : renseignez <code className="font-mono">{p.models_env}</code>{' '}
                      côté serveur.
                    </p>
                  ) : p.models.length === 0 ? (
                    <p className="px-3 pb-1 text-[11px] leading-snug text-ink-faint">
                      Aucun modèle dans <code className="font-mono">{p.models_env}</code>.
                    </p>
                  ) : (
                    p.models.map((name) => {
                      const selected = p.id === provider && name === model
                      return (
                        <button
                          key={`${p.id}:${name}`}
                          onClick={() => {
                            onSelect(p.id, name)
                            setOpen(false)
                          }}
                          className={`block w-full truncate px-3 py-2 text-left text-sm transition
                            hover:bg-surface-sunken ${selected ? 'text-violet-700' : 'text-ink'}`}
                        >
                          {name}
                        </button>
                      )
                    })
                  )}
                </div>
              ))
            )}
            <p className="border-t border-edge px-3 pb-1 pt-2 text-[11px] leading-snug text-ink-faint">
              Le modèle d’embedding est verrouillé : le corpus est déjà indexé avec lui, en changer
              rendrait les vecteurs existants incomparables.
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
          className="btn-ghost"
          disabled={messages.length === 0}
          onClick={() => exportChat(threadTitle ?? 'Discussion', messages)}
        >
          <DownloadIcon size={15} />
          Exporter
        </button>
      </div>
    </header>
  )
}
