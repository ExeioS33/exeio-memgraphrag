import { useEffect, useRef, useState } from 'react'

import type { ChatMessage, ProviderInfo } from '../api/types'
import { ChevronDownIcon, DownloadIcon, SlidersIcon } from './icons'

interface Props {
  providers: ProviderInfo[]
  provider: string
  model: string | null
  /** Models this account was refused for during the session (Together's
   *  "non-serverless" 400). Still listed, but marked, so the same click does not
   *  fail twice with the same opaque error. */
  unavailableModels: ReadonlySet<string>
  onSelect: (provider: string, model: string) => void
  messages: ChatMessage[]
  threadTitle: string | null
  onOpenSettings: () => void
}

/** Export the visible conversation as Markdown, entirely client-side. */
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

/**
 * The model is the page title, as in Open WebUI: large, with a chevron, and the
 * provider as a quiet subtitle. The picker lists every provider's real catalogue.
 */
export default function TopBar({
  providers,
  provider,
  model,
  unavailableModels,
  onSelect,
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
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const current = providers.find((p) => p.id === provider)
  const providerLabel = current?.label ?? provider

  return (
    <header className="flex items-start justify-between gap-3 px-5 pb-2 pt-3">
      <div className="relative min-w-0" ref={menuRef}>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-haspopup="listbox"
          aria-expanded={open}
          className="group flex max-w-[520px] items-center gap-1.5 rounded-lg px-2 py-1 text-left
            transition hover:bg-surface-raised"
        >
          <span className="truncate text-[17px] font-medium tracking-tight">
            {model ?? 'Modèle par défaut'}
          </span>
          <ChevronDownIcon
            size={16}
            className="shrink-0 text-ink-faint transition group-hover:text-ink"
          />
        </button>
        <p className="px-2 text-[11.5px] text-ink-faint">{providerLabel}</p>

        {open && (
          <div
            role="listbox"
            className="absolute left-0 top-[calc(100%+4px)] z-30 max-h-[70vh] w-[360px] overflow-y-auto
              rounded-card border border-edge bg-surface-raised py-1 shadow-[0_18px_40px_-20px_rgba(0,0,0,0.45)]"
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
                      const refused = unavailableModels.has(name)
                      return (
                        <button
                          key={`${p.id}:${name}`}
                          type="button"
                          role="option"
                          aria-selected={selected}
                          onClick={() => {
                            onSelect(p.id, name)
                            setOpen(false)
                          }}
                          title={refused ? 'Refusé par le fournisseur : non activé sur ce compte' : name}
                          className={`flex w-full items-center justify-between gap-2 px-3 py-2 text-left
                            text-sm transition hover:bg-surface-sunken
                            ${selected ? 'text-violet-500' : refused ? 'text-ink-faint' : 'text-ink'}`}
                        >
                          <span className="truncate">{name}</span>
                          {refused && (
                            <span className="shrink-0 text-[10.5px] uppercase tracking-wide">
                              non activé
                            </span>
                          )}
                        </button>
                      )
                    })
                  )}
                </div>
              ))
            )}
            <p className="border-t border-edge px-3 pb-1 pt-2 text-[11px] leading-snug text-ink-faint">
              Le modèle d’embedding est verrouillé : le corpus est déjà indexé avec lui.
            </p>
          </div>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-1">
        <button className="icon-btn" onClick={onOpenSettings} title="Réglages de récupération">
          <SlidersIcon size={17} />
        </button>
        <button
          className="icon-btn"
          disabled={messages.length === 0}
          onClick={() => exportChat(threadTitle ?? 'Discussion', messages)}
          title="Exporter la discussion en Markdown"
        >
          <DownloadIcon size={16} />
        </button>
      </div>
    </header>
  )
}
