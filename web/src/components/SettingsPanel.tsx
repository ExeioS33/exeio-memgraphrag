/**
 * Query settings, generated from /query/params.
 *
 * Nothing here hardcodes the parameter list: the server owns the registry, so a knob
 * added on the API side shows up without touching this file.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'

import { ApiError, queryParams } from '../api/client'
import type { ParamSpec, QueryParamsResponse, QuerySettings } from '../api/types'
import { CloseIcon, SlidersIcon } from './icons'

type SettingsRecord = Record<string, unknown>

type Props = {
  settings: QuerySettings
  onChange: (next: QuerySettings) => void
  onClose: () => void
}

function asRecord(settings: QuerySettings): SettingsRecord {
  return settings as unknown as SettingsRecord
}

function fromRecord(record: SettingsRecord): QuerySettings {
  return record as unknown as QuerySettings
}

function toNumber(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function bounds(spec: ParamSpec): { min: number; max: number; step: number } {
  const integer = spec.kind === 'int'
  return {
    min: spec.min ?? 0,
    max: spec.max ?? (integer ? 100 : 1),
    step: spec.step ?? (integer ? 1 : 0.05),
  }
}

function decimals(step: number): number {
  if (Number.isInteger(step)) return 0
  const text = String(step)
  const dot = text.indexOf('.')
  return dot < 0 ? 2 : Math.min(text.length - dot - 1, 4)
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.requestId ? `${error.message} (requête ${error.requestId})` : error.message
  }
  return error instanceof Error ? error.message : 'Erreur inconnue'
}

export default function SettingsPanel({ settings, onChange, onClose }: Props) {
  const [registry, setRegistry] = useState<QueryParamsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    queryParams()
      .then((response) => {
        if (!cancelled) {
          setRegistry(response)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(describeError(err))
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const current = useMemo(() => asRecord(settings), [settings])

  const update = useCallback(
    (name: string, value: unknown) => {
      onChange(fromRecord({ ...asRecord(settings), [name]: value }))
    },
    [onChange, settings],
  )

  const merge = useCallback(
    (values: SettingsRecord) => {
      onChange(fromRecord({ ...asRecord(settings), ...values }))
    },
    [onChange, settings],
  )

  const reset = useCallback(() => {
    if (!registry) return
    const defaults: SettingsRecord = {}
    for (const spec of registry.params) defaults[spec.name] = spec.default
    merge(defaults)
  }, [merge, registry])

  const presets = registry ? Object.entries(registry.presets) : []

  return (
    <div className="fixed inset-0 z-40 flex">
      <button type="button" aria-label="Fermer les réglages" onClick={onClose} className="flex-1 bg-ink/10 backdrop-blur-[1px]" />
      <aside className="flex h-full w-full max-w-md flex-col border-l border-edge bg-surface shadow-[0_0_60px_-20px_rgba(0,0,0,0.3)]">
        <header className="flex items-center gap-3 border-b border-edge px-6 py-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-violet-50 text-violet-600">
            <SlidersIcon size={17} />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-ink">Réglages de requête</h2>
            <p className="truncate text-xs text-ink-muted">Paramètres exposés par le serveur</p>
          </div>
          <button type="button" className="icon-btn" onClick={onClose} title="Fermer">
            <CloseIcon size={16} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          {loading ? (
            <p className="py-10 text-center text-sm text-ink-faint">Chargement des paramètres…</p>
          ) : error ? (
            <p className="rounded-card border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>
          ) : registry ? (
            <>
              {presets.length > 0 ? (
                <section className="mb-6">
                  <h3 className="mb-2 text-xs uppercase tracking-wide text-ink-faint">Préréglages</h3>
                  <div className="flex flex-wrap gap-2">
                    {presets.map(([name, values]) => (
                      <button
                        key={name}
                        type="button"
                        onClick={() => merge(values)}
                        className="rounded-full border border-edge bg-white px-3 py-1.5 text-xs text-ink-muted
                          transition hover:border-violet-300 hover:bg-violet-50 hover:text-violet-600"
                      >
                        {name}
                      </button>
                    ))}
                  </div>
                </section>
              ) : null}

              <div className="space-y-5">
                {registry.params.map((spec) => (
                  <Control key={spec.name} spec={spec} value={current[spec.name] ?? spec.default} onChange={update} />
                ))}
              </div>
            </>
          ) : null}
        </div>

        <footer className="flex items-center justify-between gap-3 border-t border-edge bg-surface-sunken px-6 py-4">
          <p className="text-xs text-ink-faint">Les valeurs s&apos;appliquent à la prochaine requête.</p>
          <button type="button" className="btn-ghost" disabled={!registry} onClick={reset}>
            Réinitialiser
          </button>
        </footer>
      </aside>
    </div>
  )
}

function Control({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec
  value: unknown
  onChange: (name: string, next: unknown) => void
}) {
  return (
    <section>
      <header className="mb-1.5 flex items-baseline gap-2">
        <span aria-hidden="true">{spec.emoji}</span>
        <span className="text-sm font-medium text-ink">{spec.name}</span>
      </header>
      <Field spec={spec} value={value} onChange={onChange} />
      <p className="mt-1.5 text-xs leading-relaxed text-ink-faint">{spec.help}</p>
    </section>
  )
}

function Field({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec
  value: unknown
  onChange: (name: string, next: unknown) => void
}) {
  if (spec.kind === 'choice') {
    const choices = spec.choices ?? []
    return (
      <select
        value={typeof value === 'string' ? value : ''}
        onChange={(event) => onChange(spec.name, event.target.value)}
        className="w-full rounded-full border border-edge-strong bg-white px-3 py-2 text-sm text-ink outline-none focus:border-violet-300"
      >
        {choices.map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    )
  }

  if (spec.kind === 'bool') {
    const on = value === true
    return (
      <button
        type="button"
        role="switch"
        aria-checked={on}
        onClick={() => onChange(spec.name, !on)}
        className={`relative h-6 w-11 rounded-full transition ${on ? 'bg-violet-600' : 'bg-edge-strong'}`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${on ? 'left-[22px]' : 'left-0.5'}`}
        />
      </button>
    )
  }

  if (spec.kind === 'int' || spec.kind === 'float') {
    const { min, max, step } = bounds(spec)
    const numeric = toNumber(value, toNumber(spec.default, min))
    return (
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={numeric}
          onChange={(event) => {
            const next = Number(event.target.value)
            onChange(spec.name, spec.kind === 'int' ? Math.round(next) : next)
          }}
          className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-edge-strong accent-violet-600"
        />
        <span className="w-14 shrink-0 text-right font-mono text-xs text-ink">
          {numeric.toFixed(spec.kind === 'int' ? 0 : decimals(step))}
        </span>
      </div>
    )
  }

  return (
    <textarea
      rows={3}
      value={typeof value === 'string' ? value : ''}
      onChange={(event) => onChange(spec.name, event.target.value)}
      placeholder="Laisser vide pour la valeur par défaut"
      className="w-full resize-y rounded-card border border-edge-strong bg-white px-3 py-2 text-sm text-ink
        outline-none placeholder:text-ink-faint focus:border-violet-300"
    />
  )
}
