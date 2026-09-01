/**
 * Sign-in modal.
 *
 * /login is rate-limited per IP (LOGIN_MAX_ATTEMPTS / LOGIN_WINDOW_SECONDS), so a 429
 * carries a Retry-After the dialog counts down rather than letting the user hammer it.
 */
import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'

import { ApiError, login } from '../api/client'
import { BrandMark } from './icons'

const DEFAULT_COOLDOWN = 30

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 429) return 'Trop de tentatives. Patientez avant de réessayer.'
    const base = error.isAuthError ? 'Identifiants refusés.' : error.message
    return error.requestId ? `${base} (requête ${error.requestId})` : base
  }
  return error instanceof Error ? error.message : 'Erreur inconnue'
}

export default function LoginDialog({ onSuccess }: { onSuccess: () => void }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [cooldown, setCooldown] = useState(0)

  useEffect(() => {
    if (cooldown <= 0) return
    const timer = window.setInterval(() => {
      setCooldown((seconds) => (seconds <= 1 ? 0 : seconds - 1))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [cooldown])

  const submit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      if (submitting || cooldown > 0) return
      setSubmitting(true)
      setError(null)
      try {
        await login(username, password)
        onSuccess()
      } catch (err) {
        setError(describeError(err))
        if (err instanceof ApiError && err.status === 429) {
          setCooldown(Math.max(1, Math.round(err.retryAfter ?? DEFAULT_COOLDOWN)))
        }
      } finally {
        setSubmitting(false)
      }
    },
    [cooldown, onSuccess, password, submitting, username],
  )

  const blocked = submitting || cooldown > 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/25 px-4 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="login-title"
        className="w-full max-w-sm animate-fade-up rounded-panel border border-edge bg-surface p-7
          shadow-[0_24px_60px_-24px_rgba(0,0,0,0.35)]"
      >
        <div className="flex flex-col items-center text-center">
          <BrandMark size={40} />
          <h1 id="login-title" className="mt-4 text-lg font-semibold text-ink">
            Connexion
          </h1>
          <p className="mt-1 text-sm text-ink-muted">Accédez à votre mémoire MemGraphRAG.</p>
        </div>

        <form className="mt-6 space-y-3" onSubmit={(event) => void submit(event)}>
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">Identifiant</span>
            <input
              type="text"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="utilisateur"
              className="w-full rounded-card border border-edge-strong bg-white px-3 py-2 text-sm text-ink
                outline-none placeholder:text-ink-faint focus:border-violet-300"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">Mot de passe</span>
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="••••••••"
              className="w-full rounded-card border border-edge-strong bg-white px-3 py-2 text-sm text-ink
                outline-none placeholder:text-ink-faint focus:border-violet-300"
            />
          </label>

          {error ? (
            <p role="alert" className="rounded-card border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {error}
            </p>
          ) : null}

          <button type="submit" className="btn-dark w-full" disabled={blocked}>
            {submitting
              ? 'Connexion…'
              : cooldown > 0
                ? `Réessayez dans ${cooldown} s`
                : 'Se connecter'}
          </button>
        </form>

        <p className="mt-5 text-center text-[11px] leading-relaxed text-ink-faint">
          Si aucun compte n&apos;est configuré côté serveur, celui-ci délivre un jeton invité : n&apos;importe quels
          identifiants sont alors acceptés.
        </p>
      </div>
    </div>
  )
}
