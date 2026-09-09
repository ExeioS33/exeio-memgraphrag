/**
 * Sign-in and sign-up.
 *
 * /login and /auth/signup are rate-limited per IP, so a 429 carries a Retry-After
 * the dialog counts down rather than letting the user hammer it. A 403 on login
 * is shown verbatim: it only ever means "right password, account pending or
 * deactivated", which the user has just proved they are entitled to hear.
 */
import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'

import { ApiError, login, signup } from '../api/client'
import { BrandMark } from './icons'

const DEFAULT_COOLDOWN = 30
const MIN_PASSWORD = 8

type Mode = 'login' | 'signup'

function describeError(error: unknown, mode: Mode): string {
  if (error instanceof ApiError) {
    if (error.status === 429) return 'Trop de tentatives. Patientez avant de réessayer.'
    if (error.status === 403 || error.status === 503) return error.message
    if (error.status === 401) return 'Identifiants refusés.'
    if (error.status === 422) return mode === 'signup' ? 'Adresse e-mail invalide.' : error.message
    return error.requestId ? `${error.message} (requête ${error.requestId})` : error.message
  }
  return error instanceof Error ? error.message : 'Erreur inconnue'
}

const field =
  'w-full rounded-lg border border-edge-strong bg-surface-raised px-3 py-2 text-sm text-ink ' +
  'outline-none placeholder:text-ink-faint focus:border-violet-300'

export default function LoginDialog({
  signupEnabled,
  onSuccess,
}: {
  signupEnabled: boolean
  onSuccess: () => void
}) {
  const [mode, setMode] = useState<Mode>('login')
  const [username, setUsername] = useState('')
  const [name, setName] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [cooldown, setCooldown] = useState(0)

  useEffect(() => {
    if (cooldown <= 0) return
    const timer = window.setInterval(() => {
      setCooldown((seconds) => (seconds <= 1 ? 0 : seconds - 1))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [cooldown])

  const switchMode = (next: Mode) => {
    setMode(next)
    setError(null)
    setNotice(null)
  }

  const submit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      if (submitting || cooldown > 0) return
      setSubmitting(true)
      setError(null)
      setNotice(null)
      try {
        if (mode === 'signup') {
          const result = await signup(username, name, password)
          // The first account is the administrator and can sign in at once;
          // anyone else waits, and is told so rather than left guessing.
          setNotice(
            result.status === 'admin'
              ? `Premier compte créé : vous êtes administrateur et pouvez vous connecter.${
                  result.threads_adopted ? ` ${result.threads_adopted} discussion(s) existante(s) vous ont été rattachées.` : ''
                }`
              : 'Compte créé. Un administrateur doit le valider avant que vous puissiez vous connecter.',
          )
          setMode('login')
          setPassword('')
        } else {
          await login(username, password)
          onSuccess()
        }
      } catch (err) {
        setError(describeError(err, mode))
        if (err instanceof ApiError && err.status === 429) {
          setCooldown(Math.max(1, Math.round(err.retryAfter ?? DEFAULT_COOLDOWN)))
        }
      } finally {
        setSubmitting(false)
      }
    },
    [cooldown, mode, name, onSuccess, password, submitting, username],
  )

  const blocked = submitting || cooldown > 0
  const passwordTooShort = mode === 'signup' && password.length > 0 && password.length < MIN_PASSWORD

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-surface-sunken px-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="login-title"
        className="w-full max-w-sm animate-fade-up rounded-panel border border-edge bg-surface p-7
          shadow-[0_24px_60px_-24px_rgba(0,0,0,0.5)]"
      >
        <div className="flex flex-col items-center text-center">
          <BrandMark size={40} />
          <h1 id="login-title" className="mt-4 text-lg font-semibold text-ink">
            {mode === 'signup' ? 'Créer un compte' : 'Connexion'}
          </h1>
          <p className="mt-1 text-sm text-ink-muted">Accédez à votre mémoire MemGraphRAG.</p>
        </div>

        {signupEnabled && (
          <div className="mt-5 grid grid-cols-2 rounded-full border border-edge bg-surface-sunken p-0.5 text-sm">
            {(['login', 'signup'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => switchMode(m)}
                className={`rounded-full py-1.5 transition ${
                  mode === m ? 'bg-surface-raised text-ink shadow-sm' : 'text-ink-muted hover:text-ink'
                }`}
              >
                {m === 'login' ? 'Se connecter' : "S'inscrire"}
              </button>
            ))}
          </div>
        )}

        <form className="mt-5 space-y-3" onSubmit={(event) => void submit(event)}>
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">
              {signupEnabled ? 'Adresse e-mail' : 'Identifiant'}
            </span>
            <input
              type={signupEnabled ? 'email' : 'text'}
              autoComplete={signupEnabled ? 'email' : 'username'}
              autoFocus
              required
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder={signupEnabled ? 'vous@exemple.fr' : 'utilisateur'}
              className={field}
            />
          </label>

          {mode === 'signup' && (
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Nom affiché</span>
              <input
                type="text"
                autoComplete="name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Prénom Nom"
                className={field}
              />
            </label>
          )}

          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">Mot de passe</span>
            <input
              type="password"
              autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
              required
              minLength={mode === 'signup' ? MIN_PASSWORD : undefined}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="••••••••"
              className={field}
            />
            {passwordTooShort && (
              <span className="mt-1 block text-[11px] text-ink-faint">
                Au moins {MIN_PASSWORD} caractères.
              </span>
            )}
          </label>

          {error ? (
            <p role="alert" className="rounded-lg border border-red-300/40 bg-red-500/10 px-3 py-2 text-xs text-red-500">
              {error}
            </p>
          ) : null}
          {notice ? (
            <p role="status" className="rounded-lg border border-violet-300/40 bg-violet-50 px-3 py-2 text-xs text-ink">
              {notice}
            </p>
          ) : null}

          <button type="submit" className="btn-dark w-full" disabled={blocked || passwordTooShort}>
            {submitting
              ? mode === 'signup'
                ? 'Création…'
                : 'Connexion…'
              : cooldown > 0
                ? `Réessayez dans ${cooldown} s`
                : mode === 'signup'
                  ? 'Créer le compte'
                  : 'Se connecter'}
          </button>
        </form>

        <p className="mt-5 text-center text-[11px] leading-relaxed text-ink-faint">
          {signupEnabled
            ? 'Le premier compte créé devient administrateur ; les suivants sont validés par lui.'
            : "Si aucun compte n'est configuré côté serveur, celui-ci délivre un jeton invité."}
        </p>
      </div>
    </div>
  )
}
