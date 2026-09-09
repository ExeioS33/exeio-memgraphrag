import { useCallback, useEffect, useState } from 'react'

import * as api from '../api/client'
import type { AdminUser } from '../api/types'
import { CheckIcon, CloseIcon, KeyIcon, ShieldIcon } from './icons'

interface Props {
  selfId: string | null
  onClose: () => void
}

const ROLE_LABEL: Record<string, string> = {
  admin: 'Administrateur',
  user: 'Utilisateur',
  pending: 'En attente',
}

/**
 * Where "validation admin" happens. Without a page for it the approval routes are
 * an API nobody can reach without curl; with it, a pending account is a row with
 * a button. Deactivation takes effect on the account's next request — the server
 * reloads account state per token — not at the token's expiry.
 */
export default function AdminPanel({ selfId, onClose }: Props) {
  const [users, setUsers] = useState<AdminUser[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const data = await api.listUsers()
      setUsers(data.users)
      setError(null)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const run = async (id: string, action: () => Promise<unknown>) => {
    setBusy(id)
    try {
      await action()
      await load()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(null)
    }
  }

  const reset = (user: AdminUser) => {
    const password = window.prompt(`Nouveau mot de passe pour ${user.email} (8 caractères minimum)`)
    if (!password) return
    if (password.length < 8) {
      setError('Le mot de passe doit faire au moins 8 caractères.')
      return
    }
    void run(user.id, () => api.resetPassword(user.id, password))
  }

  const pending = users.filter((u) => u.role === 'pending')
  const others = users.filter((u) => u.role !== 'pending')

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-ink/30 backdrop-blur-[2px]">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="admin-title"
        className="flex h-full w-full max-w-[640px] flex-col border-l border-edge bg-surface animate-fade-up"
      >
        <header className="flex items-center gap-2 border-b border-edge px-5 py-3">
          <ShieldIcon size={18} className="text-violet-500" />
          <h2 id="admin-title" className="flex-1 text-[15px] font-semibold">
            Administration des comptes
          </h2>
          <button className="icon-btn" onClick={onClose} title="Fermer">
            <CloseIcon size={16} />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {error && (
            <p role="alert" className="mb-3 rounded-lg border border-red-300/40 bg-red-500/10 px-3 py-2 text-xs text-red-500">
              {error}
            </p>
          )}

          <h3 className="mb-2 text-[12px] font-medium uppercase tracking-wide text-ink-faint">
            En attente de validation {pending.length ? `· ${pending.length}` : ''}
          </h3>
          {pending.length === 0 ? (
            <p className="mb-5 text-[13px] text-ink-faint">Aucune demande en attente.</p>
          ) : (
            <ul className="mb-5 flex flex-col gap-2">
              {pending.map((user) => (
                <li
                  key={user.id}
                  className="flex items-center gap-3 rounded-card border border-edge bg-surface-raised px-3 py-2"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13.5px] font-medium">{user.name}</p>
                    <p className="truncate text-[11.5px] text-ink-faint">{user.email}</p>
                  </div>
                  <button
                    type="button"
                    className="btn-dark py-1.5 text-[12.5px]"
                    disabled={busy === user.id}
                    onClick={() => void run(user.id, () => api.approveUser(user.id))}
                  >
                    <CheckIcon size={14} />
                    Valider
                  </button>
                  <button
                    type="button"
                    className="btn-ghost py-1.5 text-[12.5px]"
                    disabled={busy === user.id}
                    onClick={() => void run(user.id, () => api.deactivateUser(user.id))}
                  >
                    Refuser
                  </button>
                </li>
              ))}
            </ul>
          )}

          <h3 className="mb-2 text-[12px] font-medium uppercase tracking-wide text-ink-faint">
            Comptes · {others.length}
          </h3>
          <ul className="flex flex-col gap-2">
            {others.map((user) => {
              const isSelf = user.id === selfId
              return (
                <li
                  key={user.id}
                  className={`flex items-center gap-3 rounded-card border border-edge px-3 py-2 ${
                    user.active ? 'bg-surface-raised' : 'bg-surface-sunken opacity-70'
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13.5px] font-medium">
                      {user.name}
                      {isSelf && <span className="ml-1.5 text-[11px] text-ink-faint">(vous)</span>}
                    </p>
                    <p className="truncate text-[11.5px] text-ink-faint">
                      {user.email} · {ROLE_LABEL[user.role] ?? user.role}
                      {!user.active && ' · désactivé'}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="icon-btn"
                    title="Réinitialiser le mot de passe"
                    disabled={busy === user.id}
                    onClick={() => reset(user)}
                  >
                    <KeyIcon size={15} />
                  </button>
                  {!isSelf && user.active && (
                    <button
                      type="button"
                      className="btn-ghost py-1.5 text-[12.5px]"
                      disabled={busy === user.id}
                      onClick={() => void run(user.id, () => api.deactivateUser(user.id))}
                    >
                      Désactiver
                    </button>
                  )}
                </li>
              )
            })}
          </ul>

          <p className="mt-5 text-[11.5px] leading-relaxed text-ink-faint">
            Il n&apos;y a pas de réinitialisation de mot de passe en libre-service : elle se fait
            ici. Un compte désactivé perd l&apos;accès à sa prochaine requête, sans attendre
            l&apos;expiration de son jeton.
          </p>
        </div>
      </section>
    </div>
  )
}
