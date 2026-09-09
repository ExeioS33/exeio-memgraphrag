import { useEffect, useRef, useState } from 'react'

import type { AuthUser } from '../api/types'
import { useTheme } from '../state/useTheme'
import { LogoutIcon, MoonIcon, ShieldIcon, SunIcon } from './icons'

interface Props {
  user: AuthUser
  persistent: boolean
  onLogout: () => void
  onOpenAdmin?: () => void
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean)
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase()
  return name.slice(0, 2).toUpperCase()
}

const ROLE_LABEL: Record<string, string> = {
  admin: 'Administrateur',
  user: 'Utilisateur',
  guest: 'Invité',
  pending: 'En attente',
}

/**
 * The account pinned to the bottom of the sidebar, as in Open WebUI, opening a
 * small menu upward: role, theme toggle, administration for admins, sign out.
 */
export default function AccountMenu({ user, persistent, onLogout, onOpenAdmin }: Props) {
  const [open, setOpen] = useState(false)
  const [theme, toggleTheme] = useTheme()
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onClick = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false)
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

  return (
    <div className="relative" ref={ref}>
      {open && (
        <div
          className="absolute bottom-[calc(100%+6px)] left-0 z-30 w-full min-w-[220px] rounded-card border
            border-edge bg-surface-raised py-1 shadow-[0_18px_40px_-20px_rgba(0,0,0,0.45)] animate-fade-up"
        >
          <div className="border-b border-edge px-3 py-2">
            <p className="truncate text-[13px] font-medium">{user.name}</p>
            <p className="truncate text-[11px] text-ink-faint">
              {user.email ?? ROLE_LABEL[user.role] ?? user.role}
              {user.email ? ` · ${ROLE_LABEL[user.role] ?? user.role}` : ''}
            </p>
          </div>
          <button type="button" className="nav-item" onClick={toggleTheme}>
            {theme === 'dark' ? <SunIcon size={16} /> : <MoonIcon size={16} />}
            {theme === 'dark' ? 'Thème clair' : 'Thème sombre'}
          </button>
          {user.role === 'admin' && onOpenAdmin && (
            <button
              type="button"
              className="nav-item"
              onClick={() => {
                setOpen(false)
                onOpenAdmin()
              }}
            >
              <ShieldIcon size={16} />
              Administration
            </button>
          )}
          <button type="button" className="nav-item" onClick={onLogout}>
            <LogoutIcon size={16} />
            Se déconnecter
          </button>
        </div>
      )}

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition
          hover:bg-surface-raised"
      >
        <span
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-violet-600
            text-[12px] font-semibold text-white"
        >
          {initials(user.name || 'MG')}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13.5px] font-medium leading-tight">{user.name}</span>
          <span className="block truncate text-[11px] text-ink-faint">
            {persistent ? (ROLE_LABEL[user.role] ?? user.role) : 'Session locale'}
          </span>
        </span>
      </button>
    </div>
  )
}
