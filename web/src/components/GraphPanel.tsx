/**
 * Full-screen graph explorer.
 *
 * The panel itself is only chrome: the header, the close affordances, and a full-height
 * host for {@link CypherConsole}, which owns the editor, the result tabs and the
 * double-click expansion through /graph/neighborhood.
 *
 * It went full-screen rather than staying a right drawer because the console needs three
 * columns (base inventory / editor + result / result summary), which a drawer cannot hold
 * without shrinking the graph to a thumbnail.
 */
import { useEffect } from 'react'

import CypherConsole from './CypherConsole'
import { CloseIcon, GraphIcon } from './icons'

export default function GraphPanel({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-40 flex flex-col bg-surface">
      <header className="flex shrink-0 items-center gap-3 border-b border-edge px-5 py-3">
        <span className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-violet-50 text-violet-600">
          <GraphIcon size={17} />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-semibold text-ink">Graphe mémoire</h2>
          <p className="truncate text-xs text-ink-muted">
            Console Cypher en lecture seule sur les schémas, faits, entités et passages
            installés dans le graphe
          </p>
        </div>
        <span className="hidden text-[11px] text-ink-faint md:inline">Échap pour fermer</span>
        <button type="button" className="icon-btn" onClick={onClose} title="Fermer">
          <CloseIcon size={16} />
        </button>
      </header>

      <CypherConsole />
    </div>
  )
}
