import type { GraphSuggestion } from '../api/types'
import { ChartIcon, FeatherIcon, GraphIcon, Orb } from './icons'

/** The suggestions are built from the graph, so the icon follows what the server
 *  keyed the suggestion on rather than a fixed order. */
const ICONS: Record<GraphSuggestion['kind'], typeof ChartIcon> = {
  entity: GraphIcon,
  schema: ChartIcon,
  type: FeatherIcon,
}

export default function EmptyState({ account }: { account: string }) {
  return (
    <div className="flex flex-col items-center px-6 pb-4 pt-6 text-center animate-fade-up">
      <Orb />
      <h1 className="mt-5 text-[26px] font-semibold leading-tight text-gradient-violet">
        Bonjour {account}
      </h1>
      <p className="mt-1 text-[26px] font-semibold leading-tight tracking-tight">
        Que puis-je chercher pour vous ?
      </p>
      <p className="mt-3 max-w-[520px] text-[13px] leading-relaxed text-ink-muted">
        Les réponses sont construites à partir des documents que vous avez ingérés, avec les
        passages sources cités.
      </p>
    </div>
  )
}

interface SuggestionCardsProps {
  suggestions: GraphSuggestion[]
  onPick: (prompt: string) => void
}

export function SuggestionCards({ suggestions, onPick }: SuggestionCardsProps) {
  // They arrive from an async call against the graph; hold the layout with a
  // skeleton so the composer does not jump once they land.
  if (suggestions.length === 0) {
    return (
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="animate-pulse rounded-card border border-edge bg-white p-4">
            <div className="h-[18px] w-[18px] rounded-full bg-surface-sunken" />
            <div className="mt-2.5 h-[13px] w-1/2 rounded bg-surface-sunken" />
            <div className="mt-2 h-[12px] w-full rounded bg-surface-sunken" />
            <div className="mt-1.5 h-[12px] w-4/5 rounded bg-surface-sunken" />
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      {suggestions.map(({ title, body, prompt, kind }) => {
        const Icon = ICONS[kind] ?? ChartIcon
        return (
          <button
            key={`${kind}:${title}`}
            onClick={() => onPick(prompt)}
            className="rounded-card border border-edge bg-white p-4 text-left transition
              hover:border-violet-300 hover:shadow-[0_2px_10px_rgba(139,92,246,0.10)]"
          >
            <Icon size={18} className="text-violet-600" />
            <p className="mt-2.5 text-[13px] font-semibold">{title}</p>
            <p className="mt-1 text-[12px] leading-relaxed text-ink-muted">{body}</p>
          </button>
        )
      })}
    </div>
  )
}
