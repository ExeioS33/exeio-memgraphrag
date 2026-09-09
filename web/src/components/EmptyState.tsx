import type { GraphSuggestion } from '../api/types'
import { Orb, SparkIcon } from './icons'

/**
 * Open WebUI centres its mark above the model name and puts the composer right
 * under both. Ours is the violet orb from the original mockup — the one brand
 * element the project has, and the reason not to borrow one.
 */
export default function EmptyState({ model, account }: { model: string | null; account: string }) {
  return (
    <div className="flex flex-col items-center text-center animate-fade-up">
      <Orb />
      <h1 className="mt-4 text-[28px] font-semibold leading-tight tracking-tight">
        {model ?? 'MemGraphRAG'}
      </h1>
      <p className="mt-1.5 text-[14px] text-ink-muted">
        Bonjour {account} — que puis-je chercher dans vos documents ?
      </p>
    </div>
  )
}

interface SuggestionListProps {
  suggestions: GraphSuggestion[]
  onPick: (prompt: string) => void
}

/**
 * Suggestions as a list — bold title, muted second line — rather than cards. They
 * are derived from the graph, so they name entities that are actually in the
 * corpus; a hardcoded set goes stale the moment the corpus does.
 */
export function SuggestionList({ suggestions, onPick }: SuggestionListProps) {
  return (
    <div className="mt-6 w-full max-w-[560px] animate-fade-up">
      <p className="mb-2 flex items-center gap-1.5 text-[12.5px] text-ink-faint">
        <SparkIcon size={13} />
        Suggestions
      </p>
      {suggestions.length === 0 ? (
        <ul className="flex flex-col gap-3" aria-busy>
          {[0, 1, 2].map((i) => (
            <li key={i} className="animate-pulse">
              <div className="h-[15px] w-2/5 rounded bg-surface-raised" />
              <div className="mt-1.5 h-[12px] w-3/5 rounded bg-surface-raised" />
            </li>
          ))}
        </ul>
      ) : (
        <ul className="flex flex-col">
          {suggestions.map(({ title, body, prompt, kind }) => (
            <li key={`${kind}:${title}`}>
              <button
                type="button"
                onClick={() => onPick(prompt)}
                className="group -mx-2 flex w-[calc(100%+1rem)] flex-col items-start rounded-lg px-2 py-2
                  text-left transition hover:bg-surface-raised"
              >
                <span className="text-[15px] font-medium text-ink group-hover:text-violet-500">
                  {title}
                </span>
                <span className="text-[12.5px] text-ink-muted">{body}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
