import { ChartIcon, FeatherIcon, GraphIcon, Orb } from './icons'

interface Suggestion {
  title: string
  body: string
  prompt: string
  Icon: typeof ChartIcon
}

/** Phrased for a knowledge base rather than a general assistant — these run against
 *  whatever corpus was ingested, so they ask the engine to synthesise across it. */
const SUGGESTIONS: Suggestion[] = [
  {
    title: 'Synthétiser',
    body: 'Résume les points clés des documents ingérés en cinq puces.',
    prompt: 'Résume les points clés des documents ingérés en cinq puces, avec les sources.',
    Icon: ChartIcon,
  },
  {
    title: 'Relier les faits',
    body: 'Quelles entités reviennent le plus et comment sont-elles liées ?',
    prompt: 'Quelles entités reviennent le plus dans le corpus et comment sont-elles reliées ?',
    Icon: GraphIcon,
  },
  {
    title: 'Vérifier',
    body: 'Compare deux notions du corpus et signale les contradictions.',
    prompt:
      'Compare les deux notions les plus proches du corpus et signale les contradictions éventuelles.',
    Icon: FeatherIcon,
  },
]

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

export function SuggestionCards({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      {SUGGESTIONS.map(({ title, body, prompt, Icon }) => (
        <button
          key={title}
          onClick={() => onPick(prompt)}
          className="rounded-card border border-edge bg-white p-4 text-left transition
            hover:border-violet-300 hover:shadow-[0_2px_10px_rgba(139,92,246,0.10)]"
        >
          <Icon size={18} className="text-violet-600" />
          <p className="mt-2.5 text-[13px] font-semibold">{title}</p>
          <p className="mt-1 text-[12px] leading-relaxed text-ink-muted">{body}</p>
        </button>
      ))}
    </div>
  )
}
