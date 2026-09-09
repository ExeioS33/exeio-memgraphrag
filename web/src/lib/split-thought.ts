/**
 * Split a RAG answer into its reasoning and its answer.
 *
 * The QA prompt asks the model to start after "Thought:" and conclude with
 * "Answer:" (memgraphrag/prompts/templates.py, RAG_QA_SYSTEM). That chain of
 * reasoning is kept — its benefit was never measured, so dropping it would be a
 * quality change dressed as a display fix — and separated here, at render time.
 *
 * The fallback direction is the decision that matters: when no marker is found,
 * everything is the answer, never everything is reasoning. Agent mode answers
 * through a different system prompt and emits no marker at all; a heuristic that
 * hid its output behind a collapsed block would look exactly like a broken reply.
 */

export interface SplitAnswer {
  /** Reasoning text, or null when the reply carries none. */
  thought: string | null
  /** What the user is meant to read. */
  answer: string
  /** True once an "Answer:" marker has been seen; false while still reasoning. */
  answered: boolean
}

// Marker at a line start, optionally bold, with or without a trailing space.
// The prompt is English; "Réponse" is accepted because a French-pinned corpus
// (MEMGRAPHRAG_LANGUAGE) has been seen to translate the label itself.
const ANSWER_MARKER = /(?:^|\n)[ \t]*(?:\*\*)?(?:Answer|Réponse)[ \t]*:(?:\*\*)?[ \t]*/i
const THOUGHT_LABEL = /^[ \t]*(?:\*\*)?(?:Thought|Réflexion)[ \t]*:(?:\*\*)?[ \t]*/i

export function splitThought(text: string): SplitAnswer {
  const source = text ?? ''
  const match = ANSWER_MARKER.exec(source)
  if (match) {
    const before = source.slice(0, match.index)
    const after = source.slice(match.index + match[0].length)
    const thought = before.replace(THOUGHT_LABEL, '').trim()
    return { thought: thought || null, answer: after.trim(), answered: true }
  }
  if (THOUGHT_LABEL.test(source)) {
    // The label is there and the marker is not yet: we are inside the reasoning.
    return { thought: source.replace(THOUGHT_LABEL, '').trim() || null, answer: '', answered: false }
  }
  return { thought: null, answer: source, answered: false }
}

/** The text to put on the clipboard: the answer alone, never the reasoning. */
export function answerOnly(text: string): string {
  const split = splitThought(text)
  return split.answer || (split.thought === null ? text : '')
}
