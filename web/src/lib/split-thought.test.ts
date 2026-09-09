import { describe, expect, it } from 'vitest'

import { answerOnly, splitThought } from './split-thought'

describe('splitThought', () => {
  it('separates the reasoning from the answer on the first Answer: marker', () => {
    const split = splitThought('Thought: the budget is in [1].\nAnswer: 42 euros [1].')
    expect(split.thought).toBe('the budget is in [1].')
    expect(split.answer).toBe('42 euros [1].')
    expect(split.answered).toBe(true)
  })

  it('treats a reply without any marker as answer, never as reasoning', () => {
    // Agent mode answers through its own prompt and emits no marker at all.
    const split = splitThought('The capital is Paris [2].')
    expect(split.thought).toBeNull()
    expect(split.answer).toBe('The capital is Paris [2].')
    expect(split.answered).toBe(false)
  })

  it('shows a reply that has only started reasoning as reasoning in progress', () => {
    const split = splitThought('Thought: looking at the passages')
    expect(split.thought).toBe('looking at the passages')
    expect(split.answer).toBe('')
    expect(split.answered).toBe(false)
  })

  it('accepts a bold marker, a French label and a marker at the very start', () => {
    expect(splitThought('**Thought:** a\n**Answer:** b').answer).toBe('b')
    expect(splitThought('Réflexion : a\nRéponse : b').thought).toBe('a')
    expect(splitThought('Answer: only').answer).toBe('only')
    expect(splitThought('Answer: only').thought).toBeNull()
  })

  it('splits on the first marker only, so an "Answer:" quoted later stays in the answer', () => {
    const split = splitThought('Thought: x\nAnswer: first. The doc says "Answer: no".')
    expect(split.answer).toBe('first. The doc says "Answer: no".')
  })

  it('does not mistake a mid-sentence "answer:" for the marker', () => {
    const split = splitThought('Thought: the short answer: none yet')
    expect(split.answered).toBe(false)
    expect(split.thought).toBe('the short answer: none yet')
  })

  it('handles empty input', () => {
    expect(splitThought('')).toEqual({ thought: null, answer: '', answered: false })
  })
})

describe('answerOnly', () => {
  it('copies the answer without the reasoning', () => {
    expect(answerOnly('Thought: a\nAnswer: b')).toBe('b')
  })

  it('copies the whole text when there is no reasoning', () => {
    expect(answerOnly('plain')).toBe('plain')
  })

  it('copies nothing while a reply is still reasoning', () => {
    expect(answerOnly('Thought: still going')).toBe('')
  })
})
