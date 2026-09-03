import { useCallback, useEffect, useRef, useState } from 'react'

import * as api from '../api/client'
import { ApiError } from '../api/client'
import type { ChatMessage, ChatThread, QuerySettings, Reference, ToolCall } from '../api/types'

/** Turns kept in `conversation_history`. The server neither validates nor caps it,
 *  and an unbounded history walks straight into the model's context limit. */
const HISTORY_TURNS = 12

function localId(): string {
  return `local-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`
}

function draftMessage(threadId: string, role: 'user' | 'assistant', content: string): ChatMessage {
  return {
    id: localId(),
    thread_id: threadId,
    role,
    content,
    references: [],
    created_at: Math.floor(Date.now() / 1000),
  }
}

function titleFrom(text: string): string {
  const collapsed = text.replace(/\s+/g, ' ').trim()
  return collapsed.length > 120 ? `${collapsed.slice(0, 120)}…` : collapsed || 'Nouvelle discussion'
}

export interface ChatState {
  threads: ChatThread[]
  activeId: string | null
  messages: ChatMessage[]
  streaming: boolean
  pendingAnswer: string
  pendingRefs: Reference[]
  /** Agent-mode steps for the turn in flight; empty in every other mode. */
  pendingSteps: ToolCall[]
  error: string | null
  /** False when the server has no chat database; threads then live only in this tab. */
  persistent: boolean
}

/** Union of two reference lists, keyed by the citation number the answer uses.
 *
 *  A hop cannot renumber another hop's passages — the server continues the count
 *  across a turn — so `reference_id` is a stable key here. Later frames win on a
 *  collision, which only happens if a server ever restarts the numbering.
 */
function mergeReferences(current: Reference[], incoming: Reference[]): Reference[] {
  const byId = new Map<string, Reference>()
  for (const ref of [...current, ...incoming]) byId.set(ref.reference_id, ref)
  return [...byId.values()].sort(
    (a, b) => Number(a.reference_id) - Number(b.reference_id),
  )
}

export function useChat(settings: QuerySettings) {
  const [threads, setThreads] = useState<ChatThread[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [pendingAnswer, setPendingAnswer] = useState('')
  const [pendingRefs, setPendingRefs] = useState<Reference[]>([])
  const [pendingSteps, setPendingSteps] = useState<ToolCall[]>([])
  const [error, setError] = useState<string | null>(null)
  const [persistent, setPersistent] = useState(true)
  const abortRef = useRef<AbortController | null>(null)
  // Latest settings without making every callback depend on them.
  const settingsRef = useRef(settings)
  settingsRef.current = settings

  const describe = useCallback((exc: unknown): string => {
    if (exc instanceof ApiError) {
      return exc.requestId ? `${exc.message} (requête ${exc.requestId})` : exc.message
    }
    return exc instanceof Error ? exc.message : String(exc)
  }, [])

  /** A 503 from /chat/* means no application database; degrade to in-tab threads
   *  rather than blocking the whole UI on infrastructure the user may not need. */
  const handleChatError = useCallback(
    (exc: unknown): boolean => {
      if (exc instanceof ApiError && exc.status === 503) {
        setPersistent(false)
        return true
      }
      return false
    },
    [],
  )

  const refreshThreads = useCallback(async () => {
    try {
      const data = await api.listThreads(200, 0)
      setThreads(data.threads)
      setPersistent(true)
    } catch (exc) {
      if (handleChatError(exc)) return
      setError(describe(exc))
    }
  }, [describe, handleChatError])

  useEffect(() => {
    void refreshThreads()
  }, [refreshThreads])

  const openThread = useCallback(
    async (id: string) => {
      setActiveId(id)
      setPendingAnswer('')
      setPendingRefs([])
      setPendingSteps([])
      const known = threads.find((t) => t.id === id)
      if (!persistent) {
        setMessages(known?.messages ?? [])
        return
      }
      try {
        const thread = await api.getThread(id)
        setMessages(thread.messages ?? [])
      } catch (exc) {
        if (handleChatError(exc)) {
          setMessages(known?.messages ?? [])
          return
        }
        setError(describe(exc))
      }
    },
    [describe, handleChatError, persistent, threads],
  )

  const newThread = useCallback(() => {
    setActiveId(null)
    setMessages([])
    setPendingAnswer('')
    setPendingRefs([])
    setPendingSteps([])
    setError(null)
  }, [])

  const removeThread = useCallback(
    async (id: string) => {
      if (persistent) {
        try {
          await api.deleteThread(id)
        } catch (exc) {
          if (!handleChatError(exc)) {
            setError(describe(exc))
            return
          }
        }
      }
      setThreads((prev) => prev.filter((t) => t.id !== id))
      if (activeId === id) newThread()
    },
    [activeId, describe, handleChatError, newThread, persistent],
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setStreaming(false)
  }, [])

  const send = useCallback(
    async (text: string) => {
      const question = text.trim()
      if (!question || streaming) return
      setError(null)

      // Resolve the thread first so both messages land in the same place.
      let threadId = activeId
      if (!threadId) {
        const title = titleFrom(question)
        if (persistent) {
          try {
            const created = await api.createThread({ title, model: settingsRef.current.model })
            threadId = created.id
            setThreads((prev) => [created, ...prev])
          } catch (exc) {
            if (!handleChatError(exc)) {
              setError(describe(exc))
              return
            }
          }
        }
        if (!threadId) {
          const now = Math.floor(Date.now() / 1000)
          const localThread: ChatThread = {
            id: localId(),
            owner: 'guest',
            title,
            model: settingsRef.current.model ?? null,
            params: {},
            created_at: now,
            updated_at: now,
            messages: [],
          }
          threadId = localThread.id
          setThreads((prev) => [localThread, ...prev])
        }
        setActiveId(threadId)
      }

      const userMessage = draftMessage(threadId, 'user', question)
      const history = [...messages, userMessage]
      setMessages(history)
      setPendingAnswer('')
      setPendingRefs([])
      setPendingSteps([])
      setStreaming(true)

      if (persistent) {
        try {
          await api.appendMessage(threadId, { role: 'user', content: question })
        } catch (exc) {
          handleChatError(exc)
        }
      }

      const controller = new AbortController()
      abortRef.current = controller
      let answer = ''
      let refs: Reference[] = []
      try {
        const conversation = history
          .slice(-HISTORY_TURNS - 1, -1)
          .map((m) => ({ role: m.role, content: m.content }))
        const stream = api.streamQuery(
          question,
          { ...settingsRef.current, conversation_history: conversation },
          controller.signal,
        )
        for await (const frame of stream) {
          if (frame.kind === 'token') {
            answer += frame.text
            setPendingAnswer(answer)
          } else if (frame.kind === 'references') {
            // Merged, not replaced. Agent mode can retrieve more than once in a
            // turn, and each frame carries only that hop's passages — overwriting
            // dropped every source but the last one's, while the answer went on
            // citing all of them.
            refs = mergeReferences(refs, frame.references)
            setPendingRefs(refs)
          } else if (frame.kind === 'tool_call') {
            setPendingSteps((prev) => [...prev, frame.call])
          } else if (frame.kind === 'error') {
            throw new Error(frame.message)
          }
        }
      } catch (exc) {
        if (!(exc instanceof DOMException && exc.name === 'AbortError')) {
          setError(describe(exc))
        }
      } finally {
        abortRef.current = null
        setStreaming(false)
      }

      if (answer) {
        const assistant: ChatMessage = { ...draftMessage(threadId, 'assistant', answer), references: refs }
        setMessages((prev) => [...prev, assistant])
        if (persistent) {
          try {
            await api.appendMessage(threadId, {
              role: 'assistant',
              content: answer,
              references: refs,
            })
          } catch (exc) {
            handleChatError(exc)
          }
        }
        setThreads((prev) =>
          prev.map((t) =>
            t.id === threadId
              ? { ...t, updated_at: Math.floor(Date.now() / 1000), title: t.title || titleFrom(question) }
              : t,
          ),
        )
      }
      setPendingAnswer('')
      setPendingRefs([])
      setPendingSteps([])
    },
    [activeId, describe, handleChatError, messages, persistent, streaming],
  )

  useEffect(() => () => abortRef.current?.abort(), [])

  return {
    threads,
    activeId,
    messages,
    streaming,
    pendingAnswer,
    pendingRefs,
    pendingSteps,
    error,
    persistent,
    setError,
    refreshThreads,
    openThread,
    newThread,
    removeThread,
    send,
    stop,
  }
}
