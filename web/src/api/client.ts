import type {
  ChatMessage,
  ChatThread,
  CypherResponse,
  DocumentChunksResponse,
  DocumentListResponse,
  GraphHighlights,
  GraphResponse,
  GraphSchema,
  HealthResponse,
  LibraryPassages,
  LibraryPreview,
  LibraryTree,
  ModelsResponse,
  QueryParamsResponse,
  QuerySettings,
  Reference,
  StreamFrame,
  ThreadListResponse,
} from './types'

const TOKEN_KEY = 'memgraphrag.token'

/**
 * Typed error over the API's single failure shape.
 *
 * Worth knowing about this server: a missing credential yields **403**, not 401
 * (memgraphrag/api/dependencies.py). The usual "redirect to login on 401" reflex
 * would miss it, so `isAuthError` treats both as unauthenticated.
 */
export class ApiError extends Error {
  readonly status: number
  readonly requestId: string | null
  readonly retryAfter: number | null

  constructor(status: number, message: string, requestId: string | null, retryAfter: number | null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.requestId = requestId
    this.retryAfter = retryAfter
  }

  get isAuthError(): boolean {
    return this.status === 401 || this.status === 403
  }
}

function readToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY)
  } catch {
    // Private windows and blocked site data throw here rather than returning null.
    return null
  }
}

export function storeToken(token: string | null): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token)
    else window.localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* a browser refusing storage is not a reason to fail the request */
  }
}

export function hasToken(): boolean {
  return Boolean(readToken())
}

function authHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra)
  const token = readToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return headers
}

async function toApiError(response: Response): Promise<ApiError> {
  let detail = `${response.status} ${response.statusText}`
  try {
    const body = await response.json()
    if (body && typeof body.detail === 'string') detail = body.detail
    else if (typeof body === 'string') detail = body
  } catch {
    /* a non-JSON body leaves the status line as the message */
  }
  const retryRaw = response.headers.get('Retry-After')
  return new ApiError(
    response.status,
    detail,
    // Readable only because the server sets expose_headers; without that the
    // browser hides both from JS even though they are on the wire.
    response.headers.get('X-Request-ID'),
    retryRaw ? Number(retryRaw) : null,
  )
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = authHeaders(init.headers)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  const response = await fetch(path, { ...init, headers })
  if (!response.ok) throw await toApiError(response)
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

/* ------------------------------------------------------------------ auth --- */

export async function login(username: string, password: string): Promise<string> {
  // /login takes form encoding (OAuth2PasswordRequestForm), not JSON.
  const body = new URLSearchParams({ username, password })
  const response = await fetch('/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (!response.ok) throw await toApiError(response)
  const data = (await response.json()) as { access_token: string }
  storeToken(data.access_token)
  return data.access_token
}

export function logout(): void {
  storeToken(null)
}

export const health = () => request<HealthResponse>('/health')
export const listModels = () => request<ModelsResponse>('/models')
export const queryParams = () => request<QueryParamsResponse>('/query/params')

/* ----------------------------------------------------------------- chat --- */

export const listThreads = (limit = 100, offset = 0) =>
  request<ThreadListResponse>(`/chat/threads?limit=${limit}&offset=${offset}`)

export const getThread = (id: string) => request<ChatThread>(`/chat/threads/${id}`)

export const createThread = (payload: { title?: string; model?: string | null } = {}) =>
  request<ChatThread>('/chat/threads', { method: 'POST', body: JSON.stringify(payload) })

export const renameThread = (id: string, title: string) =>
  request<ChatThread>(`/chat/threads/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) })

export const deleteThread = (id: string) =>
  request<{ status: string }>(`/chat/threads/${id}`, { method: 'DELETE' })

export const appendMessage = (
  threadId: string,
  payload: { role: 'user' | 'assistant'; content: string; references?: Reference[] },
) =>
  request<ChatMessage>(`/chat/threads/${threadId}/messages`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })

/* ------------------------------------------------------------ documents --- */

export const listDocuments = (limit = 100, offset = 0, status?: string) => {
  const qs = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (status) qs.set('status', status)
  return request<DocumentListResponse>(`/documents/?${qs}`)
}

export const documentChunks = (docId: string, limit = 50) =>
  request<DocumentChunksResponse>(`/documents/${docId}/chunks?limit=${limit}`)

export const deleteDocument = (docId: string) =>
  request<unknown>(`/documents/${docId}`, { method: 'DELETE' })

export const requeueDocument = (docId: string) =>
  request<unknown>(`/documents/${docId}/requeue`, { method: 'POST' })

export const scanInputDir = () => request<unknown>('/documents/scan', { method: 'POST' })

export async function uploadFile(file: File): Promise<unknown> {
  const form = new FormData()
  form.append('file', file)
  return request<unknown>('/documents/upload', { method: 'POST', body: form })
}

export const insertText = (text: string) =>
  request<unknown>('/documents/text', { method: 'POST', body: JSON.stringify({ text }) })

/* ---------------------------------------------------------------- graph --- */

export const graphLabels = () => request<{ labels: string[] }>('/graph/label/list')

export const exploreGraph = (label: string | null, limit = 200) => {
  const qs = new URLSearchParams({ limit: String(limit) })
  if (label) qs.set('label', label)
  return request<GraphResponse>(`/graphs?${qs}`)
}

/** Run a read-only Cypher statement. Writes are refused server-side. */
export const runCypher = (query: string, limit?: number) =>
  request<CypherResponse>('/graph/cypher', {
    method: 'POST',
    body: JSON.stringify({ query, ...(limit ? { limit } : {}) }),
  })

export const graphSchema = () => request<GraphSchema>('/graph/schema')

export const graphNeighborhood = (entityId: string, hops = 1, limit = 100) =>
  request<CypherResponse>(
    `/graph/neighborhood?entity_id=${encodeURIComponent(entityId)}&hops=${hops}&limit=${limit}`,
  )

export const graphHighlights = () => request<GraphHighlights>('/graph/highlights')

/* --------------------------------------------------------------- library - */

export const libraryTree = () => request<LibraryTree>('/library/tree')

export const libraryPreview = (path: string, page = 1, pages = 1) =>
  request<LibraryPreview>(
    `/library/preview?path=${encodeURIComponent(path)}&page=${page}&pages=${pages}`,
  )

export const libraryPassages = (path: string, limit = 30) =>
  request<LibraryPassages>(
    `/library/passages?path=${encodeURIComponent(path)}&limit=${limit}`,
  )

/** URL for the raw file. Served inline so the browser's PDF viewer can render it.
 *  Note the static mount is unauthenticated but this route is not — an <iframe>
 *  cannot carry the bearer header, so the token is passed as a query parameter,
 *  which the server accepts for this read-only route only. */
export const libraryFileUrl = (path: string): string => {
  const qs = new URLSearchParams({ path })
  const token = readToken()
  if (token) qs.set('token', token)
  return `/library/file?${qs}`
}

/* --------------------------------------------------------------- streaming - */

function parseFrame(raw: string): StreamFrame | null {
  const line = raw.trim()
  if (!line.startsWith('data:')) return null
  const payload = line.slice(5).trim()
  if (!payload) return null
  if (payload === '[DONE]') return { kind: 'done' }
  try {
    const data = JSON.parse(payload) as Record<string, unknown>
    if (typeof data.error === 'string') return { kind: 'error', message: data.error }
    if (Array.isArray(data.references)) {
      return { kind: 'references', references: data.references as Reference[] }
    }
    if (typeof data.response === 'string') return { kind: 'token', text: data.response }
  } catch {
    /* a malformed frame is skipped rather than aborting the stream */
  }
  return null
}

/**
 * Stream an answer.
 *
 * The server emits `references` once, then one `response` frame per token, then
 * `[DONE]`. Retrieval itself is not streamed — PPR has no partial result — so the
 * first frame still arrives only after a full retrieval.
 */
export async function* streamQuery(
  query: string,
  settings: QuerySettings,
  signal?: AbortSignal,
): AsyncGenerator<StreamFrame> {
  const response = await fetch('/query/stream', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ ...settings, query, stream: true }),
    signal,
  })
  if (!response.ok) throw await toApiError(response)
  if (!response.body) throw new ApiError(500, 'Réponse sans corps', null, null)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // SSE separates events with a blank line; the tail is kept for the next chunk.
      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''
      for (const event of events) {
        const frame = parseFrame(event)
        if (frame) yield frame
      }
    }
    const trailing = parseFrame(buffer)
    if (trailing) yield trailing
  } finally {
    reader.releaseLock()
  }
}
