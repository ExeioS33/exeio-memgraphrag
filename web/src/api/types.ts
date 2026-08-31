/** Shapes returned by the MemGraphRAG API, as it actually answers today. */

export interface Reference {
  reference_id: string
  file_path: string
  /** Always null from /query — the server never puts passage text in a reference.
   *  Snippets come from /query/data or /documents/{id}/chunks instead. */
  content: string | null
}

export type Role = 'user' | 'assistant'

export interface ChatMessage {
  id: string
  thread_id: string
  role: Role
  content: string
  references: Reference[]
  created_at: number
}

export interface ChatThread {
  id: string
  owner: string
  title: string
  model: string | null
  params: Record<string, unknown>
  created_at: number
  updated_at: number
  messages?: ChatMessage[]
}

export interface ThreadListResponse {
  threads: ChatThread[]
  total: number
  limit: number
  offset: number
  returned: number
  next_offset: number | null
}

export type DocStatus = 'pending' | 'parsing' | 'processing' | 'processed' | 'failed'

export interface DocumentRecord {
  status: DocStatus
  file_path: string
  content_summary?: string
  content_length?: number
  parse_engine?: string
  parse_format?: string
  chunk_count?: number
  chunk_ids?: string[]
  created_at: number
  updated_at: number
  metadata?: { memory_sub_stage?: string | null; error?: string }
}

/** The list endpoint returns a map keyed by doc id, not an array. */
export interface DocumentListResponse {
  statuses: Record<string, DocumentRecord>
  total: number
  limit: number
  offset: number
  returned: number
  next_offset: number | null
  status?: string
}

export interface DocumentChunk {
  chunk_id: string
  content: string
  order: number
}

export interface DocumentChunksResponse {
  doc_id: string
  chunks: DocumentChunk[]
  total: number
  returned: number
}

export interface ModelsResponse {
  default: string | null
  models: string[]
}

export interface ParamSpec {
  name: string
  kind: 'choice' | 'int' | 'float' | 'bool' | 'str'
  emoji: string
  help: string
  default: unknown
  choices: string[] | null
  min: number | null
  max: number | null
  step: number | null
}

export interface QueryParamsResponse {
  params: ParamSpec[]
  presets: Record<string, Record<string, unknown>>
  supported_extensions: string[]
}

export interface HealthResponse {
  status: string
  core_version: string
  api_version: string
  auth_mode: 'enabled' | 'disabled'
  pipeline_busy: boolean
  ready: boolean
  retrieval_status: 'ready' | 'not_ready' | 'error'
  retrieval_error: string | null
}

export interface GraphNode {
  id: string
  label?: string
  layer?: string
  content?: string
  content_length?: number
  [key: string]: unknown
}

export interface GraphEdge {
  source: string
  target: string
  type?: string
  weight?: number
  [key: string]: unknown
}

export interface GraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
  label: string | null
  total_nodes: number
  returned_nodes: number
  returned_edges: number
  truncated: boolean
}

export interface QuerySettings {
  mode: 'ppr' | 'naive' | 'context' | 'bypass'
  model?: string | null
  top_k?: number
  linking_top_k?: number
  passage_node_weight?: number
  damping?: number
  fact_similarity_threshold?: number
  skip_fact_rerank?: boolean
  schema_top_k?: number
  schema_node_weight?: number
  user_prompt?: string | null
  /** Past turns. The server forwards these to the LLM but neither validates nor caps
   *  them, and never uses them to rewrite the retrieval query — a follow-up like
   *  "et le second ?" still retrieves against that literal text. */
  conversation_history?: { role: string; content: string }[]
}

/** One frame of the /query/stream SSE body. */
export type StreamFrame =
  | { kind: 'references'; references: Reference[] }
  | { kind: 'token'; text: string }
  | { kind: 'error'; message: string }
  | { kind: 'done' }
