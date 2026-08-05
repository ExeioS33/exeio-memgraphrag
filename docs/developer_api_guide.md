# MemGraphRAG developer API guide

Curl-oriented guide for calling the HTTP API as an external service.
For deploy, storage, and Compose, see [MemGraphRAG-API-Server.md](MemGraphRAG-API-Server.md).
For CLI / Streamlit / Python client, see [Clients.md](Clients.md).
Interactive OpenAPI: `http://localhost:9621/docs`.

Default base URL in examples: `http://localhost:9621`.

---

## 1. Auth

| Mode | How |
|------|-----|
| Open (dev) | No `MEMGRAPHRAG_API_KEY` / `AUTH_ACCOUNTS` — omit auth headers on loopback |
| API key | Header `X-API-Key: <key>` |
| JWT | `POST /login` then `Authorization: Bearer <token>` |

```bash
# Optional JWT
curl -sS -X POST http://localhost:9621/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}'
```

---

## 2. Answer contract (`POST /query`)

Response shape matches LightRAG’s query API:

```json
{
  "response": "…freeform answer text (Thought:/Answer: style from the QA prompt)…",
  "references": [
    {
      "reference_id": "1",
      "file_path": "example.pdf",
      "content": null
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `response` | Generated answer string. Render as text/Markdown in your UI. |
| `references` | Unique source documents from **retrieval** (not scraped from the answer). Use this list for citations in clients. |

`references[].content` is reserved (usually `null`). Do not rely on a `### References` section inside `response`; prefer the structured array.

QA system prompt (Thought:/Answer:) lives in `memgraphrag/prompts/templates.py` (`RAG_QA_SYSTEM`).

---

## 3. Query

### Modes

| `mode` | Behavior |
|--------|----------|
| `ppr` (default) | Fact linking + Personalized PageRank + QA |
| `naive` | Dense passage retrieval + QA |
| `context` | Retrieval only (`response` may be null; see `/query/data`) |
| `bypass` | Direct LLM, no corpus |

### Basic query

```bash
curl -sS -X POST http://localhost:9621/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is MemGraphRAG?",
    "mode": "ppr",
    "top_k": 5
  }'
```

### Useful optional body fields

| Field | Type | Notes |
|-------|------|--------|
| `top_k` | int | Passages to keep after ranking |
| `linking_top_k` | int | Fact-linking candidates |
| `only_need_context` | bool | Skip QA; retrieval only |
| `user_prompt` | string | Extra instruction appended for generation |
| `conversation_history` | `[{role, content}, …]` | Multi-turn context |
| `damping` / `passage_node_weight` / `schema_top_k` / … | — | Retrieval tuning (see OpenAPI) |

---

## 4. Retrieval only (`POST /query/data`)

Same request body; returns evidence without requiring a full QA answer:

```bash
curl -sS -X POST http://localhost:9621/query/data \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is MemGraphRAG?","mode":"ppr","top_k":5}'
```

```json
{
  "status": "success",
  "data": {
    "response": null,
    "references": [{"reference_id": "1", "file_path": "paper.pdf", "content": null}],
    "question": "What is MemGraphRAG?",
    "docs": ["…passage text…"],
    "doc_scores": [0.42]
  },
  "metadata": {"mode": "ppr"}
}
```

---

## 5. Streaming (`POST /query/stream`)

SSE events (LightRAG-compatible order):

1. `{"references":[…]}`
2. `{"response":"…"}` (full answer; token streaming not yet implemented)
3. `[DONE]`

```bash
curl -sS -N -X POST http://localhost:9621/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is MemGraphRAG?","mode":"ppr"}'
```

---

## 6. Ingest (minimal)

Wait until `GET /health` reports `"pipeline_busy": false`. Mutating document routes return **409** while busy.

```bash
# Upload a file
curl -sS -X POST http://localhost:9621/documents/upload \
  -F 'file=@./notes.pdf'

# Insert raw text
curl -sS -X POST http://localhost:9621/documents/text \
  -H 'Content-Type: application/json' \
  -d '{"text":"MemGraphRAG uses a three-layer memory graph.","file_source":"note.txt"}'

# List statuses
curl -sS http://localhost:9621/documents/
```

Admin delete / clear-all: see [MemGraphRAG-API-Server.md](MemGraphRAG-API-Server.md).

---

## 7. Health

```bash
curl -sS http://localhost:9621/health
```

---

## See also

- [MemGraphRAG-API-Server.md](MemGraphRAG-API-Server.md) — deploy, auth env, storage, Ollama bridge
- [Clients.md](Clients.md) — CLI, Streamlit, HTTP client
- [Logging.md](Logging.md) — `[RETRIEVE]` / `[LLM]` log markers
- [ProgramingWithCore.md](ProgramingWithCore.md) — in-process Python core API
