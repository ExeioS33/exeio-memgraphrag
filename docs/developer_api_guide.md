# MemGraphRAG — Query API examples (developer)

Short reference for calling the MemGraphRAG query API against a running server.

Verified on **2026-08-05** with:

| Role | Provider | Model |
|------|----------|-------|
| LLM (QA / OpenIE) | Together (OpenAI-compatible) | `openai/gpt-oss-20b` |
| Embedding | Together | `intfloat/multilingual-e5-large-instruct` (dim **1024**) |

Base URL used below: `http://127.0.0.1:9622`

> **Port note:** If LightRAG already occupies `9621` (`cf-lightrag-api`), start MemGraphRAG on another port, e.g. `PORT=9622 uv run memgraphrag-server`. Compose defaults to `9621` when that port is free.

---

## 0. Structured answer contract

`POST /query` asks the QA LLM for **JSON** by default (`structured_output=true`).
The system prompt lives in `memgraphrag/prompts/templates.py`
(`RAG_QA_STRUCTURED_SYSTEM` / `render_rag_qa_structured`).

Expected LLM object:

```json
{
  "thought": "<brief reasoning grounded in the passages and Source filenames>",
  "answer": "<answer that names Source filename(s) and uses [n] citations>",
  "citations": [1, 2],
  "sources": [{"passage": 1, "file_path": "paper.pdf"}],
  "confidence": "high"
}
```

Passages are labeled `[Passage N | Source: <filename>]`. The API **always**
returns a `references` array built from retrieved passage sources (even if the
LLM omits them). Set `"structured_output": false` for the legacy freeform
`Thought:` / `Answer:` prompt (references still included).

---

## 1. Authentication — `POST /login`

Auth is **optional**. When `AUTH_ACCOUNTS` is unset, `/login` returns a guest
JWT and most routes accept calls without a Bearer token (dev / loopback).
When `AUTH_ACCOUNTS=user:pass` + `TOKEN_SECRET` are set, obtain a JWT before
calling `/query`. Alternatively use `MEMGRAPHRAG_API_KEY` → header `X-API-Key`.

### Path / query parameters

| Kind | Name | Required | Description |
|------|------|----------|-------------|
| Path | — | — | None |
| Query string | — | — | None |

### Headers

| Header | Required | Value |
|--------|----------|-------|
| `Content-Type` | Yes | `application/x-www-form-urlencoded` |

### Body (form fields)

| Field | Required | Example |
|-------|----------|---------|
| `username` | Yes | `admin` |
| `password` | Yes | `change_me` |

### Example request

```bash
export API=http://127.0.0.1:9622

curl -s -X POST "$API/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=change_me"
```

### Success response — `200 OK` (auth enabled)

```json
{
  "access_token": "<JWT>",
  "token_type": "bearer",
  "auth_mode": "enabled",
  "core_version": "0.1.0",
  "api_version": "0.1.0"
}
```

### Success response — `200 OK` (auth disabled / guest)

```json
{
  "access_token": "<JWT>",
  "token_type": "bearer",
  "auth_mode": "disabled",
  "message": "Authentication is disabled. Using guest access.",
  "core_version": "0.1.0",
  "api_version": "0.1.0"
}
```

Capture the token for later calls:

```bash
export TOKEN=$(curl -s -X POST "$API/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=change_me" | jq -r .access_token)
```

When auth is disabled you can skip `Authorization` on `/query`. When enabled:

```bash
# JWT
curl ... -H "Authorization: Bearer $TOKEN"
# or API key
curl ... -H "X-API-Key: $MEMGRAPHRAG_API_KEY"
```

### Failure responses

**Invalid credentials — `401 Unauthorized`**

```json
{
  "detail": "Incorrect credentials"
}
```

**Missing credentials (protected route) — `401 Unauthorized`**

```json
{
  "detail": "No credentials provided. Please login."
}
```

---

## 2. RAG query — `POST /query`

Non-streaming retrieve + QA. Parameter `stream` in the body is ignored on this
endpoint (use `POST /query/stream` for SSE).

### Path / query parameters

| Kind | Name | Required | Description |
|------|------|----------|-------------|
| Path | — | — | None (`/query` is fixed) |
| Query string | — | — | None — all options are JSON body fields |

### Headers

| Header | Required | Value |
|--------|----------|-------|
| `Authorization` | When JWT auth enabled | `Bearer <access_token>` from `/login` |
| `X-API-Key` | When API-key auth enabled | Value of `MEMGRAPHRAG_API_KEY` |
| `Content-Type` | Yes | `application/json` |
| `Accept` | No | `application/json` (default) |

### JSON body fields

| Field | Type | Required | Default | Constraints / notes |
|-------|------|----------|---------|---------------------|
| `query` | string | **Yes** | — | Min length **1** |
| `mode` | string | No | `"ppr"` | One of: `ppr`, `naive`, `context`, `bypass` |
| `top_k` | int | No | server `TOP_K` | `>= 1` — passages returned / seeded |
| `linking_top_k` | int | No | server default | `>= 1` — fact linking candidates |
| `passage_node_weight` | float | No | `0.05` | PPR seed weight for passages |
| `damping` | float | No | `0.5` | Personalized PageRank damping |
| `fact_similarity_threshold` | float | No | `0.6` | Used when `skip_fact_rerank` |
| `skip_fact_rerank` | bool | No | `true` | Skip LLM fact filter |
| `schema_top_k` | int | No | `5` | Ontology schemas linked from query |
| `schema_node_weight` | float | No | server default | Schema seed weight |
| `only_need_context` | bool | No | `false` | If `true`, return retrieved docs only (no QA) |
| `structured_output` | bool | No | `true` | JSON QA (`answer`/`thought`/`citations`/`confidence`) |
| `conversation_history` | object[] | No | `null` | `[{"role":"user\|assistant","content":"..."}]` — LLM only |
| `user_prompt` | string | No | `null` | Extra instruction appended to the QA user prompt |
| `stream` | bool | No | ignored | Ignored on `/query`; use `/query/stream` |

#### Mode cheat-sheet

| `mode` | Behaviour |
|--------|-----------|
| `ppr` | Fact linking + hierarchical PPR + QA (**default**) |
| `naive` | Dense passage retrieval only + QA |
| `context` | Retrieval without QA (same as `only_need_context=true`) |
| `bypass` | LLM only — no KB retrieval |

Unlike LightRAG, MemGraphRAG does **not** expose `local` / `global` / `hybrid` / `mix`.

---

## 3. Worked example — structured success

### Request

```bash
curl -s -X POST "$API/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is MemGraphRAG and how does retrieval work?",
    "mode": "ppr",
    "top_k": 5,
    "structured_output": true
  }' | jq .
```

Equivalent raw HTTP:

```http
POST /query HTTP/1.1
Host: 127.0.0.1:9622
Content-Type: application/json

{
  "query": "What is MemGraphRAG and how does retrieval work?",
  "mode": "ppr",
  "top_k": 5,
  "structured_output": true
}
```

### Success response — `200 OK`

```json
{
  "question": "What is MemGraphRAG and how does retrieval work?",
  "response": "MemGraphRAG retrieves via PPR over a three-layer memory graph [1] (MemGraphRAG.pdf).",
  "answer": "MemGraphRAG retrieves via PPR over a three-layer memory graph [1] (MemGraphRAG.pdf).",
  "thought": "Passage 1 (MemGraphRAG.pdf) describes PPR retrieval.",
  "citations": [1],
  "confidence": "high",
  "structured": true,
  "sources": ["MemGraphRAG.pdf"],
  "references": [
    {
      "reference_id": "1",
      "file_path": "MemGraphRAG.pdf",
      "content": null
    }
  ],
  "docs": [
    "…passage text…"
  ],
  "doc_scores": [0.4123],
  "gold_answers": null,
  "gold_docs": null
}
```

#### Response fields

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | Echo of the user query |
| `response` | string \| null | Alias of `answer` (LightRAG-style client compatibility) |
| `answer` | string \| null | Clean answer text (parsed from LLM JSON; should name sources) |
| `thought` | string \| null | Brief reasoning from the structured LLM object |
| `citations` | int[] | 1-based passage indices supporting the answer |
| `confidence` | string \| null | `high` \| `medium` \| `low` |
| `structured` | bool | `true` when the LLM returned parseable JSON |
| `sources` | string[] | Document source label per retrieved passage |
| `references` | array | Always present — `reference_id` + `file_path` for retrieved docs |
| `docs` | string[] | Top retrieved passages (up to 5 in the payload) |
| `doc_scores` | float[] \| null | Corresponding scores (rounded, up to 5) |

When `structured` is `false`, `answer` still contains usable text (legacy
`Thought:`/`Answer:` or raw model output); `citations` is `[]`.

---

## 4. Failure / error examples

### `401 Unauthorized` — missing credentials (auth enabled)

```bash
curl -s -X POST "$API/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is MemGraphRAG?","mode":"ppr"}'
```

Example body:

```json
{
  "detail": "No credentials provided. Please login."
}
```

### `422 Unprocessable Entity` — validation (empty query)

```bash
curl -s -X POST "$API/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"","mode":"ppr"}'
```

Example body (FastAPI / Pydantic):

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "query"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

### `422 Unprocessable Entity` — invalid `mode`

```bash
curl -s -X POST "$API/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is MemGraphRAG?","mode":"hybrid"}'
```

Example body:

```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "mode"],
      "msg": "Input should be 'ppr', 'naive', 'context' or 'bypass'",
      "input": "hybrid"
    }
  ]
}
```

### `500 Internal Server Error` — query processing / LLM failure

```json
{
  "detail": "Internal Server Error"
}
```

Check API logs for `[FAIL] api.query` / provider errors. Prefer treating empty
`docs` + empty `answer` as a soft retrieval miss in clients.

---

## 5. All four modes (same question)

```bash
export Q='What is MemGraphRAG and how does retrieval work?'

for mode in ppr naive context bypass; do
  echo "=== mode=$mode ==="
  curl -s -X POST "$API/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$Q\",\"mode\":\"$mode\",\"top_k\":5,\"structured_output\":true}" \
    | jq '{mode: "'"$mode"'", answer: .answer[0:160], structured, citations, docs: (.docs|length)}'
done
```

---

## 6. Related endpoints (optional)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/query/stream` | SSE answer stream (structured fields in the first `data:` event) |
| `POST` | `/query/data` | Retrieval payload only (`docs` / `doc_scores`) — debug without QA LLM |
| `GET` | `/health` | Liveness + `core_version` / `api_version` / `pipeline_busy` |
| `POST` | `/documents/upload` | Upload a file into the ingest pipeline |
| `GET` | `/documents` | List document statuses |
| `GET` | `/graphs` | Explore the memory graph |
| `GET` | `/openapi.json` | Full schema |

Interactive docs: `$API/docs`

---

## 7. Source of truth

- Request/response models: `memgraphrag/api/routers/query.py` (`QueryRequest`, `QueryResponse`)
- Query modes / knobs: `memgraphrag/base.py` (`QueryParam`)
- Structured QA prompts: `memgraphrag/prompts/templates.py` (`RAG_QA_STRUCTURED_*`, `parse_structured_qa`)
- Engine path: `memgraphrag/core.py` (`aquery` / `arag_qa`)

---

*Primary smoke question for manager/dev demos after structured JSON QA became the default `/query` contract.*
