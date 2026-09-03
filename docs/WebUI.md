# Web UI

A React chat interface served by the API itself. One process, one port, no CORS.

## Run it

```bash
docker compose up -d postgres-app     # chat persistence (host port 5433)
cp env.example .env                   # then set the LLM/embedding bindings
uv run memgraphrag-server             # http://localhost:9621/
```

The bundle lives in `memgraphrag/api/static/` and is **not committed** — it is a build
artifact. Without it the server logs `Web UI not built; serving API only` and every
API route keeps working. Build it with:

```bash
cd web && npm install && npm run build
```

`npm run dev` serves on :5173 and proxies the API routes to :9621, so the client
never needs a base URL in either mode.

The Docker image builds the bundle in its own `node` stage; nothing extra is needed
for `docker compose up`.

## What the screens map to

| Screen | Endpoints |
|---|---|
| Conversation | `POST /query/stream` (answer), `POST /chat/threads/{id}/messages` (persist) |
| Citation → library | client-side: a citation carries `chunk_id` + `source_path`, the panel opens on that file and scrolls to the passage |
| Sidebar thread list | `GET /chat/threads`, grouped client-side by age |
| Provider + model picker | `GET /models`, sent back as `provider` / `model` on the query |
| Suggestion cards | `GET /graph/highlights` — derived from the corpus, not hardcoded |
| Settings panel | `GET /query/params` — the form is generated from the registry |
| Library | `GET /library/tree`, `/library/file`, `/library/preview`, `/library/passages` |
| Cypher console | `POST /graph/cypher`, `GET /graph/schema`, `GET /graph/neighborhood` |
| Login | `POST /login` (form-encoded), Bearer token in `localStorage` |

## Provider routing

A request names a provider **id**; the server resolves the credential from its own
environment, so a browser never carries a key. `together`, `ollama`, `openai`,
`vllm` and `default` (the binding the server started with) are registered in
`memgraphrag/llm/providers.py`. Each takes an optional `<ID>_MODELS` allow-list;
when none is pinned the endpoint's own `GET /v1/models` catalogue is used instead,
filtered to chat-capable entries — Together AI advertises 278 models on one
endpoint, embeddings and image generators included.

Two things this deliberately does not do. **Embeddings are never routed**: the
corpus is indexed with one model at one dimension, and sending query embeddings
elsewhere returns vectors from a different space. And the catalogue is read with a
raw HTTP GET rather than `client.models.list()`, because the OpenAI SDK insists on
the `{"object": "list", "data": [...]}` envelope while Together answers with a bare
JSON array and raises while parsing a perfectly good 200.

## Cypher console

Read-only, enforced in three layers because any one of them alone has a hole:

1. the graph backend must be `Neo4JStorage` (the default is `IgraphStorage`, which
   has no session to open);
2. write keywords are rejected **after** string literals and comments are stripped,
   so `WHERE n.content CONTAINS 'DELETE the invoice'` is not a false positive and
   `n.created_at` is not mistaken for `CREATE`;
3. execution runs inside `default_access_mode="READ"` — Neo4j itself refuses to
   write from that transaction, which is the only layer a parser bypass cannot beat.

A `LIMIT` is injected when the statement has none, and every query is scoped to the
workspace label. That last part is not cosmetic: this Neo4j instance is shared with
a LightRAG deployment whose 14 556 nodes live under a different label, and an
unscoped query renders two unrelated knowledge graphs at once.

## Three things worth knowing

**Streaming is real, retrieval is not streamed.** `/query/stream` emits `references`
first, then one `response` frame per token, then `[DONE]`. Retrieval runs before the
first token because PPR has no partial result to emit — on the reference corpus that
is roughly 5 s to the references frame and 11 s to the first token. The composer
shows a "Récupération en cours…" indicator for exactly that window. The wire format
is unchanged, so `MemGraphRAGClient.query_stream` and the Streamlit playground keep
working; they simply receive many small frames instead of one.

**Chat lives in its own database.** `APP_DATABASE_URL` points at the `postgres-app`
service, deliberately separate from the RAG's PostgreSQL: conversations are product
data, not knowledge, and wiping a corpus must not take them along. With the variable
unset, `/chat/*` answers **503** and the UI keeps conversations in the browser tab
only — it degrades, it does not break.

**Model selection is allow-listed.** `LLM_MODELS` names what a caller may pick;
`LLM_MODEL` is always included. Anything else is refused with 400 rather than
forwarded, so a typo cannot bill a model nobody sanctioned.

## Clickable citations

A citation is one retrieved passage, not one document. `references[]` is numbered
exactly as `fence_passages` numbered the prompt, so the `[3]` the model wrote is
`reference_id` 3 — they used to disagree, because references were collapsed per
document and ten passages from three files produced three references.

Each entry carries `chunk_id` and, when doc-status knew one, `source_path`. Clicking
a pill opens the library on that document and scrolls to the passage. The pills are
grouped per document in the UI (`[1][4] paper.pdf`) so ten passages do not render as
ten near-identical chips.

Two ways the jump can miss, both stated rather than shown as an empty panel:
`LIBRARY_ROOT` and `INPUT_DIR` are independent settings, so a cited file need not be
under the library root; and `/library/passages` returns at most 200 passages with no
offset, so a chunk cited from a long document can fall outside the window.

## Agent mode

`mode=agent` gives the model a `retrieve` tool and lets it decide what to search
for. It is the only mode where a follow-up works: everywhere else the retrieval
query is the user's literal text, so "et le second ?" searches the corpus for that
phrase. The loop reads the history and writes a standalone search string instead.

The UI shows each step as it happens — a forty-second turn with no visible progress
is indistinguishable from a hang — and merges the references from every hop.
Overwriting them, which is what the earlier code did, kept only the last hop's
sources while the answer went on citing all of them.

The first turn is forced with `tool_choice`, which doubles as the capability check:
a model that answers a *forced* tool call in prose cannot call tools, and the mode
refuses it by name rather than degrading into an ungrounded answer.

**Deciding turns are capped, answering turns are not.** A deciding turn's only
product is a tool call or its absence — its prose is discarded — but left unbounded
one spent **37 s generating 4 606 characters** of reasoning nobody reads. Capped at
`AGENT_DECIDE_MAX_TOKENS` (256) the same turn answers in ~2.5 s, taking a full agent
turn from ~51 s to ~13 s on the reference corpus. The forced opening call still
emits its tool call at every cap tested down to 64, and is itself faster capped.

The trade-off is stated rather than hidden: if the cap cuts a turn off before it can
ask for a second search, the loop reads that as "ready to answer" and still answers
from the passages it has. That is a lost hop, not a failure, so the server says so
once per process and puts `finish_reason` on every `memgraphrag.agent.think` span.
On `openai/gpt-oss-20b` it costs nothing measurable: uncapped, that turn produced no
tool call in 46 s even when the retrieved passages plainly did not answer the
question.

## Limits inherited from the API

- **Citations carry a filename, not a snippet.** `references[].content` is always
  `null`. Passage text is reachable through `POST /query/data` or
  `GET /documents/{id}/chunks`, not through a citation.
- **Provenance depends on doc-status.** `_passage_id_to_source` is built
  exclusively from doc-status records (`file_path` + `chunk_ids`), so a corpus
  ingested by a script that calls `core.ainsert()` directly cites `"unknown"` and
  shows an empty library — the two are one missing table, not two bugs. It is
  recoverable without re-ingesting, because a chunk id is a content hash: re-running
  the same chunking over the same sidecars reproduces the same ids.
  `scripts/backfill_rfe_sources.py` does that for one specific corpus — it imports
  `build_chunks` from `scripts/ingest_rfe.py` and works only there, so treat it as a
  worked example rather than a general tool. Whatever repairs provenance must verify
  the overlap against the graph before writing: a misaligned backfill attaches the
  wrong filename to a passage, which is worse than no citation at all.
- **Conversation history is not used for retrieval, except in agent mode.** It
  reaches the LLM everywhere, but the PPR query is the literal question. The UI caps
  history at 12 turns; the server neither validates nor caps it.
- **`mode=bypass` now passes history on both paths.** It used to reach the model
  when streaming and not when buffered, so the same mode remembered the conversation
  over SSE and forgot it over JSON.
- **Ingestion has no push channel.** The library polls `GET /documents/` every four
  seconds while anything is pending, parsing or processing, and stops when nothing is
  in flight.
- **A document ingested by a script that bypasses the pipeline has no doc-status
  record**, so it never appears in the library even though its passages are in the
  graph. That is a property of the ingestion path, not of this UI.
- **Static assets are public.** A `StaticFiles` mount is not covered by the per-route
  auth dependency — there is no global auth middleware. The SPA shell is not a
  secret and every call it makes is still authenticated, but do not put anything
  sensitive in the bundle.

## Design provenance

The layout follows a Figma draft that is a **flattened PNG** — one rectangle with an
image fill, no layers, no components, no variables. The palette in
`web/tailwind.config.js` was therefore sampled from the exported pixels rather than
read from Figma variables: surfaces and the violet steps 50–400 are measured, and
steps 500–700 are extrapolated on the same hue (~258°) because the export contains no
flat region of them. Spacing is eyeballed from a 912 px-wide render. Treat those
numbers as a starting point, not as a spec.
