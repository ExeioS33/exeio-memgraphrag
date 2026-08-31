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
| Sidebar thread list | `GET /chat/threads`, grouped client-side by age |
| Model picker | `GET /models`, sent back as `model` on the query |
| Settings panel | `GET /query/params` — the form is generated from the registry |
| Library | `GET /documents/`, `GET /documents/{id}/chunks`, upload / text / scan / delete / requeue |
| Graph explorer | `GET /graph/label/list`, `GET /graphs` |
| Login | `POST /login` (form-encoded), Bearer token in `localStorage` |

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

## Limits inherited from the API

- **Citations carry a filename, not a snippet.** `references[].content` is always
  `null`. Passage text is reachable through `POST /query/data` or
  `GET /documents/{id}/chunks`, not through a citation.
- **Conversation history is not used for retrieval.** It reaches the LLM, but the
  PPR query is the literal question, so "et le second ?" retrieves against that text.
  The UI caps history at 12 turns; the server neither validates nor caps it.
- **`mode=bypass` ignores history entirely** (`core.py`, bypass branch).
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
