"""Centralized configuration defaults for MemGraphRAG.

Adapted from LightRAG ``lightrag/constants.py`` (chunk size, async limits,
Ollama emulation defaults, working/input dirs) with MemGraphRAG-native
retrieval defaults (PPR, linking, fact similarity) from the research engine
(``MemGraphRAG/code/src/utils/config_utils.py``).
"""

from __future__ import annotations

# Query / retrieval
TOP_K = 10
LINKING_TOP_K = 50
PASSAGE_NODE_WEIGHT = 0.05
DAMPING = 0.5
FACT_SIMILARITY_THRESHOLD = 0.6
SKIP_FACT_RERANK = True
PPR_ENGINE = "igraph"
SCHEMA_TOP_K = 5
SCHEMA_NODE_WEIGHT = 0.1

# Ontology / conflict construction
ONTOLOGY_BATCH_SIZE = 20
#: Chunks extracted between two OpenIE cache writes. A killed run keeps every
#: completed sub-batch and re-bills only the one in flight.
OPENIE_CHECKPOINT_SIZE = 64
ONTOLOGY_MIN_FREQUENCY = 2
# Safety valve for the ontology filter. A corpus small enough that almost every
# schema is seen once carries no frequency signal, and applying an absolute
# threshold there would deactivate nearly the whole fact layer. Above this fraction
# the filter is skipped and logged instead of silently emptying the index.
ONTOLOGY_MAX_DEACTIVATION_RATIO = 0.5
CONFLICT_ENABLED = True
CONFLICT_MAX_GROUPS = 50
# Minimum LLM self-reported confidence before a detected conflict is acted on.
# Resolution may DISCARD a fact, so an unguarded hallucinated conflict destroys
# correct knowledge. Aligned with the research implementation.
CONFLICT_MIN_CONFIDENCE = 0.85

# Language of extracted entities, relations, types and answers. "auto" keeps the
# model's own choice, which on a non-English corpus mixes languages and fragments the
# schema layer. Set to e.g. "French" for a French corpus.
MEMGRAPHRAG_LANGUAGE = "auto"

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP_SIZE = 100

# Server
HOST = "0.0.0.0"
PORT = 9621

# Request limits / anti-abuse
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MiB per uploaded document
LOGIN_MAX_ATTEMPTS = 10
LOGIN_WINDOW_SECONDS = 60.0

# Ollama API emulation
DEFAULT_OLLAMA_MODEL_NAME = "memgraphrag"
DEFAULT_OLLAMA_MODEL_TAG = "latest"

# Async / embeddings
MAX_ASYNC_LLM = 4
EMBEDDING_DIM = 1536

# Paths
WORKING_DIR = "./data/rag_storage"
INPUT_DIR = "./data/inputs"
