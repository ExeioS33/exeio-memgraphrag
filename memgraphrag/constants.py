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

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP_SIZE = 100

# Server
HOST = "0.0.0.0"
PORT = 9621

# Ollama API emulation
DEFAULT_OLLAMA_MODEL_NAME = "memgraphrag"
DEFAULT_OLLAMA_MODEL_TAG = "latest"

# Async / embeddings
MAX_ASYNC_LLM = 4
EMBEDDING_DIM = 1536

# Paths
WORKING_DIR = "./data/rag_storage"
INPUT_DIR = "./data/inputs"
