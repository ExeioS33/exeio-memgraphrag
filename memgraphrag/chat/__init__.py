"""Application-level chat data: threads and messages.

Separate from ``memgraphrag/storage`` on purpose — that package holds the RAG's
knowledge stores; this one holds product data and lives in its own database.
"""

from memgraphrag.chat.models import ChatMessage, ChatThread, derive_title
from memgraphrag.chat.store import (
    BaseChatStore,
    ChatStoreUnavailable,
    InMemoryChatStore,
    PostgresChatStore,
    create_chat_store,
)
from memgraphrag.chat.users import (
    AppUser,
    BaseUserStore,
    EmailTaken,
    InMemoryUserStore,
    PostgresUserStore,
    create_user_store,
)

__all__ = [
    "AppUser",
    "BaseChatStore",
    "BaseUserStore",
    "EmailTaken",
    "InMemoryUserStore",
    "PostgresUserStore",
    "create_user_store",
    "ChatMessage",
    "ChatStoreUnavailable",
    "ChatThread",
    "InMemoryChatStore",
    "PostgresChatStore",
    "create_chat_store",
    "derive_title",
]
