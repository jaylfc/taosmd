__version__ = "0.2.0"

from .knowledge_graph import TemporalKnowledgeGraph as KnowledgeGraph
from .vector_memory import VectorMemory
from .archive import ArchiveStore as Archive
from .memory_extractor import extract_facts_from_text
from .intent_classifier import classify_intent, get_search_strategy
from .context_assembler import ContextAssembler
from .browsing_history import BrowsingHistoryStore as BrowsingHistory

# v0.2.0 — retrieval & processing
from .retention import retention_score, classify_tier, score_and_tier, find_near_duplicates, composite_score
from .secret_filter import redact_secrets, contains_secrets, filter_text
from .query_expansion import expand_query_fast, expand_query_llm, extract_entities_regex
from .graph_expansion import expand_from_results, format_expanded_context
from .crystallize import CrystalStore
from .reflect import InsightStore
from .session_catalog import SessionCatalog
from .catalog_pipeline import CatalogPipeline
from .retrieval import retrieve
from .api import ingest, search
from .cross_encoder import CrossEncoderReranker
from .access_tracker import AccessTracker
from .preference_extractor import extract_preferences
from .temporal_boost import temporal_rerank

# Backend interface (for taOS and other platforms)
from .backend import MemoryBackend
from .taosmd_backend import TaOSmdBackend

# Agent registry — multi-agent isolation on a single taosmd install
from .agents import (
    AgentRegistry,
    AgentExistsError,
    AgentNotFoundError,
    InvalidAgentNameError,
    LIBRARIAN_TASKS,
    FANOUT_LEVELS,
    register_agent,
    list_agents,
    agent_exists,
    get_agent,
    delete_agent,
    ensure_agent,
    get_librarian,
    set_librarian,
    is_task_enabled,
    effective_fanout,
)
from . import prompts

from importlib.resources import files


def agent_rules() -> str:
    """Return the verbatim `docs/agent-rules.md` contract.

    Works from both editable and wheel installs.
    """
    return files("taosmd").joinpath("docs/agent-rules.md").read_text(encoding="utf-8")


__all__ = [
    "agent_rules",
    # Core memory layers
    "KnowledgeGraph",
    "VectorMemory",
    "Archive",
    "SessionCatalog",
    "CrystalStore",
    "BrowsingHistory",
    # Processing
    "extract_facts_from_text",
    "extract_preferences",
    "CatalogPipeline",
    # Retrieval
    "retrieve",
    "ingest",
    "search",
    "CrossEncoderReranker",
    "classify_intent",
    "get_search_strategy",
    "expand_query_fast",
    "expand_query_llm",
    "extract_entities_regex",
    "expand_from_results",
    "format_expanded_context",
    "temporal_rerank",
    # Context assembly
    "ContextAssembler",
    # Scoring & filtering
    "retention_score",
    "classify_tier",
    "score_and_tier",
    "find_near_duplicates",
    "composite_score",
    "AccessTracker",
    "redact_secrets",
    "contains_secrets",
    "filter_text",
    # Insights
    "InsightStore",
    # Backend interface
    "MemoryBackend",
    "TaOSmdBackend",
    # Agent registry
    "AgentRegistry",
    "AgentExistsError",
    "AgentNotFoundError",
    "InvalidAgentNameError",
    "LIBRARIAN_TASKS",
    "register_agent",
    "list_agents",
    "agent_exists",
    "get_agent",
    "delete_agent",
    "ensure_agent",
    "get_librarian",
    "set_librarian",
    "is_task_enabled",
    "effective_fanout",
    "FANOUT_LEVELS",
    # Librarian prompts
    "prompts",
]
