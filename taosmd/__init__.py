__version__ = "0.2.0"

from .knowledge_graph import TemporalKnowledgeGraph as KnowledgeGraph
from .vector_memory import VectorMemory
from .archive import ArchiveStore as Archive
from .memory_extractor import extract_facts_from_text
from .intent_classifier import classify_intent, get_search_strategy
from .context_assembler import ContextAssembler
from .browsing_history import BrowsingHistoryStore as BrowsingHistory

# v0.2.0 — new modules inspired by agentmemory
from .retention import retention_score, classify_tier, score_and_tier, find_near_duplicates
from .secret_filter import redact_secrets, contains_secrets, filter_text
from .query_expansion import expand_query_fast, expand_query_llm, extract_entities_regex
from .graph_expansion import expand_from_results, format_expanded_context
from .crystallize import CrystalStore
from .reflect import InsightStore
from .leases import LeaseManager
from .mesh_sync import MeshSync
from .session_catalog import SessionCatalog
from .catalog_pipeline import CatalogPipeline
from .retrieval import retrieve
from .cross_encoder import CrossEncoderReranker
from .backend import MemoryBackend
from .taosmd_backend import TaOSmdBackend
from .job_queue import JobQueue
from .resource_manager import ResourceManager

# Agent registry — multi-agent isolation on a single taosmd install
from .agents import (
    AgentRegistry,
    AgentExistsError,
    AgentNotFoundError,
    InvalidAgentNameError,
    register_agent,
    list_agents,
    agent_exists,
    get_agent,
    delete_agent,
    ensure_agent,
)

__all__ = [
    # Core memory stack
    "KnowledgeGraph",
    "VectorMemory",
    "Archive",
    "ContextAssembler",
    "BrowsingHistory",
    # Extraction & classification
    "extract_facts_from_text",
    "classify_intent",
    "get_search_strategy",
    # Retention & decay
    "retention_score",
    "classify_tier",
    "score_and_tier",
    "find_near_duplicates",
    # Security
    "redact_secrets",
    "contains_secrets",
    "filter_text",
    # Query expansion
    "expand_query_fast",
    "expand_query_llm",
    "extract_entities_regex",
    # Graph expansion
    "expand_from_results",
    "format_expanded_context",
    # Session crystallization
    "CrystalStore",
    # Cross-memory insights
    "InsightStore",
    # Multi-agent coordination
    "LeaseManager",
    # Worker mesh sync
    "MeshSync",
    # Session catalog (timeline directory)
    "SessionCatalog",
    "CatalogPipeline",
    # Retrieval pipeline
    "retrieve",
    "CrossEncoderReranker",
    # Backend interface
    "MemoryBackend",
    "TaOSmdBackend",
    # Job queue + resource management
    "JobQueue",
    "ResourceManager",
    # Agent registry
    "AgentRegistry",
    "AgentExistsError",
    "AgentNotFoundError",
    "InvalidAgentNameError",
    "register_agent",
    "list_agents",
    "agent_exists",
    "get_agent",
    "delete_agent",
    "ensure_agent",
]
