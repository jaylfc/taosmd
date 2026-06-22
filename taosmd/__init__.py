__version__ = "0.4.0"

from .knowledge_graph import TemporalKnowledgeGraph as KnowledgeGraph
from .vector_memory import VectorMemory
from .archive import ArchiveStore as Archive
from .memory_extractor import extract_facts_from_text
from .intent_classifier import classify_intent, get_search_strategy
from .context_assembler import ContextAssembler
from .browsing_history import BrowsingHistoryStore as BrowsingHistory

# v0.2.0: retrieval & processing
from .retention import retention_score, classify_tier, score_and_tier, find_near_duplicates, composite_score
from .secret_filter import redact_secrets, contains_secrets, filter_text
from .query_expansion import expand_query_fast, expand_query_llm, extract_entities_regex
from .graph_expansion import expand_from_results, format_expanded_context
from .crystallize import CrystalStore
from .reflect import InsightStore
from .pending_decisions import PendingDecisionsStore
from . import predicate_vocab
from . import emem_event_lift
from . import loaders
from .session_catalog import SessionCatalog
from .catalog_pipeline import CatalogPipeline
from .retrieval import retrieve
from .api import (
    ingest,
    ingest_batch,
    search,
    list_projects,
    list_shelves,
    list_pending_decisions,
    resolve_pending_decision,
)

# Project identity: git-remote-derived fingerprint for cross-agent memory
# sharing (the project= / also_include= args on ingest/search).
from . import project
from .project import get_project_id, ProjectResolver, ProjectInfo

# Activation surfaces: shared service layer + local HTTP/REST API (#85).
# The MCP server (#84) reuses the same `service` core.
from . import service
from .http_server import serve

# MCP server (#84). The module imports the `mcp` SDK lazily, so this import
# never fails when the optional dependency is absent; only building/running a
# server raises MissingMCPDependencyError. Importing `taosmd` stays lean.
from . import mcp_server
from .cross_encoder import CrossEncoderReranker
from .access_tracker import AccessTracker
from .preference_extractor import extract_preferences
from .temporal_boost import temporal_rerank

# Backend interface (for taOS and other platforms)
from .backend import MemoryBackend
from .taosmd_backend import TaOSmdBackend

# Agent registry: multi-agent isolation on a single taosmd install
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

# Install-wide config: the memory/Librarian model is a global setting.
from . import config
from .config import get_memory_model, set_memory_model

# Recipes: first-class named config bundles (schema, registry, recommend,
# resolve, apply). See docs/superpowers/specs/2026-06-08-recipes-design.md.
from .recipes import (
    recipe_schema,
    list_recipes,
    get_recipe,
    recommend,
    resolve_recipe,
    apply_recipe,
    Recipe,
    local_probe,
)

from importlib.resources import files


def agent_rules() -> str:
    """Return the verbatim `docs/agent-rules.md` contract.

    Works from both editable and wheel installs.
    """
    return files("taosmd").joinpath("docs/agent-rules.md").read_text(encoding="utf-8")


def a2a_setup_guide() -> str:
    """Return the verbatim `docs/a2a-comms.md` setup guide.

    The guide is addressed to an agent and covers installing taOSmd, starting
    the A2A bus, joining a channel, and inviting other agents. Works from both
    editable and wheel installs.
    """
    return files("taosmd").joinpath("docs/a2a-comms.md").read_text(encoding="utf-8")


__all__ = [
    "agent_rules",
    "a2a_setup_guide",
    # Core memory layers
    "KnowledgeGraph",
    "VectorMemory",
    "Archive",
    "SessionCatalog",
    "CrystalStore",
    "PendingDecisionsStore",
    "BrowsingHistory",
    # Processing
    "extract_facts_from_text",
    "extract_preferences",
    "CatalogPipeline",
    # Retrieval
    "retrieve",
    "ingest",
    "ingest_batch",
    "search",
    # Project identity + cross-agent discovery
    "project",
    "get_project_id",
    "ProjectResolver",
    "ProjectInfo",
    "list_projects",
    "list_shelves",
    # Activation surfaces
    "service",
    "serve",
    "mcp_server",
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
    # Global config: system-wide memory model
    "get_memory_model",
    "set_memory_model",
    # Recipes: named config bundles
    "recipe_schema",
    "list_recipes",
    "get_recipe",
    "recommend",
    "resolve_recipe",
    "apply_recipe",
    "Recipe",
    "local_probe",
    # Librarian prompts
    "prompts",
]
