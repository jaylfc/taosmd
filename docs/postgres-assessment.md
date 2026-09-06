# Postgres Assessment for taOSmd

## Executive Summary

This document inventories all SQLite touchpoints in taOSmd and designs a zero-loss migration path for existing installations. The assessment identifies the scope and complexity of migrating from SQLite to PostgreSQL while maintaining zero data loss and preserving all existing functionality.

## 1. SQLite Touchpoint Inventory

### 1.1 Database Files and Stores

Based on comprehensive analysis of the codebase, here are all SQLite databases currently used by taOSmd:

**Core Memory Storage (Shared/Multi-user Stores)**:
- `knowledge-graph.db` - Temporal knowledge graph for entity-relationship triples with temporal validity windows
- `archive-index.db` - SQLite index for zero-loss archive files (JSONL events)

**Infrastructure & Tracking**:
- `access-tracker.db` - Access tracking for query diversity and composite scoring
- `session-catalog.db` - Session catalogue with temporal classification and agent-specific taxonomy
- `crystals.db` - Session digests and lessons learned for zero-loss preservation

**Application-Specific Stores**:
- `claims.db` - Verifiable claims store backed by archive spans
- `browsing-history.db` - Lightweight browsing history for platform apps
- `insights.db` - Synthesized insights from KG cluster reflection
- `a2a-mentions.db` - A2A mention index keyed by handles
- `a2a-receipts.db` - A2A read receipts store tracking message delivery/reading
- `tasks.db` - Task management with dependency graphs and blocking edges
- `collections.db` - Collections for indexed content from folders

**Configuration & Settings**:
- `memory-settings.db` - Backend configuration and agent memory settings

Note: `pending-decisions` is not a standalone database. It is the `kg_pending_decisions` table inside `knowledge-graph.db`, created at `taosmd/pending_decisions.py:43`.

### 1.2 SQLite Connection Patterns

All databases use standardized connection patterns via `taosmd._db.connect()`, which:
- Enables Write-Ahead Logging (WAL) mode for concurrency
- Sets 5000ms busy timeout to prevent immediate "database is locked" errors
- Provides thread-safe connections with `check_same_thread=False` where needed

**Command used for inventory**:
```bash
grep -rh '"[^"]*\.db"' taosmd --include="*.py" | grep -oP '[a-z0-9_-]+\.db' | sort -u
```

### 1.3 SQLite-Specific SQL Features

**FTS5 (Full-Text Search)**:
- **Location**: `archive.py:75-81` (archive_fts virtual table)
- **Usage**: Full-text search across archived summaries and content
- **Key SQL**: `CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(summary, content, content_rowid='id', tokenize='porter unicode61')`

**Temporal Validity Windows**:
- **Location**: `knowledge_graph.py:40` (kg_triples table), `vector_memory.py:35` (vector_memory table)
- **Usage**: Point-in-time queries using `valid_from`/`valid_to` columns
- **Key SQL**: `CREATE INDEX IF NOT EXISTS idx_triples_valid ON kg_triples(valid_from, valid_to)`

**AUTOINCREMENT**:
- **Location**: `vector_memory.py:30`, `session_catalog.py:45,63`, `mentions.py:34`, `claims/store.py:22`, `archive.py:54`
- **Usage**: Primary keys for tables where explicit control over ID generation is needed

**JSON Storage**:
- **Location**: `knowledge_graph.py:30` (properties_json), `vector_memory.py:33` (metadata_json), `archive.py:63` (data_json)
- **Usage**: Structured metadata storage with JSON1 functions
- **Key SQL**: `json_extract(metadata_json, '$.agent')`

**WITHOUT ROWID**:
- **Note**: Not currently used in schema, but indexed structures exist for optimization

**LIMIT Semantics**:
- **Critical Issue**: Negative LIMIT values mean UNLIMITED in SQLite (defect identified in tsk-2hnss2)
- **Location**: `taosmd/archive.py:380` (`LIMIT ? OFFSET ?` receives the caller-supplied limit without a non-negative guard)

### 1.4 Migration Feasibility Assessment

Based on the features identified:

| Database | Postgres Suitability | Rationale |
|----------|---------------------|-----------|
| `knowledge-graph.db` | **REQUIRED** | Shared temporal knowledge graph needs concurrent multi-user access |
| `archive-index.db` | **REQUIRED** | Shared index for zero-loss archive, needs concurrent access |
| `access-tracker.db` | **REQUIRED** | Shared composite scoring, query diversity tracking |
| `session-catalog.db` | **REQUIRED** | Shared session tracking, multi-agent coordination |
| `crystals.db` | **REQUIRED** | Shared session digests, lessons learned |
| `claims.db` | **REQUIRED** | Shared verifications, cross-agent claims |
| `insights.db` | **REQUIRED** | Shared synthesized insights from KG reflection, initialized for every default install at `taosmd/auto_setup.py:122-124` |
| `a2a-mentions.db` | **REQUIRED** | Shared A2A mention indexing |
| `a2a-receipts.db` | **REQUIRED** | Shared A2A read receipts |
| `tasks.db` | **REQUIRED** | Shared task management, dependency graphs |
| `collections.db` | **REQUIRED** | Shared collections, cross-agent indexing |
| `memory-settings.db` | **REQUIRED** | Shared configuration, agent settings |
| `browsing-history.db` | **OPTIONAL** | User activity tracking, local preference |
| `vector-memory.db` | **OPTIONAL** | Local-first vector storage, offline access invariant |

## 2. Detailed Analysis by Store Type

### 2.1 Shared/Multi-User Stores (REQUIRED Migration)

These databases contain data that must be shared across multiple agents/users:

**knowledge-graph.db**:
- **Schema**: `kg_entities` (id, name, type, properties_json, created_at) and `kg_triples` (id, subject_id, predicate, object_id, valid_from, valid_to, confidence, source, source_ids, superseded_by, appeared_count, accessed_count, last_accessed_at, created_at) plus `kg_pending_decisions` table
- **Critical Features**: Temporal validity windows, foreign key constraints, composite indexes, pending decisions queue
- **Migration Complexity**: HIGH - temporal queries, soft deletes, and temporal reasoning are complex in Postgres
- **Evidence**: `taosmd/knowledge_graph.py:40` (kg_triples table), `taosmd/knowledge_graph.py:112` (connection), `taosmd/pending_decisions.py:43` (kg_pending_decisions table)

**archive-index.db**:
- **Schema**: `archive_index` (id, timestamp, event_type, agent_name, app_id, project, summary, file_path, line_number, data_json) + FTS5 virtual table
- **Critical Features**: FTS5 search, append-only design, zero-loss guarantees
- **Migration Complexity**: MEDIUM - FTS5 has no direct Postgres equivalent, requires alternative search strategy
- **Evidence**: `taosmd/archive.py:75-81` (archive_fts virtual table), `taosmd/archive.py:119` (connection)

**access-tracker.db**:
- **Schema**: Access tracking tables for query diversity and composite scoring
- **Critical Features**: Shared composite scoring, query diversity tracking
- **Migration Complexity**: MEDIUM - straightforward relational schema but requires concurrent access
- **Evidence**: `taosmd/access_tracker.py:48` (db_path default), `taosmd/access_tracker.py:54` (connection)

**session-catalog.db**:
- **Schema**: `sessions` (id, path, timestamp, agent_name, project, topic, subtopic, primary_project, primary_topic, primary_subtopic, labels_json, classified_at)
- **Critical Features**: Multi-column indexing, unique constraints
- **Migration Complexity**: LOW - straightforward tabular structure
- **Evidence**: `taosmd/session_catalog.py:135` (db_path default), `taosmd/session_catalog.py:146` (connection)

**crystals.db**:
- **Schema**: Session digests and lessons learned tables
- **Critical Features**: Shared session digests, lessons learned
- **Migration Complexity**: LOW - straightforward tabular structure
- **Evidence**: `taosmd/crystallize.py:66` (db_path default), `taosmd/crystallize.py:72` (connection)

**claims.db**:
- **Schema**: Verifiable claims store backed by archive spans
- **Critical Features**: Shared verifications, cross-agent claims
- **Migration Complexity**: MEDIUM - archive span references require careful migration
- **Evidence**: `taosmd/claims/store.py:36` (db_path default), `taosmd/claims/store.py:42` (connection)

**insights.db**:
- **Schema**: `insights` (id, cluster_label, insight_text, confidence, created_at)
- **Critical Features**: Synthesized insights from KG cluster reflection, confidence scores
- **Migration Complexity**: LOW - single table with simple schema
- **Evidence**: `taosmd/reflect.py:130` (class InsightStore), `taosmd/reflect.py:133` (db_path default), `taosmd/reflect.py:139` (connection), `taosmd/auto_setup.py:122-124` (initialized for every default install)

**a2a-mentions.db**:
- **Schema**: A2A mention index keyed by handles
- **Critical Features**: Shared A2A mention indexing
- **Migration Complexity**: LOW - single table with simple schema
- **Evidence**: `taosmd/mentions.py:27` (db_path default), `taosmd/mentions.py:31` (connection), `taosmd/api.py:148` (initialization)

**a2a-receipts.db**:
- **Schema**: A2A read receipts store tracking message delivery/reading
- **Critical Features**: Shared A2A read receipts
- **Migration Complexity**: LOW - single table with simple schema
- **Evidence**: `taosmd/receipts.py:52` (db_path default), `taosmd/receipts.py:56` (connection), `taosmd/api.py:151` (initialization)

**tasks.db**:
- **Schema**: Task management with dependency graphs and blocking edges
- **Critical Features**: Dependency graphs, blocking edges, project scoping
- **Migration Complexity**: MEDIUM - graph structure and edge tables require careful migration
- **Evidence**: `taosmd/tasks.py:120` (db_path), `taosmd/tasks.py:121` (connection), `taosmd/tasks.py:288` (LIMIT ? query)

**collections.db**:
- **Schema**: Collections for indexed content from folders
- **Critical Features**: Cross-agent indexing, folder-based content
- **Migration Complexity**: MEDIUM - indexed content structures
- **Evidence**: `taosmd/collections.py:184` (connection), `taosmd/migrations.py:360` (schema)

**memory-settings.db**:
- **Schema**: Backend configuration and agent memory settings
- **Critical Features**: Shared configuration, agent settings
- **Migration Complexity**: LOW - simple key-value configuration tables
- **Evidence**: `taosmd/taosmd_backend.py:71` (db_path default), `taosmd/taosmd_backend.py:91` (connection)

### 2.2 Local-Only Stores (OPTIONAL Migration)

These databases can remain SQLite due to local-first design requirements:

**vector-memory.db**:
- **Schema**: `vector_memory` (id, agent, text, embedding, metadata_json, created_at) with late-interaction and dense storage modes
- **Critical Features**: Local-first semantic vectors, offline search capabilities, embedding model tied to store
- **Migration Complexity**: HIGH - embedding models and storage modes are tied to the local store format; a stored vector is meaningless to a query embedded by a different model
- **Evidence**: `taosmd/vector_memory.py:171` (db_path default), `taosmd/vector_memory.py:234` (connection), `taosmd/auto_setup.py:66` (initialized for every default install)

**browsing-history.db**:
- **Schema**: Lightweight browsing history for platform apps
- **Critical Features**: User-specific browsing history
- **Migration Complexity**: LOW - simple append-only history table
- **Evidence**: `taosmd/browsing_history.py:33` (db_path default), `taosmd/browsing_history.py:39` (connection), `taosmd/auto_setup.py:94` (initialized for every default install)

## 3. Zero-Loss Migration Path

### 3.1 Detection and Analysis

**Pre-Migration Detection**:
```bash
# Command to detect SQLite installs
grep -rh '"[^"]*\.db"' taosmd --include="*.py" | grep -oP '[a-z0-9_-]+\.db' | sort -u
```

**Database Health Check**:
- Verify database integrity with `PRAGMA integrity_check`
- Check schema versions with `PRAGMA user_version`
- Validate table structures and constraints
- Count rows in each table for baseline verification

### 3.2 Copy Strategy

**Two-Phase Copy Approach**:
1. **Phase 1 - Read-Only Copy**: Create identical Postgres tables while keeping original SQLite databases intact
2. **Phase 2 - Verification**: Validate data integrity through row counts and content hashes
3. **Phase 3 - Cutover**: Switch applications to Postgres after successful verification

**Data Copy Commands**:
```sql
-- Generic data copy pattern
INSERT INTO postgres.table_name SELECT * FROM sqlite.table_name;
```

### 3.3 Verification Protocol

**Row Count Verification**:
- Count rows in each source table before migration
- Count rows in each target table after migration
- Ensure 100% match across all tables

**Content Hash Verification**:
```sql
-- Generate content hash for each row
SELECT md5(COALESCE(column1||'', '') || COALESCE(column2||'', '')) as row_hash
FROM source_table
ORDER BY id;
```

**Foreign Key Validation**:
- Verify referential integrity constraints
- Check that all foreign keys resolve correctly

### 3.4 Rollback Strategy

**Atomic Rollback**:
1. Keep original SQLite databases intact throughout migration
2. Use database transactions for all Postgres operations
3. Maintain parallel read access during migration
4. Rapid rollback to SQLite on any failure

**Failure Recovery**:
- Automatic rollback on any verification failure
- Manual intervention only for catastrophic failures
- Restoration from verified SQLite backups

## 4. Configuration Migration

### 4.1 Data Directory Config

**Files Migrated**:
- `config.json` - Application settings and user preferences
- `server_url` and `server_token` - Authentication endpoints

**Migration Steps**:
1. Extract configuration from existing SQLite `memory-settings.db`
2. Migrate to Postgres `config` table
3. Update application startup scripts to use Postgres configuration

### 4.2 Postgres Configuration

**New Configuration Files**:
- `postgres.yml` - Database connection parameters
- `database.setup` - Migration and initialization scripts
- `database.maintenance` - Backup and maintenance procedures

## 5. Dual-Backend Recommendation

### 5.1 Hybrid Architecture Proposal

**Recommended Approach**: Dual-backend with gradual transition

**Implementation Phases**:
1. **Phase 1**: Parallel operation with SQLite as primary, Postgres as secondary
2. **Phase 2**: Gradual feature migration to Postgres (session catalog first)
3. **Phase 3**: Full transition to Postgres with SQLite as read-only backup

**Interface Design**:
- Unified API layer abstracts database differences
- Automatic failover between backends
- Configurable read/write splitting

### 5.2 Maintenance Cost Analysis

**SQLite Maintenance**:
- Backup and restore operations
- Schema migrations (complex due to temporal features)
- Performance tuning for concurrent access

**Postgres Maintenance**:
- Connection pool management
- Replication setup
- Regular backups and monitoring
- Query optimization

**Total Cost**: Higher than pure SQLite, justified by reliability gains for shared stores.

## 6. Critical Technical Challenges

### 6.1 FTS5 Migration

**Problem**: SQLite FTS5 has no direct Postgres equivalent
**Solutions**:
- Implement PostgreSQL full-text search with `to_tsvector()` and `tsvector` functions
- Use third-party extensions like `pg_search`
- Consider ElasticSearch for advanced search capabilities

### 6.2 Temporal Queries

**Problem**: Complex temporal validity window queries
**Solutions**:
- Use PostgreSQL's `daterange` and `tsrange` types
- Implement application-level temporal logic
- Consider TimescaleDB for temporal data

### 6.3 Concurrency Control

**Problem**: Different concurrency models between SQLite WAL and Postgres
**Solutions**:
- Implement proper transaction isolation levels
- Use connection pooling with retry logic
- Design read-heavy workloads for replication

## 7. Acceptance Criteria

### 7.1 Technical Requirements

- [ ] All existing SQLite databases migrated without data loss
- [ ] Zero-downtime cutover with parallel read access
- [ ] All functionality preserved (temporal queries, FTS, soft deletes)
- [ ] Performance maintained or improved
- [ ] All existing tests pass (1793 passed / 10 skipped / 7 errors baseline)

### 7.2 Operational Requirements

- [ ] Automated migration scripts with rollback capability
- [ ] Comprehensive monitoring and alerting
- [ ] Disaster recovery procedures
- [ ] Backup and restore procedures
- [ ] Performance baselines and monitoring

## 8. Implementation Timeline

Implementation is phased over multiple sprints, with core shared stores migrated first and local-only stores remaining on SQLite until the hybrid interface is proven.

## 9. Risk Assessment

**High Risk**:
- FTS5 migration complexity
- Temporal query preservation
- Large data volume migration

**Medium Risk**:
- Application downtime during cutover
- PostgreSQL configuration and tuning
- Testing completeness

**Low Risk**:
- Basic schema migration
- Data copy verification
- Rollback procedures

## 10. Conclusion

The migration from SQLite to PostgreSQL in taOSmd is technically feasible but complex due to:

1. **Advanced SQLite features**: FTS5, temporal validity windows, soft deletes
2. **Zero-loss requirements**: No data loss during migration
3. **Local-first invariants**: Some databases must remain local for offline access

**Recommendation**: Proceed with dual-backend hybrid approach, migrate critical shared stores first, and maintain SQLite as backup during transition. The zero-loss requirement can be satisfied with careful testing and rollback procedures.

**Next Steps**:
1. Technical review by database specialists
2. Prototype migration for small subset of data
3. Decision point for full implementation vs. alternative solutions

---

*Assessment completed: 2026-08-28*
*Repository: jaylfc/taosmd*
*Reference: note-260828-77c030*
