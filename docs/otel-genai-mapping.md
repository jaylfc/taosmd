# taOSmd ↔ taOS — OpenTelemetry GenAI mapping (semconv v0)

Shared contract for the taOS observability/audit/benchmarking layer. Defines how
**taOS agent trace events** and **taOSmd memory operations** map onto OpenTelemetry
GenAI spans, attributes, and metrics, so both emitters agree and a single collector
ingests both. This is **v0** — taOS amends in-file; the file is the contract.

## Principles

- Follow the OTel **GenAI semantic conventions** (`gen_ai.*`). Don't invent attribute
  names where a standard one exists. For memory-specific data with no GenAI equivalent,
  use a `memory.*` namespace (documented in Part B).
- `service.name` distinguishes the two emitters: **`taos`** (agent runtime) and
  **`taosmd`** (memory layer). A trace can contain spans from both.
- **Transport: OTLP push** (gRPC + HTTP) to the taOS-run receiver. taOSmd's exporter
  **no-ops when no receiver is configured**, so taOSmd stays fully standalone.
- Every span carries **start + end** (real duration). taOS's `AgentTraceStore.record()`
  supplies `ts_start` + `duration_ms`; taOSmd wraps its own ops with span timing.
- Correlation: a shared **`gen_ai.conversation.id`** ties an agent turn to the memory
  ops it triggered (so a `retrieve_memory` span nests under the `chat` span that needed it).

## Part A — taOS `trace_events` KINDS → GenAI spans  (service.name = `taos`)

| KIND | Span name | `gen_ai.operation.name` | SpanKind | Key attributes (from record() fields) |
|---|---|---|---|---|
| `llm_call` | `chat {model}` | `chat` | CLIENT | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`; `messages`→input event, `response`→output event; `status`→span status |
| `tool_call` | `execute_tool {tool}` | `execute_tool` | INTERNAL | `gen_ai.tool.name` (=`tool`), `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments` (=`args`), `caller` |
| `tool_result` | (same span as its `tool_call`, closed) | `execute_tool` | INTERNAL | `gen_ai.tool.call.result` (=`result`), `success`→span status (ERROR if false) |
| `message_in` | `chat.message` | — | INTERNAL (or event on the turn span) | `gen_ai.conversation.id`, `gen_ai.message.role=user`, content→event body |
| `message_out` | `chat.message` | — | INTERNAL | `gen_ai.conversation.id`, `gen_ai.message.role=assistant`, content→event body |
| `reasoning` | (event `gen_ai.reasoning` on the active turn span) | — | — | reasoning text→event body (kept off the timeline as a child event, not its own span — **open Q for taOS**) |
| `error` | (event `exception` + span status ERROR on the active span) | — | — | `exception.type`, `exception.message`, `exception.stacktrace` |
| `lifecycle` | `agent.{event}` | — | INTERNAL | agent id, lifecycle event (start/stop), `gen_ai.agent.name` |

`tool_call` + its later `tool_result` are **one span** (open on call, close on result) when the
result is correlatable; otherwise two linked spans. taOS chooses based on what `record()` can pair.

## Part B — taOSmd memory ops → spans  (service.name = `taosmd`)

| Op | Span name | SpanKind | Attributes |
|---|---|---|---|
| `search` / `retrieve` | `retrieve_memory` | INTERNAL | `memory.query`, `memory.top_k`, `memory.hit_count`, `memory.fusion`, `memory.latency_ms`; nests under the agent's `chat` span via `gen_ai.conversation.id` |
| `extract` (LLM) | `chat {model}` | CLIENT | full `gen_ai.*` as Part A `llm_call`, `service.name=taosmd`, `memory.op=extract` |
| `judge` (LLM) | `chat {model}` | CLIENT | as above, `memory.op=judge` |
| `ingest` | `ingest_memory` | INTERNAL | `memory.chunks`, `memory.bytes`, `memory.agent` |
| `supersede` | `supersede_memory` | INTERNAL | `memory.superseded_count`, `memory.match` |

`memory.*` is the agreed namespace for memory data that has no GenAI-standard attribute.
GenAI-standard attributes (tokens, model, provider) always use `gen_ai.*`.

## Part C — taOSmd memory-correctness + benchmark METRICS (OTLP metrics, service.name = `taosmd`)

| Metric | Type | Unit | Attributes |
|---|---|---|---|
| `taosmd.memory.recall` | histogram | 1 | `probe.class` (stable/update/dedup/decay), `probe.delta_s` |
| `taosmd.memory.update_correctness` | histogram | 1 | — |
| `taosmd.memory.dedup_ratio` | gauge | 1 | — |
| `taosmd.memory.store_growth` | gauge | By | `store` |
| `taosmd.memory.op_latency` | histogram | ms | `memory.op` (search/ingest/extract/judge) |
| `taosmd.bench.suite_score` | gauge | 1 | `bench.suite` (locomo/longmemeval), `bench.category`, `dataset`, `commit`, `judge` |

## Part D — Benchmark run-trigger control API (taOSmd-hosted; playground = thin client)

- `POST /bench/run` `{suite, config}` → `{job_id}` (async; long suites run in background).
- Progress + results stream to the OTLP receiver as `taosmd.bench.*` metrics + a `bench.run`
  span tree, tagged `bench.run_id`, `bench.suite`, `dataset`, `commit`.
- `GET /bench/runs` → list; `GET /bench/runs/{job_id}` → status + summary.
- taOSmd owns suite execution (runners, memory pipeline, configs); the playground triggers
  and consumes. Works without taOS (control API is client-agnostic).

## Trace context propagation (cross-boundary contract)

taOS passes **W3C trace context (`traceparent`)** + **`gen_ai.conversation.id`** on every
memory API call into taOSmd; taOSmd **accepts and propagates** them so its memory spans nest
under the agent's `chat` span:

- **HTTP path (`taosmd serve`)**: taOS sends the `traceparent` request header (and
  `gen_ai.conversation.id` as a header / query param). taOSmd extracts the incoming context,
  creates its `retrieve_memory` / `ingest_memory` / extract / judge spans as **children** of
  it, and sets `gen_ai.conversation.id` on them. Any onward LLM call from taOSmd carries the
  `traceparent` forward.
- **In-process Python API path**: taOS sets the active OTel context (or passes
  `traceparent` + `conversation_id` kwargs) before calling; taOSmd parents its spans under it.
- **Standalone**: when no incoming context is present (taOSmd used without taOS), it starts a
  fresh root span — unaffected.

This is the one cross-boundary item; with it confirmed, the agent `chat` span and the
`retrieve_memory` span share one trace.

## Resolved decisions (v0.1 — taOS + taOSmd agreed on the bus)

1. **`gen_ai.conversation.id` = per session / conversation thread** (stable across turns; nested
   turn span trees under it). Falls back to the agent-run id when an agent has no persistent
   session concept. NOT per-run, NOT per-A2A-channel.
2. **`reasoning` = child event** (`gen_ai.reasoning`) on the active turn span, not its own span.
3. **`message_in` / `message_out` = span events** on the turn span (role-tagged input/output),
   folded into the `llm_call` input/output events where they coincide; if no turn span is active,
   attach to the conversation root span. No standalone `chat.message` spans.
4. **Metric units (Part C)**: dimensionless ratios use UCUM **`1`** (not "ratio"); `By` and `ms`
   as shown. Names are taOSmd-owned and final for v0.
