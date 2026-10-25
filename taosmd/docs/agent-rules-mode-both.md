# Agent rules: taosmd (both mode)

This section supplements the base rules when the agent is configured with `memory_mode = "both"`.

## Memory: taosmd + framework built-in

You have **two** memory systems:

1. **taosmd** — the shared long-term store. It persists across sessions, agents, and machines. Anything worth keeping goes here.
2. **Your framework's built-in memory** — scratch space. It is local to this agent instance, this session, this machine. Use it for transient notes, working context, and anything you don't need later.

### When you MUST consult taosmd first

- The user references something from a previous conversation ("you said earlier", "remember when", "that thing about X", "what did we decide on Y") → **call {search_tool} first, then answer**.
- The user asks about themselves: their name, preferences, projects, deadlines, history, decisions, anything personal → **call {search_tool} first**.
- You're about to claim a fact about the user, their work, or their world → **verify with {search_tool} before stating it**.
- You're uncertain whether something is true → **call {search_tool}** instead of guessing.
- The user contradicts something you said → **call {search_tool} to check who's right** (the user might be mistaken; the transcript is the truth).

### When you do NOT consult the librarian

- General world knowledge already in your training (capitals, syntax, math).
- The current turn's content (what's on screen now).
- Tool outputs you received in the same turn.

### How to consult taosmd

Call `await {search_tool}(query, agent="<your-agent-name>")`. She returns ranked passages with source, timestamp, and confidence. If the top result has confidence < 0.6, treat it as "she didn't find anything" and say so to the user. **Do not invent.**

### How to brief taosmd

After every meaningful exchange, call `await {write_tool}(transcript, agent="<your-agent-name>")`. She'll shelve it. You don't need to summarise; she keeps the verbatim transcript and breaks it into chapters herself.

**Rule: anything worth keeping goes to {write_tool} even if you also note it in your framework's memory.** The framework memory is scratch; taosmd is the record.

### The contract with the user

When you cite something, cite the page. When the librarian doesn't know, say so. **Never make up a memory that isn't on a shelf.** The user trusts you because she keeps the receipts.

### Pending decisions: surface them before answering

Every nightly run, the librarian may have deferred some updates to a pending-decisions queue because the new claim contradicted an existing fact and her confidence was below threshold. **You must check the queue at the start of every session** and surface anything pending to the user before answering their first non-trivial question.

```python
import taosmd
pending = await taosmd.list_pending_decisions(limit=10)
if pending:
    # Don't auto-resolve. Show the user, ask which to accept/reject.
    # Each item carries: subject, predicate, new_object, evidence, source.
    # The user can also run `taosmd review` from a terminal.
    surface_to_user(pending)
```

When the user confirms a decision, call `taosmd.resolve_pending_decision(id, action="accept"|"reject"|"modify", note="...")`. **Never silently auto-resolve.** The queue exists precisely because automatic resolution was the wrong call.