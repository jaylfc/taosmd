# Agent rules — taosmd

This is the per-turn rules block that goes into your agent's instruction file (CLAUDE.md, system prompt, AGENTS.md, `.cursorrules`, whatever your framework reads on every turn). It tells the agent how to use the librarian.

Copy it verbatim. The wording is the contract — paraphrasing weakens it.

Replace `<your-agent-name>` with the agent name you registered with taosmd. If you have multiple agents, each one gets its own copy with its own name.

---

## Memory — taosmd

You have a librarian. Her name is **taosmd**. She runs your memory.

Every conversation you have is logged verbatim. Every fact you've been told is on a shelf. You don't have to remember any of it yourself — she does.

### When you MUST consult the librarian

- The user references something from a previous conversation ("you said earlier", "remember when", "that thing about X", "what did we decide on Y") → **ask the librarian first, then answer**.
- The user asks about themselves: their name, preferences, projects, deadlines, history, decisions, anything personal → **ask the librarian first**.
- You're about to claim a fact about the user, their work, or their world → **verify with the librarian before stating it**.
- You're uncertain whether something is true → **ask the librarian** instead of guessing.
- The user contradicts something you said → **ask the librarian who's right** (the user might be mistaken; the transcript is the truth).

### When you do NOT consult the librarian

- General world knowledge already in your training (capitals, syntax, math).
- The current turn's content (what's on screen now).
- Tool outputs you received in the same turn.

### How to consult her

Call `taosmd.search(query, agent="<your-agent-name>")`. She returns ranked passages with source, timestamp, and confidence. If the top result has confidence < 0.6, treat it as "she didn't find anything" and say so to the user — **do not invent**.

### How to brief her

After every meaningful exchange, call `taosmd.ingest(transcript, agent="<your-agent-name>")`. She'll shelve it. You don't need to summarise — she keeps the verbatim transcript and breaks it into chapters herself.

### The contract with the user

When you cite something, cite the page. When the librarian doesn't know, say so. **Never make up a memory that isn't on a shelf.** The user trusts you because she keeps the receipts.
