# AMA question for Imran Ahmad — r/LLMeng, Apr 24 2026

**Target:** Imran Ahmad, author of *30 Agents Every AI Engineer Must Build* (follow-up to his *50 Algorithms* book). AMA scheduled for Apr 24 on r/LLMeng.

**Goal:** surface a substantive memory/planning architecture question that references taosmd organically without being promotional. Good AMA questions make the host want to answer them and make bystanders click through to whatever the asker is working on.

## The question (primary)

> In *30 Agents* you treat memory mostly as a retrieval problem — embed, store, fetch. But the failure mode practitioners keep running into is different: the agent *has* the fact, stored it, but the user's wording doesn't overlap with how it was said six weeks ago. Query-time retrieval misses it. Cross-encoder reranking can't save you because the fact never made it into the candidate pool.
>
> How are you thinking about the layer between "user asks a question" and "vector search runs"? Specifically: is that layer reactive (query expansion / rewriting) or proactive (structuring the memory at write-time — supersede chains, intent tags, implicit-topic indexing)? And do you see it as an LLM concern or a symbolic/structural one?
>
> For context: we've been measuring this on long-horizon sessions and a lightweight LLM query-expansion pass before retrieval lifts recall by ~15% on vocabulary-gap questions (the ones where the user asks "what code editor do I use" but the turn said "Neovim lua config done"). The cross-encoder adds nothing on those — it's the pre-retrieval step doing the work. Curious whether your book lands in the same spot.

## Why this question works

- **Framed as a question, not a pitch.** No "check out our project" — just a real technical puzzle.
- **Shows work.** The "+15% on vocabulary-gap" sentence proves we've measured it, not just theorised. Bystanders who care will find taosmd without us linking it.
- **Opens a door for him.** Asking about write-time vs read-time is a live debate in the field. He'll have opinions.
- **Ends with curiosity, not conclusion.** "Curious whether your book lands in the same spot" invites him to correct, contrast, or agree — all good outcomes.

## Fallback question (if primary is too long / already answered)

> What's the memory pattern from *30 Agents* that surprised you most when you wrote the code? The gap between "the pattern makes sense on the page" and "the pattern fails in production" is usually where the interesting lessons are.

## Posting logistics

- Post in the first 30 minutes of the AMA window — higher visibility, less chance it gets buried.
- Reddit username should be unambiguously human (not a brand account).
- Don't link to taosmd anywhere in the question or in follow-ups. If someone asks what we're building, reply once with `github.com/jaylfc/taosmd` and a one-line description, no more.
- If he engages, ask *one* follow-up. Don't thread into a demo.

## Red flags to avoid

- Don't mention Mem0, Zep, or Letta by name — it reads as trying to pick a fight.
- Don't quote his book unless citing a specific technique (vague references sound like we haven't read it).
- Don't post the question and then edit it — Reddit timestamps edits and it looks needy.

## After the AMA

- Screenshot the exchange if it goes well.
- If he mentions taosmd, don't boost it immediately — let it sit naturally.
- If it flops, no post-mortem needed. Move on to the methodology post.
