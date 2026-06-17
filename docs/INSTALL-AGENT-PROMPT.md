# Install taOSmd with an agent

This is the hardware-agnostic install prompt. Paste it into a tool-capable agent
(Claude Code, Cowork, or any agent that can run shell commands). If you have
already cloned the repo and can run the CLI, prefer the machine-tailored version:
`taosmd setup-prompt` prints a prompt already filled in with your detected
hardware. Use this static version when the CLI is not yet available.

---

You are setting up taOSmd (a local-first AI memory system) on this machine.

1. Detect the hardware. Run `python -c "import json; from taosmd import recipes; print(json.dumps(recipes.local_probe()))"` if the package is importable; otherwise inspect CPU, RAM, and GPU (and VRAM) yourself. Classify the tier: gpu-12gb / gpu-8gb / gpu-4gb / pi-npu / cpu. If detection fails, assume the cpu tier.

2. Recommend a profile and confirm it with the user:
   - Minimal: fastest and lightest. Plain retrieval, no rerank, no self-verify, no claims gate. For weak hardware or speed-first users.
   - Quality: arctic-embed + cross-encoder rerank + answer self-verification. Best accuracy on capable hardware.
   - Integrity: Quality plus the verified-memory recall gate, with provenance and the audit surface. For auditable, zero-served-hallucination memory.
   Lean Minimal on cpu/pi hardware, Quality on GPU hardware, and Integrity if the user states an audit or compliance need. The user always confirms or changes it.

3. Apply the hardware tier automatically (match the model to the silicon, no consent needed) and report it.

4. For every quality or integrity switch the profile turns on, ASK first and state the cost before enabling. Never enable one silently:
   - Cross-encoder reranking (rerank): adds a reranker pass per query (latency + a model download).
   - Answer self-verification (self_verify): roughly doubles answer latency (a second LLM pass).
   - Verified-memory recall gate (prefer_verified): adds an entailment verify-pass at write time; eliminates served-hallucination at no measured accuracy cost.
   Arctic-embed is a free, same-latency upgrade applied without asking; just report it.

5. Run the install scripts (`scripts/install-server.sh` or `scripts/install-client.sh`, `.ps1` on Windows) with the resolved choices, and write the switch states into `~/.taosmd/config.json`.

6. Close by telling the user every choice stays changeable later in the web dashboard, and how to re-run this setup.

If the install scripts fail (for example a permissions problem), report the failure and point the user at the manual install docs rather than leaving a half-configured state.
