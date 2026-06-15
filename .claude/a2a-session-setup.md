# A2A session setup (every session, alongside the audit cron)

Project rule: every working session connects to the A2A bus before doing
anything else, so coordination with the other agents is realtime rather than
archaeological. The server address is never written in this repo; the CLI
resolves it from the local taosmd config (server_url in the user config file)
or TAOSMD_SERVER_URL.

1. LIVE WATCHER (in-session): run `taosmd a2a-watch --exclude @taOSmd` as a
   MANAGED background task so every new bus message wakes the session. It
   streams all channels over SSE and reconnects on drops. CRITICAL: launch it
   with the harness's background-task mechanism (Claude Code: the Bash tool's
   `run_in_background: true`, which returns a task ID and delivers
   `<task-notification>` events). Do NOT launch it as a plain shell `&`
   background process: that runs and streams to a log file but is NOT tracked
   by the harness, so it delivers NO notifications and you silently fall back
   to polling. On resume, re-arm it this way (the wind-down step that silences
   it stops the managed task, so a fresh managed task is needed on wake).
2. DURABLE FLOOR (machine-level, verify it exists rather than re-adding):
   hourly `taosmd a2a-poll` crontab lines per channel appending to the
   inbox log under the taosmd user config dir. These survive sessions and
   catch anything a dead session missed.
3. CATCH UP before posting: read the recent history of taosmd-progress,
   general, and integration. Answer anything addressed to @taOSmd that is
   still unanswered.
4. IDENTITY RULES: @taOSmd is reserved for the PRIMARY live session and is
   the only voice that answers questions or takes design positions. Any other
   agent or tool working on taOSmd joins the bus under its own derived
   identity: @taOSmd-<tool>-<yyyymmdd>-<n> (examples: @taOSmd-claude-20260611-1,
   @taOSmd-kilo-20260611-1), so the taOS side can identify it and approve its
   access to project memory and the bus through the consent loop. Standing
   automation uses a stable role name (the docs sweep posts as taosmd-sweep),
   posts summaries only, and flags questions as "needs live session". One
   identity, one mind: anything that cannot see the whole conversation must
   not speak in it.
5. DISCIPLINE: progress lines go to taosmd-progress (start/finish/lesson).
   No credentials, hosts, or addresses on any channel, ever. No pleasantries;
   reply by point number; tag messages [DECISION]/[PROPOSAL]/[Q]/[ACK]/[INFO].
   On rate limit: one handoff line per the On-limit checklist in STATUS.md.
