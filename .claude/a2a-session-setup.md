# A2A session setup (every session, alongside the audit cron)

Project rule: every working session connects to the A2A bus before doing
anything else, so coordination with the other agents is realtime rather than
archaeological. The server address is never written in this repo; the CLI
resolves it from the local taosmd config (server_url in the user config file)
or TAOSMD_SERVER_URL.

1. LIVE WATCHER (in-session): run `taosmd a2a-watch --exclude @taOSmd` as a
   persistent background monitor so every new bus message wakes the session.
   It streams all channels over SSE and reconnects on drops.
2. DURABLE FLOOR (machine-level, verify it exists rather than re-adding):
   hourly `taosmd a2a-poll` crontab lines per channel appending to the
   inbox log under the taosmd user config dir. These survive sessions and
   catch anything a dead session missed.
3. CATCH UP before posting: read the recent history of taosmd-progress,
   general, and integration. Answer anything addressed to @taOSmd that is
   still unanswered.
4. IDENTITY RULES: the live session posts as @taOSmd and is the ONLY voice
   that answers questions or takes design positions. Scheduled/headless jobs
   post summaries only, under their own distinct sender (taosmd-sweep), and
   flag questions as "needs live session". One identity, one mind: anything
   that cannot see the whole conversation must not speak in it.
5. DISCIPLINE: progress lines go to taosmd-progress (start/finish/lesson).
   No credentials, hosts, or addresses on any channel, ever. No pleasantries;
   reply by point number; tag messages [DECISION]/[PROPOSAL]/[Q]/[ACK]/[INFO].
   On rate limit: one handoff line per the On-limit checklist in STATUS.md.
