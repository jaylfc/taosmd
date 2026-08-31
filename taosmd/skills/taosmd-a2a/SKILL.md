---
name: taosmd-a2a
version: 1.0.0
description: Set up agent-to-agent comms and named channels via the taOSmd A2A bus.
user-invocable: true
---

Print the setup guide from the taosmd package, then execute it. taOSmd must be
importable to read the guide, so install it first if it is missing:

```
python -c "import taosmd" 2>/dev/null || pip install taosmd
python -c "import taosmd; print(taosmd.a2a_setup_guide())"
```

Follow every step in the guide in order. The guide is the source of truth for
checking for an existing server, starting the bus, creating/joining a channel,
and generating the invite block for the user's other agents.

After joining a channel, set up **BOTH** the realtime wake (either `taosmd a2a-watch` for instant pickup or `taosmd a2a-bridge` to wake dormant sessions) **AND** the hourly cron floor. The realtime wake provides instant pickup while your session is active, and the hourly cron survives session death. They are complementary: the bridge only fires while it is running, the cron is the durable backup.

Follow the **"Durable monitoring"** section in the guide for idempotent cron/schtasks setup instructions. That section also covers realtime pickup: `taosmd a2a-watch` (stream new messages one line each, wrap in a process monitor for instant in-session pickup) and `taosmd a2a-bridge` (run a trigger command per message to wake a dormant session). Keep the hourly cron as the durable floor underneath either realtime wake mechanism.
