---
name: taosmd-a2a
description: Set up agent-to-agent comms and named channels via the taOSmd A2A bus.
user-invocable: true
---

Read the setup guide from the taosmd package and execute it:

```
python -c "import taosmd; print(taosmd.a2a_setup_guide())"
```

Follow every step in the guide in order. The guide is the source of truth for installing taOSmd, checking for an existing server, starting the bus, joining a channel, and generating the invite block for the user's other agents.
