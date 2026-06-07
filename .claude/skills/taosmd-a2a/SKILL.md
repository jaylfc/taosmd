---
name: taosmd-a2a
description: Set up agent-to-agent comms and named channels via the taOSmd A2A bus.
user-invocable: true
---

Print the setup guide from the taosmd package, then execute it. taOSmd must be
importable to read the guide, so install it first if it is missing:

```
python -c "import taosmd" 2>/dev/null || pip install git+https://github.com/jaylfc/taosmd
python -c "import taosmd; print(taosmd.a2a_setup_guide())"
```

Follow every step in the guide in order. The guide is the source of truth for
checking for an existing server, starting the bus, creating/joining a channel,
and generating the invite block for the user's other agents.
