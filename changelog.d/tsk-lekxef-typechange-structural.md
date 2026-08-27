### Fixed

Treat typechange (T) as structural for doc-gate rules. Previously, a T status on taosmd/http_server.py or taosmd/service.py would not trigger the changelog or a2a-handlers rules, allowing undocumented structural changes. Now T is included alongside A and D in structural path evaluation.
