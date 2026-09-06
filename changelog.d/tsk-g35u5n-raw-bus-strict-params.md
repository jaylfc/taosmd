### Fixed

- The A2A read endpoints (`GET /a2a/messages`, `GET /a2a/stream`, `GET /a2a/mentions`, and all other `GET /a2a/*`) now reject unknown query parameters with HTTP 400, naming both the offending parameter and the accepted set. This closes the silent-dropping defect where `after=` and `since_id=` (message-id cursors that these endpoints never accepted) were silently ignored, making a misspelt cursor indistinguishable from a working one. The 400 surfaces match the controller proxy (taOS #2390). No shipped client in this repo sends `after` or `since_id` to these endpoints.
