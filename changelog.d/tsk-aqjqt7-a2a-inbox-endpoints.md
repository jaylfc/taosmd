### Added
* A2A inbox HTTP endpoints (`GET /a2a/inbox`, `POST /a2a/inbox/advance`, `POST /a2a/ack`, `GET /a2a/inbox/unhandled`) with registry auth derived from the verified token `sub`; remote forwarding for `a2a_inbox`, `a2a_inbox_advance`, `a2a_ack`, and `a2a_inbox_unhandled`.
