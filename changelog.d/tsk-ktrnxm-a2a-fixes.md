### Fixed
- Fixed short page issue in GET /a2a/messages where ACL read-filter loop could break early when service.a2a_feed dropped rows, causing fewer than limit messages to be returned even when more were available
- Fixed repeated identity resolution in ACL checks by caching the verified caller ID once per request instead of resolving it for every row
