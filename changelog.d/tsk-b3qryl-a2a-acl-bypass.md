### Fixed
- A2A per-channel read ACL: filter channels, members, census, and thread-messages endpoints so restricted channels are not disclosed to unauthenticated or unauthorized callers
- A2A admin channel-acl GET route: make it reachable with admin_token independent of server_token
- A2A message feed limit starvation: over-fetch when no thread filter is set so restricted rows cannot consume the public limit window
- ACL config parsing: deny on malformed read/post values and list-shaped entries instead of failing open to wildcard
