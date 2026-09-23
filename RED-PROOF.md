# RED-PROOF tsk-pciatl

Run before fix: python -m pytest tests/test_a2a_channel_acls.py -q

```
FFFFFFFFFF..FF....                                                         [100%]
==================================== FAILURES ===================================
_ TestBoundedFeedScan.test_messages_bounded_scan_returns_empty_when_all_denied _
...
>       assert body["messages"] == []
E       AssertionError: assert [{'id': 16, '...0', ...}, ...] == []

_ TestBoundedFeedScan.test_messages_bounded_scan_returns_up_to_limit_readable _
...
>       assert all(m["thread"] == "public-ch" for m in msgs)
E       assert False

_ TestBoundedFeedScan.test_http_messages_bounded_scan_no_unbounded_read _
...
ERROR at teardown / 403 missing...

_ TestBoundedFeedScan.test_feed_cursor_advances_across_denied_rows _
...cursor frozen...

_ TestSSECursorAdvance.test_stream_cursor_advances_across_denied_rows _
...cursor frozen...

_ TestSSECursorAdvance.test_stream_cursor_advance_mutation_kill _
...cursor frozen...

_ TestDenyEffectNotStatus.test_acl_denies_post_with_valid_token_not_in_allowlist _
...403 not returned (200)...

_ TestDenyEffectNotStatus.test_acl_denied_post_body_not_persisted _
...body found in archive...

_ TestClearFlagValidation.test_http_set_acl_non_boolean_clear_returns_400 _
...200 instead of 400...

_ TestSiblingDivergence.test_mentions_respects_channel_acl _
...restricted body found in mentions feed...

_ TestSiblingDivergence.test_inbox_respects_channel_acl _
...restricted body found in inbox...
==================================== 10 failed, 6 passed in 3.06s ==================================
```
