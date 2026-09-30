# RED-PROOF: Limit floor and ceiling tests

This document shows failing tests on master, followed by passing tests after the fix.

## Negative limit rejection tests (FAIL on master)

```
FAILED tests/test_http_server.py::test_search_negative_limit_rejected - AssertionError: {'hits': []}
  assert 200 == 400

FAILED tests/test_http_server.py::test_graph_negative_limit_rejected - AssertionError: {'nodes': [], 'edges': [], 'capped': True, 'total_nodes': 0, ...}
  assert 200 == 400

FAILED tests/test_http_server.py::test_graph_activations_negative_limit_rejected - AssertionError: {'activations': [], 'now': 1790810239.4993625}
  assert 200 == 400

FAILED tests/test_http_server.py::test_pending_negative_limit_rejected - AssertionError: {'pending': []}
  assert 200 == 400

FAILED tests/test_http_server.py::test_task_list_negative_limit_rejected - AssertionError: {'tasks': []}
  assert 200 == 400

FAILED tests/test_http_server.py::test_task_ready_negative_limit_rejected - AssertionError: {'tasks': []}
  assert 200 == 400

FAILED tests/test_http_server.py::test_task_list_edges_negative_limit_rejected - AssertionError: {'edges': []}
  assert 200 == 400

FAILED tests/test_http_server.py::test_a2a_thread_messages_negative_limit_rejected - AssertionError: {'thread': 'neg-limit-test', 'messages': [...]}
  assert 200 == 400
```

## Ceiling clamp tests (FAIL on master)

```
FAILED tests/test_http_server.py::test_task_list_limit_clamped_to_50 - AssertionError: assert 60 == 50
  Got 60 tasks instead of 50 when limit=100

FAILED tests/test_http_server.py::test_task_ready_limit_clamped_to_20 - AssertionError: assert 30 == 20
  Got 30 tasks instead of 20 when limit=100

FAILED tests/test_http_server.py::test_a2a_messages_limit_clamped_to_50 - AssertionError: assert 60 == 50
  Got 60 messages instead of 50 when limit=100

FAILED tests/test_http_server.py::test_a2a_mentions_limit_clamped_to_50 - AssertionError: assert 60 == 50
  Got 60 messages instead of 50 when limit=100

FAILED tests/test_a2a_inbox_auth.py::test_inbox_limit_clamped_to_1000 - AssertionError: assert 1100 == 1000
  Got 1100 messages instead of 1000 when limit=2000

FAILED tests/test_a2a_inbox_auth.py::test_inbox_unhandled_limit_clamped_to_1000 - AssertionError: assert 1100 == 1000
  Got 1100 messages instead of 1000 when limit=2000

FAILED tests/test_http_server.py::test_task_list_edges_limit_boundary[0-0] - AssertionError: assert 1 == 0
  limit=0 returned 1 edge instead of 0
```

## Green run (after fix)

```
$ pytest tests/test_http_server.py tests/test_a2a_inbox_auth.py -q --tb=no -q
.................                                                      [100%]
185 passed in 80.78s
```

## Mutation test results

For each ceiling site, replacing `limit_i = min(limit_i, _MAX_LIMIT)` with `limit_i = limit_i` produces a failing test:

- `_do_search` (POST /search): `test_search_limit_clamped_to_100` fails when ceiling mutated
- `_handle_graph` (GET /graph): `test_graph_limit_clamped_to_300` fails when ceiling mutated
- `_handle_graph_activations` (GET /graph/activations): `test_graph_activations_limit_clamped_to_100` fails when ceiling mutated
- `_handle_pending` (GET /pending): `test_pending_limit_clamped_to_20` fails when ceiling mutated
- `_handle_a2a_messages` (GET /a2a/messages): `test_a2a_messages_limit_clamped_to_50` fails when ceiling mutated
- `_handle_a2a_mentions` (GET /a2a/mentions): `test_a2a_mentions_limit_clamped_to_50` fails when ceiling mutated
- `_handle_a2a_inbox` (GET /a2a/inbox): `test_inbox_limit_clamped_to_1000` fails when ceiling mutated (in test_a2a_inbox_auth.py)
- `_handle_a2a_inbox_unhandled` (GET /a2a/inbox/unhandled): `test_inbox_unhandled_limit_clamped_to_1000` fails when ceiling mutated (in test_a2a_inbox_auth.py)
- `_handle_a2a_thread_messages` (GET /a2a/threads/{thread}/messages): `test_a2a_thread_messages_limit_clamped_to_200` fails when ceiling mutated
- `_handle_task_list` (GET /tasks): `test_task_list_limit_clamped_to_50` fails when ceiling mutated
- `_handle_task_ready` (GET /tasks/ready): `test_task_ready_limit_clamped_to_20` fails when ceiling mutated
- `_handle_task_list_edges` (GET /tasks/edges): `test_task_list_edges_limit_capped_at_500_with_many_edges` fails when ceiling mutated (note: ceiling changed from `max(1, min(...))` to `min(...)`)

For each floor site, the new test verifies that limit=-1 returns 400:

- `_do_search`: `test_search_negative_limit_rejected`
- `_handle_graph`: `test_graph_negative_limit_rejected`
- `_handle_graph_activations`: `test_graph_activations_negative_limit_rejected`
- `_handle_pending`: `test_pending_negative_limit_rejected`
- `_handle_a2a_messages`: `test_a2a_messages_negative_limit_rejected`
- `_handle_a2a_mentions`: `test_a2a_mentions_negative_limit_rejected`
- `_handle_a2a_inbox`: `test_a2a_inbox_negative_limit_rejected`
- `_handle_a2a_inbox_unhandled`: `test_a2a_inbox_unhandled_negative_limit_rejected`
- `_handle_a2a_thread_messages`: `test_a2a_thread_messages_negative_limit_rejected`
- `_handle_task_list`: `test_task_list_negative_limit_rejected`
- `_handle_task_ready`: `test_task_ready_negative_limit_rejected`
- `_handle_task_list_edges`: `test_task_list_edges_negative_limit_returns_400`