### Fixed

- `tests/test_a2a_inbox_cursors.py`: the gate for `a2a_inbox`'s default kind exclusion was vacuous. Its alarm, ack, receipt and digest fixture messages were not addressed to the consumer, so the addressed-filter removed them and deleting the kind filter entirely left the suite green. Those four messages now mention the consumer, so the kind filter is the only rule that can exclude them, and `include_kinds` gained the coverage it had none of.
