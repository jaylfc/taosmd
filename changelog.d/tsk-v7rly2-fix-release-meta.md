### Fixed

- `scripts/release_meta.py`: close silent "no migration" paths and pin mutants. UTF-8 BOM on trailer lines is now stripped; subdir README.md fragments are parsed (only top-level README.md skipped); inline backtick spans no longer toggle fence state; near-miss prefixes (`> -`, `1.`, `- [ ]`, `##`, full-width colon, ZWSP, plurals) now refuse with exit 2; non-UTF-8 fragments refuse with exit 2 instead of traceback; unreachable empty Manual-Step branch removed. All six mutation-test targets now killed by dedicated tests.
