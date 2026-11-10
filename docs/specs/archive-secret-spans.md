# Encrypted secret spans in the raw archive

Status: DRAFT for Jay. Implements his 2026-11-10 ruling (option 1). No code ships
with this document. Open decisions are listed in section 9.

## Goal

The raw archive is the permanent, word-for-word record. Today `Archive.record`
rewrites secrets to `[REDACTED:<kind>]` before the JSONL write, so the raw copy
loses bytes (and false positives, see tsk-ycyw7m, mangle ordinary prose for good).

Jay's ruling: raw stays byte-verbatim; every matched secret span is stored
ENCRYPTED inside the raw record; every readable or derived view (bus reads,
search, vector, claims, exports) shows the redacted form.

## 1. What is wrong today (measured on master 48db90a6)

- `archive.py` `record()` calls `redact_secrets` on `summary` and on the string
  values of `data` under the keys `content`, `text`, `msg`, `query`, `body`, and
  writes the redacted text to JSONL, `archive_index.data_json` and `archive_fts`.
  The original secret bytes are gone.
- It assigns `data[key] = ...` in place, so the caller's dict is mutated.
- The FTS loop indexes `content`, `text`, `msg`, `query`, `tool`, `result`,
  `error`. `tool`, `result` and `error` are never redacted, so a secret in a tool
  result lands in plaintext in the JSONL, in `data_json` and in `archive_fts`.
- `redact_secrets` applies patterns one after another to the already-rewritten
  string. Match offsets are relative to a mutated string, and a later pattern can
  match inside an earlier placeholder. It cannot produce spans of the original.

## 2. Span finder

New pure function in `taosmd/secret_filter.py`:

    find_secret_spans(text: str) -> list[SecretSpan]   # SecretSpan(start, end, kind)

- Runs every pattern against the ORIGINAL string, never a rewritten one.
- Overlaps resolve as: earliest start wins; on a tie, the longest span wins;
  any span overlapping a kept span is dropped. Output is sorted and disjoint.
- Offsets are Python `str` code point indices (what survives a JSON round trip),
  not byte offsets.
- For patterns with a capture group (`aws_secret`, `generic_api_key`,
  `password_field`) the span is the WHOLE match, the same text `redact_secrets`
  replaces today, so the readable view does not change shape.
- `redact_secrets` is reimplemented on top of it (splice placeholders over the
  spans) so there is exactly one detector. Its return shape is unchanged.

## 3. Scan scope

One scope, used for encryption AND for every derived copy: every string leaf of
`data`, walked recursively through dicts and lists, plus `summary`. A leaf is
addressed by a JSON path list, e.g. `["data", "result"]` or
`["data", "messages", 3, "content"]`. Non-string leaves are untouched.

## 4. On-disk record shape (schema v2)

The stored line carries the event exactly as today, except:

- each string leaf that had spans holds the REDACTED form (placeholders spliced
  over the spans), so every existing reader sees what it sees today;
- a new top-level key `enc_spans`: a list of
  `{"path": [...], "start": int, "end": int, "kind": str, "kid": str,
  "nonce": b64, "ct": b64}`, where `start`/`end` index the ORIGINAL leaf string;
- a new top-level key `"secret_schema": 2`. Rows without it are legacy: their
  redaction is permanent and nothing can recover it.

`reveal(event) -> event` decrypts each span and rebuilds each original leaf by
walking the redacted leaf left to right: copy text up to the placeholder,
emit the plaintext, skip the placeholder. The result must equal the caller's
input byte for byte (leading/trailing whitespace and non-ASCII included).

The existing line `sha256` keeps covering the stored line (placeholders plus
ciphertext). No hash of any plaintext secret is stored anywhere: a plain digest
of a short password is a dictionary oracle. The AEAD tag is the integrity check.

Direct JSONL readers (`session_catalog._read_archive_file`, the catalog line
fallback, `verify_day`, `export_day`, `catalog_pipeline`) parse each line into a
dict and ignore keys they do not use, the same way they ignore `sha256` today
(read 2026-11-10). They see the redacted form with no change.

## 5. Crypto

- AEAD: AES-256-GCM from the `cryptography` package
  (`cryptography.hazmat.primitives.ciphers.aead.AESGCM`), 12-byte random nonce
  per span from `os.urandom`.
- Associated data binds a span to its place:
  `json.dumps([timestamp, event_type, path, start, end, kind, kid])` (compact,
  sorted). A span copied into another record, field or position fails to decrypt.
- No home-made primitives, no Fernet (it is AES-CBC plus HMAC with no AAD).

## 6. Keys

- Every span records its `kid`. Rotation = mint a new key and new kid for new
  writes; old keys stay readable forever. Rows are NEVER re-encrypted or
  rewritten (the archive is append-only).
- Key provider interface, chosen at `Archive.init()`:
  - taOS present: the key lives in taOS Secrets (Jay's ruling).
  - Standalone taosmd: a local key file `~/.taosmd/keys/archive-<kid>.key`,
    mode 0600, directory 0700, created on first init if absent. This is MY
    extension of the ruling, so standalone still works without taOS; it is
    decision D2 below.
- The key never lives under the archive dir, is never written to the index, and
  is excluded from export and backup of the archive itself.
- `Archive.init()` fails LOUD if a key cannot be loaded or created. It never
  falls back silently at `record()` time.

## 7. Read and derive paths

Because the stored leaves are already redacted, every current reader shows the
redacted form with no change: `search_fts`, `query`, `get_event`, `recent`,
`export_day`, `verify_day`, the session catalogue and the bus (which reads the
index). `VectorMemory.add` keeps its own redaction; claims get redacted text via
tsk-oisysv.

The only path that touches a key is an explicit `Archive.reveal(event)` (and a
CLI `taosmd archive export --reveal` built on it). It is NOT exposed by
`http_server.py`, the bus, or MCP. Any future remote route needs its own spec.

## 8. Cards, in order (all labelled security, each PR needs Jay's per-PR word)

1. **Copy-before-transform + derived-copy leak.** `record()` deep-copies `data`
   before any change, so the caller's dict is untouched. Apply the section 3
   scope to `data_json` and `archive_fts` (closes the `tool`/`result`/`error`
   leak). Raw JSONL behaviour unchanged in this card. Pure, no crypto.
2. **`find_secret_spans`** per section 2, with `redact_secrets` rebuilt on it.
   Tests: the three prose strings from tsk-ycyw7m return no span; overlapping
   matches resolve per the rule; a mutation that swaps the tie rule goes red.
3. **Codec + key provider + `reveal`** (no wiring into `record()` yet).
   Tests: `reveal(encode(x)) == x` for padded, multi-line and non-ASCII input;
   one flipped byte in `ct`, `nonce` or any AAD field makes decrypt refuse;
   key file is created 0600; init with an unreadable key fails loud.
4. **Wire into `record()`** (schema v2 lines) + `export --reveal`. Positive
   control: after `record()` of a known token, the token's bytes are absent from
   the JSONL file, `data_json` and `archive_fts`, and `reveal` returns it.

## 9. Decisions for Jay

- **D1 (dependency):** `cryptography` is only in the `registry` and `dev`
  extras today. Recommended: promote it to a core dependency (wheels exist for
  x86_64, aarch64 and macOS), so the only runtime failure left is key I/O.
  Alternative: keep it optional and, without it, write today's permanent
  redaction with no `enc_spans` (raw stays lossy on those installs).
- **D2 (standalone key):** a 0600 key file under `~/.taosmd/keys/` when taOS is
  absent (recommended), versus requiring taOS Secrets and so no encrypted spans
  in standalone.
