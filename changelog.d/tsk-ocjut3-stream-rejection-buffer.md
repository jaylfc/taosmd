### Fixed

- `_stream_rejection` in `tests/test_a2a.py` now accumulates all received data into a single buffer until the `\r\n\r\n` separator is found, instead of discarding already-read bytes across separate 128-byte chunk reads. This eliminates the latent fragility where the function worked only by luck of buffer sizes.