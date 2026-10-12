## Problem
The `qmd-mode` embedder identity in `taosmd/vector_memory.py` returns the literal `"qmd:qmd"` in qmd embed mode, which prevents the store_meta mode marker from distinguishing between two different models served by qmd. If the model behind qmd serve changes, `_check_store_mode()` cannot detect this mismatch and new vectors may be mixed into a store built in a different vector space.

## Solution
Fix the `_embedder_identity()` method to return specific qmd model names when known, and update `_check_store_mode()` to properly handle qmd model transitions.

## Changes
1. In `init()`, fetch the qmd embed model from the health endpoint before `_check_store_mode()` runs and store it as `self._qmd_embed_model`
2. Update `_embedder_identity()` to return `f"qmd:{self._qmd_embed_model}"` when known, else the existing `"qmd:qmd"`
3. Fix `_check_store_mode()` case (b) logic to properly detect when the current embedder is "qmd:qmd" (health unreachable) and the recorded embedder is specific
4. Scope qmd special handling to only apply when the only conflict is on the embedder component
5. Ensure onnx and local identities behave exactly as before

## Tests
All existing tests in `tests/test_store_mode_guard.py` and `tests/test_zero_loss_integrity.py` continue to pass with the fixes applied.