# E-048 run(): persist per-query rankings to a sidecar

## Summary

Fixes the core issue where `true_rank` was incorrectly computed from only the top-10 scores instead of the full ranking. This ensured that the true item rank reflects its position in the complete ranking (which can be >= 10), not just within the top-K returned by `rank_topk`.

### Key Changes

1. **Fixed true_rank calculation** in `benchmarks/e048/retrieval.py:268-271`:
   - Previously used `rank_topk` output scores (top-10 only) to compute `true_rank`
   - Now computes full ranking using all item vectors to correctly position items that rank beyond top-10

2. **Updated test assertion** in `tests/test_e048_retrieval.py:822-823`:
   - Changed `assert 0 <= entry["true_rank"] < 10` to `assert 0 <= entry["true_rank"]`
   - Now correctly allows `true_rank >= 10` as required by spec

3. **Added comprehensive test** `test_run_with_rankings_path` verification:
   - New test verifies that `true_rank` values from the rankings file reproduce `R@1/R@5/R@10` metrics exactly
   - Ensures consistency between the saved rankings data and computed metrics

4. **Added CLI end-to-end test** `test_cli_with_rankings_out`:
   - Tests the `--rankings-out` CLI argument functionality
   - Validates that the CLI properly writes rankings files and includes metadata

### Impact

- Fixes critical bug where `true_rank` calculations would fail (IndexError) for items not in top-10
- Ensures dual-evidence arm requirements are met with complete per-query rankings
- Maintains backward compatibility with existing `recall_at_k` metrics (unchanged)
- Enables proper persistence of per-query rankings to sidecar files via CLI

### Tests Added

- Test verifying `true_rank` values reproduce `R@1/R@5/R@10` from rankings file
- CLI end-to-end test with `--rankings-out` option
