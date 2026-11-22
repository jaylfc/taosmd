### Fixed
- data_dir now reaches VectorMemory via _ensure_stores constructor call
- JSON integer embedder_gpu_mem_mb values are now accepted by get_model_role
- Non-positive embedder_gpu_mem_mb values (<=0) are now rejected with a warning
