### Fixed
- binary_quant and late_interaction stores now write `dense_format=none` marker instead of `dense_format=f32`; existing markers with `dense_format=none` or no `dense_format` key are accepted on reopen; dense-format logic applies only to plain dense stores
