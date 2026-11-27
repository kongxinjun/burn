# hunt/harness (fork only, never upstream)

Bug-hunting probes for Burn, run on win-bench (`kongxinjun/win-bench`, suites `burn/probe` and
`burn/probe-cuda`). Each probe is a test target in `crates/burn-backend-tests/tests/` that runs every
case on `Device::default()` (`BURN_DEVICE`) and on Flex, and compares the results.

- `wb_hunt.rs`: fused concat, NaN predicates and nan reductions, random programs with syncs
  (forward and gradients), quantization.
- `wb_probe.rs`, `wb_focus.rs`, `wb_reduce_perf.rs`: earlier rounds (copied from win-bench `_probes`).

```sh
winbench run burn probe-cuda hunt/harness probe=wb_hunt fusion=1
winbench run burn probe hunt/harness probe=wb_hunt backend=vulkan fusion=1
# locally: BURN_DEVICE=wgpu cargo test -p burn-backend-tests --no-default-features \
#   --features flex,wgpu,fusion,std --test wb_hunt -- --nocapture
```

`WB_SCALE` (`scale=` on win-bench) multiplies the number of random programs; `WB_VERBOSE=1` prints
passing cases too.
