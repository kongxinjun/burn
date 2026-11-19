# Burn Inference Lab

A personal suite for finding inference bugs and regressions in Burn. It lives on the `lab` branch of
the fork and runs on GitHub Actions against the latest upstream `main`: every job checks out
`tracel-ai/burn` and copies this crate into it, so the branch never needs rebasing.

## What runs

| Workflow | When | What it does |
| --- | --- | --- |
| `lab-functional` | Every push to `lab`, daily | Small and pretrained models on Flex (Linux and macOS), CubeCL CPU and wgpu (lavapipe), with and without fusion, plus the known-bug tests |
| `lab-fuzz` | Every 6 hours | Random inference programs on CubeCL CPU and wgpu against Flex; PyTorch judges each failure, and the distinct ones are listed in the job summary |
| `lab-performance` | Daily | Latency and memory history on GitHub Pages, Callgrind instruction counts, and a same-machine comparison with the previous day's Burn that flags regressions |
| `lab-sanitizers` | Weekly | AddressSanitizer on the suites, ThreadSanitizer on the concurrency tests |
| `lab-metal` | By hand | Everything above on Metal, on a self-hosted runner labelled `metal` |

All of them can be started by hand from the Actions tab, with an upstream branch, tag or commit.
Hosted runners have no usable GPU: wgpu runs on lavapipe, Mesa's software Vulkan driver, and the
macOS runners' paravirtual GPU lacks the Metal 3 that CubeCL requires. CUDA and ROCm are not
covered.

To run `lab-metal` on your Mac, add a runner under Settings → Actions → Runners → New self-hosted
runner, give it the extra label `metal`, and start it only while you dispatch the workflow. The
workflows never run on pull requests, so code from other people's forks does not reach it.

## Triage

`lab-fuzz` and `lab-metal` write every failure with its whole program, and `ci/oracle.py`
recomputes it with PyTorch. The verdict is `device-wrong`, `flex-wrong` (Flex, the reference, is
the buggy one), `both-wrong` or `pytorch-rejected`. `ci/summarize_findings.py` lists the distinct
findings with their verdicts in the job summary, grouping panics by message; the programs are in the
`findings-*` artifacts. The fuzzer stops when most programs fail, which points to the environment.

Findings are leads, not confirmed bugs, so no workflow opens issues. Open one by hand only once a
bug reproduces in a standalone program on upstream `main`, an independent reference such as
PyTorch shows Burn is wrong, and no existing issue covers it.

Confirmed bugs become tests in `tests/known_bugs.rs`, which CI runs in a step allowed to fail; a
test that starts passing marks the fix. `LAB_SKIP` skips tests by name fragment for known
limitations of a runner.

## Models

`zoo/generate.py` exports pretrained weights under Burn's module paths, with the outputs PyTorch
computes for them:

| Model | Source | What it exercises |
| --- | --- | --- |
| ResNet-50 | torchvision `IMAGENET1K_V2` | Convolutions, batch norm, pooling, residuals |
| MobileNetV2 | torchvision `IMAGENET1K_V2` | Depthwise convolutions, ReLU6, inverted residuals |
| all-MiniLM-L6-v2 | sentence-transformers | BERT encoder, padding masks, mean pooling |
| distilgpt2 | Hugging Face | Causal masks, tied embeddings, greedy and KV-cached generation |

`fixtures/` holds two tiny models whose weights are committed, for runs without the export.

## Running locally

```sh
uv run zoo/generate.py ../../target/lab-models   # Once; about 700 MB
cargo test-metal                                 # Tiny models
cargo zoo-metal                                  # Pretrained models
cargo zoo-metal-no-fusion
cargo run --release --bin lab-bench --features metal,fusion
cargo run --release --bin lab-fuzz --features metal,fusion -- --seconds 120
```

Aliases exist for `flex`, `cpu`, `wgpu`, `metal` and `cuda`. Set `BURN_LAB_MODELS` when the models
live elsewhere than `target/lab-models`.

## Tools

- `lab-bench` measures load time, first request, p50 and p90 latency, and peak memory per
  scenario. Every request reads its output back, so timings include execution.
- `lab-fuzz` builds random chains of tensor and module operations in f32, f16 or bf16 (`--dtypes`)
  and runs each one lazily, letting fusion see the whole chain; eagerly, synchronizing after every
  operation; on another batch size when no operation depends on it; and lazily again, once kernels,
  tuning results and memory are cached. Flex in f32 is the reference. For f16 and bf16, Flex also
  runs the program in that precision, and its distance from f32 sets how much error the device may
  add; programs that overflow or lose most of their precision are skipped. A mismatch is reduced to
  its shortest failing prefix and written as JSON with the command that reproduces it. Fingerprints
  name the dtype only when the program passes in f32.
- Each finding also has a `.case.json`: the session of the process up to the failure, every run it
  made on the device, by seed. A failure can depend on kernels, tuning results or memory left by
  earlier programs, and only reproduces with them. `lab-fuzz replay <case>` replays a case in a new
  process; `lab-fuzz reduce <case> --out <path>` shrinks it, replaying each candidate in a new
  process, to the earlier runs, operations and values the failure needs, with random values
  replaced by short patterns when the failure survives it.

The `fusion` feature turns fusion on; without it, the CubeCL backends run unfused. Burn's `default`
feature would enable fusion everywhere, so this crate does not use it.

## Known issues

On macOS, CubeCL's exit handler aborts Metal programs after `main` returns. The test and tool
binaries leave through `_exit` to report their real status; set `LAB_NORMAL_EXIT=1` to see the
abort. `lab-functional` tracks it in a step allowed to fail.
