# Burn Inference Lab

A personal suite for finding inference bugs and regressions in Burn. It lives on the `lab` branch of
the fork and runs on GitHub Actions against the latest upstream `main`: every job checks out
`tracel-ai/burn` and copies this crate into it, so the branch never needs rebasing.

## What runs

| Workflow | When | What it does |
| --- | --- | --- |
| `lab-functional` | Every push to `lab`, daily | Small and pretrained models on Flex, CubeCL CPU, wgpu (lavapipe) and Metal, with and without fusion |
| `lab-fuzz` | Every 6 hours | Random inference programs on each backend against Flex; files one issue per distinct failure |
| `lab-performance` | Daily | Latency and memory history on GitHub Pages, Callgrind instruction counts, and a same-machine comparison with the previous day's Burn that files an issue on regression |
| `lab-sanitizers` | Weekly | AddressSanitizer on the suites, ThreadSanitizer on the concurrency tests |

All of them can be started by hand from the Actions tab, with an upstream branch, tag or commit.
Hosted runners have no NVIDIA or AMD GPU: wgpu runs on lavapipe, Mesa's software Vulkan driver, and
Metal on the macOS runners' virtual GPU. CUDA and ROCm are not covered.

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
- `lab-fuzz` builds random chains of tensor and module operations and runs each one lazily, letting
  fusion see the whole chain, and eagerly, synchronizing after every operation. A mismatch with
  Flex is reduced to its shortest failing prefix and written as JSON with the command that
  reproduces it.

The `fusion` feature turns fusion on; without it, the CubeCL backends run unfused. Burn's `default`
feature would enable fusion everywhere, so this crate does not use it.

## Known issues

On macOS, CubeCL's exit handler aborts Metal programs after `main` returns. The test and tool
binaries leave through `_exit` to report their real status; set `LAB_NORMAL_EXIT=1` to see the
abort. `lab-functional` tracks it in a step allowed to fail.
