window.BENCHMARK_DATA = {
  "lastUpdate": 1795068447393,
  "repoUrl": "https://github.com/kongxinjun/burn",
  "entries": {
    "Inference latency": [
      {
        "commit": {
          "author": {
            "name": "kongxin",
            "username": "kongxinjun",
            "email": "kongxin326@gmail.com"
          },
          "committer": {
            "name": "kongxin",
            "username": "kongxinjun",
            "email": "kongxin326@gmail.com"
          },
          "id": "1b1f3ec3f411a3d2a09c760bd380c5d6dcf843c3",
          "message": "test(lab): judge fuzz findings with PyTorch and track confirmed bugs\n\nDump each failing program with its data so ci/oracle.py can tell which side\nis wrong, group panics by message, stop on environment failures, cap new\nissues per run, and add the permuted-reduction bug as a known-bug test. Move\nMetal to a self-hosted workflow: hosted macOS runners lack Metal 3.",
          "timestamp": "2026-11-19T05:35:36Z",
          "url": "https://github.com/kongxinjun/burn/commit/1b1f3ec3f411a3d2a09c760bd380c5d6dcf843c3"
        },
        "date": 1795068446612,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "flex resnet50/b1 p50",
            "value": 68.562315,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b1 p90",
            "value": 69.20283400000001,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b1 first request",
            "value": 70.18705,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b1 load",
            "value": 77.603082,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b1 peak memory",
            "value": 238.14453125,
            "unit": "MiB"
          },
          {
            "name": "flex resnet50/b8 p50",
            "value": 510.01064399999996,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b8 p90",
            "value": 515.103295,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b8 first request",
            "value": 513.067733,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b8 load",
            "value": 72.913111,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b8 peak memory",
            "value": 243.05078125,
            "unit": "MiB"
          },
          {
            "name": "flex mobilenet_v2/b1 p50",
            "value": 16.007755,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b1 p90",
            "value": 16.240218000000002,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b1 first request",
            "value": 16.224812,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b1 load",
            "value": 13.675234,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b1 peak memory",
            "value": 243.05078125,
            "unit": "MiB"
          },
          {
            "name": "flex mobilenet_v2/b16 p50",
            "value": 267.3096,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b16 p90",
            "value": 269.809032,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b16 first request",
            "value": 277.991153,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b16 load",
            "value": 10.200034,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b16 peak memory",
            "value": 243.05078125,
            "unit": "MiB"
          },
          {
            "name": "flex minilm/b4 p50",
            "value": 13.290258,
            "unit": "ms"
          },
          {
            "name": "flex minilm/b4 p90",
            "value": 13.415954000000001,
            "unit": "ms"
          },
          {
            "name": "flex minilm/b4 first request",
            "value": 16.011015999999998,
            "unit": "ms"
          },
          {
            "name": "flex minilm/b4 load",
            "value": 66.428926,
            "unit": "ms"
          },
          {
            "name": "flex minilm/b4 peak memory",
            "value": 243.05078125,
            "unit": "MiB"
          },
          {
            "name": "flex distilgpt2/prefill p50",
            "value": 42.595071,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/prefill p90",
            "value": 43.547568,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/prefill first request",
            "value": 45.891647,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/prefill load",
            "value": 574.589072,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/prefill peak memory",
            "value": 1270.8671875,
            "unit": "MiB"
          },
          {
            "name": "flex distilgpt2/decode16 p50",
            "value": 295.12994399999997,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/decode16 p90",
            "value": 301.868056,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/decode16 first request",
            "value": 305.031428,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/decode16 load",
            "value": 574.8456090000001,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/decode16 peak memory",
            "value": 1270.9453125,
            "unit": "MiB"
          },
          {
            "name": "flex resnet50/b1 p50",
            "value": 671.247292,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b1 p90",
            "value": 1128.927167,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b1 first request",
            "value": 72.28,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b1 load",
            "value": 223.507959,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b1 peak memory",
            "value": 235.109375,
            "unit": "MiB"
          },
          {
            "name": "flex resnet50/b8 p50",
            "value": 610.5748749999999,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b8 p90",
            "value": 922.253875,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b8 first request",
            "value": 5040.793958,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b8 load",
            "value": 63.928709,
            "unit": "ms"
          },
          {
            "name": "flex resnet50/b8 peak memory",
            "value": 322,
            "unit": "MiB"
          },
          {
            "name": "flex mobilenet_v2/b1 p50",
            "value": 235.991959,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b1 p90",
            "value": 517.340875,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b1 first request",
            "value": 15.682708,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b1 load",
            "value": 68.383084,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b1 peak memory",
            "value": 335.703125,
            "unit": "MiB"
          },
          {
            "name": "flex mobilenet_v2/b16 p50",
            "value": 1320.0897499999999,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b16 p90",
            "value": 2137.876667,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b16 first request",
            "value": 914.605875,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b16 load",
            "value": 15.888375,
            "unit": "ms"
          },
          {
            "name": "flex mobilenet_v2/b16 peak memory",
            "value": 460.53125,
            "unit": "MiB"
          },
          {
            "name": "flex minilm/b4 p50",
            "value": 48.985875,
            "unit": "ms"
          },
          {
            "name": "flex minilm/b4 p90",
            "value": 119.10775,
            "unit": "ms"
          },
          {
            "name": "flex minilm/b4 first request",
            "value": 38.637,
            "unit": "ms"
          },
          {
            "name": "flex minilm/b4 load",
            "value": 335.78475000000003,
            "unit": "ms"
          },
          {
            "name": "flex minilm/b4 peak memory",
            "value": 591.390625,
            "unit": "MiB"
          },
          {
            "name": "flex distilgpt2/prefill p50",
            "value": 57.371542,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/prefill p90",
            "value": 69.079459,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/prefill first request",
            "value": 71.051542,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/prefill load",
            "value": 1738.5232500000002,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/prefill peak memory",
            "value": 1617.71875,
            "unit": "MiB"
          },
          {
            "name": "flex distilgpt2/decode16 p50",
            "value": 195.4915,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/decode16 p90",
            "value": 236.074875,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/decode16 first request",
            "value": 255.19750000000002,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/decode16 load",
            "value": 783.138291,
            "unit": "ms"
          },
          {
            "name": "flex distilgpt2/decode16 peak memory",
            "value": 1617.71875,
            "unit": "MiB"
          },
          {
            "name": "cpu+fusion resnet50/b1 p50",
            "value": 247.215889,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion resnet50/b1 p90",
            "value": 264.09684100000004,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion resnet50/b1 first request",
            "value": 161693.022692,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion resnet50/b1 load",
            "value": 307.45819,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion resnet50/b1 peak memory",
            "value": 4276.98828125,
            "unit": "MiB"
          },
          {
            "name": "cpu+fusion resnet50/b8 p50",
            "value": 1160.952486,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion resnet50/b8 p90",
            "value": 1187.061987,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion resnet50/b8 first request",
            "value": 147798.862658,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion resnet50/b8 load",
            "value": 93.65175500000001,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion resnet50/b8 peak memory",
            "value": 4341.4765625,
            "unit": "MiB"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b1 p50",
            "value": 66.171137,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b1 p90",
            "value": 69.143995,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b1 first request",
            "value": 166039.126464,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b1 load",
            "value": 355.849032,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b1 peak memory",
            "value": 4341.4765625,
            "unit": "MiB"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b16 p50",
            "value": 451.54649600000005,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b16 p90",
            "value": 475.19086799999997,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b16 first request",
            "value": 111557.33899999999,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b16 load",
            "value": 112.297594,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion mobilenet_v2/b16 peak memory",
            "value": 4341.4765625,
            "unit": "MiB"
          },
          {
            "name": "cpu+fusion minilm/b4 p50",
            "value": 102.444226,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion minilm/b4 p90",
            "value": 117.236098,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion minilm/b4 first request",
            "value": 36613.323759,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion minilm/b4 load",
            "value": 93.747093,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion minilm/b4 peak memory",
            "value": 4474.25,
            "unit": "MiB"
          },
          {
            "name": "cpu+fusion distilgpt2/prefill p50",
            "value": 169.93564,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion distilgpt2/prefill p90",
            "value": 188.90731399999999,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion distilgpt2/prefill first request",
            "value": 50723.952022,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion distilgpt2/prefill load",
            "value": 606.529811,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion distilgpt2/prefill peak memory",
            "value": 4474.25,
            "unit": "MiB"
          },
          {
            "name": "cpu+fusion distilgpt2/decode16 p50",
            "value": 1970.08479,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion distilgpt2/decode16 p90",
            "value": 2017.2469999999998,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion distilgpt2/decode16 first request",
            "value": 241285.305885,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion distilgpt2/decode16 load",
            "value": 679.055039,
            "unit": "ms"
          },
          {
            "name": "cpu+fusion distilgpt2/decode16 peak memory",
            "value": 4474.25,
            "unit": "MiB"
          }
        ]
      }
    ]
  }
}