# SO-101 engine comparison

Compare π0.5 inference latency and full action-chunk time with EmbodiInfer,
SGLang, and native LeRobot. The [report](../../docs/en/demos/engine-e2e-contrast.md)
contains the results, timing breakdown, and execution video.

## Conditions

| Setting | Value |
|---|---|
| Robot | One SO-101 arm, front and wrist cameras at 640 × 480 |
| Control host | Raspberry Pi 4 |
| Inference host | Jetson AGX Thor |
| Task | Pick up the cube and place it in the bowl |
| Policy | Same SO-101 π0.5 checkpoint, 10 denoising steps |
| Action chunk | 50 steps at 20 Hz |
| Measurement | One run of 15 chunks per configuration; median latency |

EmbodiInfer uses BF16, Inductor, CUDA graphs, and Triton attention. SGLang uses
upstream defaults; native LeRobot uses eager execution. All configurations use
the same cameras, checkpoint, and Wi-Fi network.

## Data and figure

[recorded-summary.json](recorded-summary.json) contains the medians and conditions
from the published report. The current dataset contains aggregate values;
per-chunk samples are unavailable, so the figure reports medians without P95
or error bars.

Generate the SVG from the repository root with Python 3.10+:

```bash
python benchmarks/engine-comparison/generate_plot.py
python benchmarks/engine-comparison/generate_plot.py --check
```

The output is
[so101-engine-comparison.svg](../../docs/assets/performance/so101-engine-comparison.svg).
Separate panels show inference and full-chunk latency. CI checks that the
tracked figure matches the summary data.

## Run on your hardware

The [engine comparison Recipe](../../examples/engine_comparison/README.md)
provides configuration profiles and launch commands. Preserve per-chunk timings
and run metadata when taking new measurements; use the
[measurement contract](../README.md#measurement-contract) for published results.

| Additional data | Status |
|---|---|
| Original per-chunk samples, exact checkpoint/source revisions, warmup count, and baseline dtypes | 🟨 待补充 |
| Repeated trials and uncertainty estimates | 🟨 待补充 |

[中文报告](../../docs/zh/demos/engine-e2e-contrast.md)
