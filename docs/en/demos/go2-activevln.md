# ActiveVLN on a physical Go2

EmbodiRun connects the Go2 camera and hardware owner to EmbodiInfer's ActiveVLN
HTTP service. The same instruction can execute one chunk, continue through
multiple chunks, or overlap the next inference with current motion.

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

[3× version (9 seconds)](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.mp4) · [Deployment guide](../go2-activevln.md)

The demo is 9 seconds at 3× playback speed. The three cases use:

| Case | CLI options | Behavior |
|---|---|---|
| Chunk 1 | `--mode chunk --max-chunks 1` | Observe, infer and execute once |
| Chunk 100 | `--mode chunk --max-chunks 100` | Repeat synchronous turns until STOP or a task bound |
| RTC + Chunk 100 | `--mode overlap --max-chunks 100 --overlap-lead-s 1.5` | Begin the next inference during the current motion chunk |

Here 1 and 100 are the maximum number of **chunks per instruction**, not the
number of actions returned by a single inference. Each model response contains
at most three actions. The video re-typesets CLI options and log excerpts for
readability; they are not a live terminal or synchronized telemetry.

## Recorded setup and limits

The footage was recorded on 2026-10-29 with a Go2, onboard Orin NX, D435I RGB,
and ActiveVLN `Arvil/Qwen2.5-VL-3B_rl_r2r_4000`
(revision `160987313e3e869705f42400d1b8f28177044518`). The JP5 eager service
uses BF16-converted weights on JetPack 5.1.1 / CUDA 11.4. The deployment guide
separates this setup from the newer optimized benchmark environment.

The RTC segment retains the first run's turn, forward motion, natural pause,
and following forward motion. It excludes the later manual head-return sequence.
The static timeline illustrates **1.493 seconds of overlap** and **1.861 seconds
of boundary waiting** from that run's log. This demonstrates execution modes;
it is not a controlled comparison of success rates or a general speedup claim.
The three clips are different tasks or takes, rather than matched trials.

## Run the modes

After preparing the owner and inference service with the deployment guide:

```bash
CONFIG=configs/activevln/go2-orin-nx-jp5-eager.json
embodirun go2 activevln plan --config "$CONFIG"
embodirun go2 activevln check --config "$CONFIG"
embodirun go2 activevln interactive --config "$CONFIG" --mode chunk --max-chunks 1 --execute
embodirun go2 activevln interactive --config "$CONFIG" --mode chunk --max-chunks 100 --execute
embodirun go2 activevln interactive --config "$CONFIG" --mode overlap --max-chunks 100 \
  --overlap-lead-s 1.5 --execute
```

Run one mode at a time. The owner must be deliberately armed by the operator;
omitting `--execute` performs inference without motion. Overlap requires the
service's execution-prefix acknowledgement capability. STOP and the configured
runtime/chunk bounds end the task; exhausting a bound is not a success result.

[Reproduce the demos](../examples.md) · [Support matrix](../support-matrix.md)
