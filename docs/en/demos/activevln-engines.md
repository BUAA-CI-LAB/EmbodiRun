# ActiveVLN engines inside EmbodiRun

EmbodiInfer, SGLang and vLLM all supply inference to **EmbodiRun**, which owns the
same Habitat observation and action loop. The original-speed clip shows total
task elapsed time and current geodesic distance to the goal.

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_engines/activevln_engines_1x.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_engines/activevln_engines_1x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Recorded task

All three runs follow R2R validation-unseen episode 17 in scene `2azQ1b91cZZ`:
cross the billiard room to the opposite window overlooking the garden. They use
the same initial state, checkpoint (`Arvil/Qwen2.5-VL-3B_rl_r2r_4000`, revision
`160987313e3e869705f42400d1b8f28177044518`), RTX 4090, BF16, batch 1, greedy
decoding, 640×480 RGB, 25 cm forward steps and 15 degree turns.

| Inference engine | Task time | Final distance | Primitive steps / calls |
|---|---:|---:|---:|
| EmbodiInfer | 1.979 s | 0.673 m | 49 / 11 |
| SGLang | 2.736 s | 0.673 m | 49 / 11 |
| vLLM | 2.682 s | 0.673 m | 49 / 11 |

Each selected run issues STOP within the task's 3 m success threshold. This is
one episode recorded on 2026-11-02, not an aggregate navigation benchmark.

Timing starts after session creation and ends at actual STOP, including HTTP,
inference and simulation. It excludes service startup, scene loading, one warmup
episode and offline video encoding. Frames retain their measured monotonic
capture times. The 4.9 s edit includes 0.65 s of introduction and 1.5 s of ending;
playback is 1× with no slowdown or generated intermediate motion.

SGLang initially stopped at 2.458 m. After a fresh service startup, all three
recorded reruns stopped at 0.673 m; the video uses the **first** recorded rerun,
selected before collecting them, not the fastest. The original is retained in
the experiment archive. The cause of the cross-startup output variation remains
unresolved. All selected runs use their engines' tuned configurations; identical
precision and checkpoint do not imply identical kernels.

## Integration and reproduction

The [vLLM service package](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/integrations/vllm_activevln)
provides the new `provider: vllm` action endpoint, and the
[Habitat example](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/activevln_engines)
shows its deployment configuration. EmbodiInfer and SGLang integrations already
exist. SGLang's established packaged provider is for π0.5; this ActiveVLN video
uses the experimental native wrapper with the common action HTTP contract.

The footage records the experimental services used for the comparison. The new
reusable vLLM package has CPU protocol and failure-path tests; its packaged
entrypoint has not been rerun on a GPU and its defaults do not claim these timings.
Keep that validation separate from the recorded result.
