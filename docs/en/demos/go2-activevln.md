# Go2 physical demo

EmbodiRun executes ActiveVLN predictions from EmbodiInfer on a Go2 with an onboard
Orin NX and D435I RGB camera. This recording shows three execution modes.

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

[Watch the video — 3× playback, 9 seconds](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/activevln_go2/go2_activevln_3x.mp4)

| Mode | Recorded behavior |
|---|---|
| Chunk 1 | Observe, infer and execute one chunk. |
| Chunk 100 | Repeat synchronous chunks until STOP or the task limit. |
| RTC + Chunk 100 | Start the next inference while the current chunk executes. |

The numbers 1 and 100 limit chunks per instruction. Each inference returns at
most three actions. The footage was recorded on 2026-10-29 using ActiveVLN
Qwen2.5-VL-3B, BF16 and JetPack 5.1.1 / CUDA 11.4.

The RTC segment retains the autonomous turn and subsequent forward motion;
the later manual head-return sequence is excluded. The clips are different
recorded takes. Commands and timing annotations were typeset from the logs;
they illustrate the execution modes rather than a controlled speed comparison.

The implementation and deployment instructions remain in the
[Go2 adaptation PR](https://github.com/BUAA-CI-LAB/EmbodiRun/pull/13), pending integration.
