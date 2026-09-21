# Optional integrations

These packages have separate installation and process lifecycles. Installing
Deploy's core does not install model runtimes, camera libraries, or robot SDKs.

| Package | Responsibility | Core boundary |
| --- | --- | --- |
| [sglang_pi05](sglang_pi05/README.md) | SGLang-backed policy service | Versioned inference API through a model provider |
| [xlerobot_owner](xlerobot_owner/README.md) | XLeRobot hardware owner, teleoperation and collection | Robot owner HTTP API through the XLeRobot adapter |
| [microduck_vln](microduck_vln/README.md) | Optional MicroDuck simulation SDK and ActiveVLN service process | Existing EmbodiRun HTTP client; episode loop in `examples/microduck_vln` |
| [so101_wired_teleop](so101_wired_teleop/README.md) | Wired multi-leader SO-101 teleoperation and follower-side episode collection | Standalone peer package; no runtime import, peers only over its own UDP protocol |

Upper-layer task algorithms belong in [agents/](../agents/README.md), while
robot-independent deployment and device coordination stay in
`src/embodirun`. These are source packages with explicit owners, not copies
of a second Deploy runtime.

An integration's software tests do not authorize a service switch on an
occupied robot. Follow its installation and handoff instructions; the current
device owner must release its resources before another process can take over.
