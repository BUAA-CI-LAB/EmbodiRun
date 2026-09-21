# Algorithm migration manifest

This manifest keeps the real prototype responsibilities visible without
copying its old runtime dependencies into Deploy. Current owner describes
where the behavior exists today; Deploy boundary describes the interface that
a later integration may use.

| Current source | Current owner | Algorithm responsibility | Deploy boundary / status |
| --- | --- | --- | --- |
| RPent-deploy-integration/robots/mobile_cart/cooperative.py | RPent mobile-cart application | Observe -> pi proposal -> Astra decision -> short segment -> reobserve; freshness/evidence policy | `agents/rpent/session.py` and `agents/astra_pi05/cooperative.py` retain the public-client ordering and fresh-observation result semantics without the old runtime |
| RPent-deploy-integration/robots/mobile_cart/astra_reviewer.py | RPent reviewer service | Stage three camera inputs, request Astra structured output, validate identity and decision branch | `agents/astra_pi05/reviewer.py` retains the isolated `gpt-6-astra` structured review seam; caller supplies media paths and tests inject a fake command runner |
| RPent-deploy-integration/robots/mobile_cart/cooperative_observation.py | RPent observation bridge | Decode canonical XLeRobot observations, map camera roles, emit software image artifacts | `agents/astra_pi05/so101.py` consumes the public `Observation` plus `ControlClient.media`; the old direct bridge remains external |
| RPent-deploy-integration/robots/mobile_cart/so101_review.py | RPent SO101 reviewer | Define the SO101 camera/state/trajectory packet and correction schema | `SO101ReviewPacketBuilder`, `extract_bi_so101_state`, and the explicit `BiSO101ActionEncoder` preserve the public packet shape; old reviewer process remains external |
| RPent-deploy-integration/robots/mobile_cart/so101_live_trial.py | RPent SO101 trial | Convert reviewed waypoints to timed hardware commands and run a live trial | `agents/rpent/so101_correction.py` provides bounded geometry/calibration mapping into public actions; live device trial and feedback stay Deploy-owned/external |
| RPent-deploy-integration/robots/mobile_cart/cooperative_demo.py | RPent replay fixture | Software-only replay and fake executor | Remains an external replay fixture; no duplicate Deploy loop is created |
| RPent-deploy-integration/tests/test_cooperative.py and test_astra_reviewer.py | RPent tests | Verify the prototype full loop, reviewer process, and cleanup | Focused fake-client/fake-runner coverage belongs in `tests/test_astra_session.py`; hardware and managed-runtime tests remain external |
| EmbodiRun-managed-runtime/docs/astra-pi05-implementation.md | Managed-runtime design/evidence | Records proposed lifecycle, 50x12 output, and review/feedback limits | Reference only; do not import or copy embedded_runtime code |
| src/embodirun/model_services/* and bindings/lerobot/so101/pi05 | Deploy | Inference transport and policy-to-robot action mapping | Remains Deploy-owned; model math stays in Inference, and action units are enforced by binding/runtime |
| src/embodirun/application/*, devices/* and robots/* | Deploy | Control authority, robot/sensor lifecycle, execution, stop, and feedback | Public Agent calls enter this boundary through the Control HTTP service |

The prototype is specifically BiSO101 dual-arm 50x12 with
biso101_so101_v1 / left6_right6 absolute actions. The correction contract is
nominal so101_shoulder_plane m_deg waypoints. Do not generalize either to
arbitrary 6D actions or claim that Deploy currently supports generic EE
corrections.

Software integration covered in this branch:

1. Keep the pure validator and its tests software-only.
2. Complete shared device ownership and observations (#22-#23).
3. Expose the existing Deploy control/feedback semantics through the Agent API
   (#24-#25).
4. Add a RPent client adapter that submits validated decisions; keep waypoint
   conversion in a robot-specific adapter and reject unsupported capability.
5. Run fake replay and existing regression suites, then track real hardware
   validation separately.

These paths have software coverage through a real local HTTP service with
fake model and robot endpoints. Physical execution, real model output and
paid Astra inference remain separate validation work. The implementation
does not replace the complete RPent project or create a second hardware owner.

## Intentionally external experiments

These source files were inspected but are not migrated because they own
physical devices, direct camera capture, SSH hops, or an old managed runtime.
They remain in the original RPent worktree and are not evidence that this
Deploy worktree supports a physical trial:

| Source file | Reason it remains external |
| --- | --- |
| `robots/mobile_cart/cooperative_observation.py` | Direct observation/image bridge; requires the Deploy shared observation/media adapter |
| `robots/mobile_cart/cooperative_demo.py` | Imports `embodied_runtime` simulation/runtime classes; this worktree has its own public HTTP demo |
| `robots/mobile_cart/so101_live_trial.py` and `so101_trial.py` | Direct robot/camera trial ownership and action execution; must be replaced by public Agent calls plus a named robot adapter |
| `robots/mobile_cart/so101_session.py` and `so101_ssh.py` | Remote SSH/NX lifecycle and decision-file transport; public Agent session has no SSH transport |
| `robots/mobile_cart/so101_recording.py` | Owns a camera capture thread and ffmpeg process; only detached public-media bytes may be recorded here |
| `robots/mobile_cart/so101_review.py` and `navigation_review.py` | Robot/navigation-specific review packet and physical-trial assumptions; no generic Agent contract exists yet |
| `robots/mobile_cart/so101_session_view.html` | Old session UI coupled to the SSH/session runner; no public Deploy dashboard contract has been migrated |
