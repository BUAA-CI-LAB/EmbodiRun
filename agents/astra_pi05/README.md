# Astra x pi0.5 decision boundary

The external prototype is an RPent mobile-cart application loop. It receives one
BiSO101 proposal with a fixed H=50, D=12 action shape, asks the specified
`gpt-6-astra` reviewer to choose one bounded branch, and then submits a short
segment through Deploy's public Agent client. The algorithm layer owns the
decision semantics; Deploy owns device state, action mapping, limits, authority,
and feedback.

Run `PYTHONPATH=.:src python -m agents.astra_pi05 --demo` for a complete
hardware-free public HTTP example. It starts a local fake action service and
Control server, uses a demo-only simulated twelve-joint binding, runs
observe -> propose -> reviewer prefix 1..15 -> execute -> fresh observe, and
prints the before/after simulated state. Use
`PYTHONPATH=.:src python -m agents.astra_pi05 --validate-only` for the pure
decision check without starting services.

`agents.rpent.PublicCooperativeSession` is the repeated entrypoint. It uses
`ControlClient.observe()` -> `ControlClient.propose()` -> Astra review ->
`ControlClient.execute()` -> a newer `ControlClient.observe()` for each round,
and can write detached events with `SessionRecorder`. The convenience command
`PYTHONPATH=.:src python -m agents.rpent --demo` runs the same local fake
Control/model demo and never touches hardware.

The fake model, simulated robot, and local HTTP services exist only to make
the public API calls observable. This command does not load a real Astra
checkpoint, call a real model service, or access hardware; it demonstrates the
same API composition that an external RPent installation can consume.

The small decision.py module contains only pure checks. The cooperative.py
adapter is the Deploy-side Astra/pi0.5 boundary; RPent supplies the proposal
and reviewer, or `PublicCooperativeSession` obtains a proposal through the
public Deploy route:

- proposal rows are finite, absolute BiSO101 SO101 actions with
  action_layout=left6_right6, action_dim=12, exactly 50 rows, and an explicit
  degrees or range_m100_100 unit;
- execute_prefix accepts 1-15 rows and no corrections;
- correct accepts 1-5 correction waypoints and no prefix rows;
- hold accepts no action;
- correction waypoints declare frame=so101_shoulder_plane, units=m_deg, a
  positive duration, and at least one acting arm.

For a configured `lerobot.bi_so101` public binding, pass
`BiSO101ActionEncoder` to `CooperativeLoop` or `PublicCooperativeSession`.
That explicit adapter encodes the flat 12-value review row into the public
`{type: joint_position, left, right}` action shape. The generic loop never
guesses this shape. A `SO101PlanarCorrectionMapper` likewise requires an
explicit `control_hz`; each reviewer waypoint must equal one public executor
step (`duration_s == 1 / control_hz`). `ControlClient.execute` schedules by
`control_hz`, so arbitrary waypoint durations are rejected rather than being
treated as physical motion timing.

`decision.py` remains pure. `cooperative.py` provides the one-round ordering
adapter; `PublicCooperativeSession` repeats it and records the result. Deploy's
`/v1/propose` endpoint performs one
non-executing inference step on an already retained observation and returns
binding-mapped RobotAction proposals. The cooperative adapter may submit the
reviewer's selected prefix through the existing public `/v1/execute` route and
wait for a newer observation; that receipt is execution evidence only, not
business-task success. Neither layer converts units (including
range_m100_100 to degrees), solves generic IK, or silently enforces hardware
limits. A robot-specific `CorrectionMapper` must be injected before a
correction can be executed; missing mapping returns `unsupported` without
sending actions.

num_steps remains a model denoising/inference option. It is not the 50-step
proposal horizon and is not the number of rows accepted for execution. After
a segment, the adapter reports the discarded unexecuted tail and waits a
bounded interval for a fresh observation. If no newer snapshot arrives it
returns `post_observation_pending` instead of reusing the old snapshot.
