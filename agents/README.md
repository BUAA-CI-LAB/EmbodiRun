# Deploy upper-layer algorithms

agents/ is the temporary home for upper-layer task algorithms while the Deploy
device-service boundary is being built. It is intentionally outside
src/embodirun: task planning, proposal review, and business success
decisions belong to Agent/RPent; Deploy owns device access, observations,
adapter mappings, units, limits, control authority, execution, and feedback.

```text
agents/
├── README.md
├── astra_pi05/
│   ├── __init__.py
│   ├── README.md
│   ├── decision.py          # pure proposal/decision contract checks
│   ├── cooperative.py       # one-round public HTTP boundary adapter
│   ├── corrections.py       # injectable robot-specific mapping seam
│   ├── recording.py         # detached JSONL/session evidence
│   ├── reviewer.py          # isolated, fake-testable Astra reviewer
│   └── so101.py             # public media packet/state/action bridge
└── rpent/
    ├── README.md             # public RPent integration entrypoint
    ├── __main__.py           # software-only demo entrypoint
    ├── cooperative.py        # compatibility exports
    ├── session.py            # repeated public-client session
    └── so101_correction.py   # named SO101 geometry/calibration mapper
```

The pure validator in astra_pi05/decision.py checks the real BiSO101 contract:
a 50-step,
12-value-per-row proposal with an explicit degree or range_m100_100 unit, an
Astra-approved prefix of 1-15 steps, or 1-5
constrained correction waypoints. It does not call a model, spawn a reviewer,
open images, import a driver, perform IK, apply limits, or send actions.
Correction waypoints use the existing so101_shoulder_plane and m_deg contract.
The validator never converts a declared proposal unit. The current Astra
packet adapter may require degree-valued proposals, so any producer-side
conversion must remain explicit and outside this helper.
They are not a promise of generic six-dimensional end-effector control.

The upper-layer `agents.rpent.session.PublicCooperativeSession` can obtain one
50x12 proposal from the public `/v1/propose` route, pass it to a supplied Astra
reviewer, execute only the selected prefix or an injected robot-specific
correction, wait for a newer observation, and record each round as detached
JSONL evidence. `BiSO101ActionEncoder` and the structured proposal decoder are
required explicitly for the public dual-arm `left/right` action shape;
`SO101PlanarCorrectionMapper` requires named features, calibration, and a
public control rate. These helpers perform no device I/O. Deploy remains the
owner of current feedback admission, collision/limit enforcement, authority,
stop behavior, and physical validation.

`agents.astra_pi05.reviewer.AstraCodexReviewer` names the required
`gpt-6-astra` model and stages only caller-provided image files. Tests inject a
fake command runner; the software demo never invokes Astra or hardware.

Use astra_pi05/README.md for the algorithm boundary, rpent/README.md for the
external integration pointer, and the migration manifest in MIGRATION.md for
the source-to-owner mapping. A future standalone repository can be created
with ordinary Git when a real remote and ownership decision exist; this task
does not create a submodule.
