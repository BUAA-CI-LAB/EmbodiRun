# Agent client boundary

`embodirun.client.ControlClient` is a dependency-free wrapper for the
existing versioned Control HTTP routes.  It carries caller/session identity on
each request and exposes:

- `describe()` — capabilities plus the configured `binding` section (`kind`,
  `maximum_chunk_steps`, `action_feature_names`) so a caller can build a
  bounded `execute` request without importing robot or model packages
- `observe()` and `media()` for shared observations
- `propose()` for one non-executing inference step against a retained
  `observation_id`
- `execute()` for one already mapped, bounded action segment
- `inspect()`, `cancel()`, and `stop()` for the original request ID
- `manual_acquire()`, `manual_deadman()`, and `manual_release()` for the
  existing explicitly scoped manual takeover lease

The client does not import Control internals, open a robot connection, retry an
execute after a transport failure, or map a waypoint/vector to joints.  A
transport error after submission is reported as an unknown outcome; callers
must inspect the same request ID before deciding what to do.

`propose()` is one model step over the exact retained shared snapshot. It
returns both the raw policy result and the binding-mapped `RobotAction`
payloads, then closes only its temporary inference session. It does not
prepare a device, capture a new camera frame, submit an action, or create a
job. `/v1/tasks` remains the legacy model-inference-and-execution path and is
not used as an Agent proposal API. RPent or another upper-layer application
supplies a proposal and reviewer to `agents.astra_pi05.CooperativeLoop`; the
loop then validates the existing Astra decision contract, executes only the
selected prefix or an explicitly mapped correction, and obtains a new
observation. It drops the unselected proposal rows and never claims
business-task success from an HTTP receipt.

Correction waypoints require a robot-specific mapper and declared
`so101_shoulder_plane`/`m_deg` semantics.  Without one, the loop returns
`unsupported`; it does not invent generic 6D inverse kinematics.

For a source checkout, install this repository with `uv sync --frozen` and
use `PYTHONPATH=.:src` for a source run.  RPent is a separate dependency with
its own remote and installation instructions; this repository does not create
a submodule or assume a local checkout path.
