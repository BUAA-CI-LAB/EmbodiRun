# Manual control

The control service keeps model execution and operator intent on the same robot
command boundary. Model chunks are submitted by the task runtime. Manual input
submits CPU-only intent actions, and the service-side resolver converts those
intents to robot actions inside the arbiter worker.

Supported HTTP control routes:

- `GET /v1/control` returns runtime authority, active task state, and
  `last_error`.
- `POST /v1/control/emergency-stop` latches emergency stop outside the model
  queue.
- `POST /v1/control/reset` clears the software emergency latch.
- `POST /v1/control/manual/acquire` gives manual control ownership.
- `POST /v1/control/manual/release` returns ownership to model control.
- `POST /v1/control/manual/deadman` accepts `{"active": true}` or
  `{"active": false}`.
- `POST /v1/control/manual/action` accepts a `RobotAction` JSON object with
  `values` and `metadata`; asynchronous action submission returns ticket state.

Keyboard polling is always enabled by the HTTP teleop CLI:
`python -m embodirun.services.control.teleop --endpoint http://127.0.0.1:8100`.
Pass `--joystick /dev/input/js0` to add joystick motion intent.
`--robot-kind arx5` or `--robot-kind so101` selects the namespace for joystick
axes, and is required when `--joystick` is present. If omitted, keyboard-only
emergency/reset/quit still works and no robot mapper is selected.

Polling is non-blocking with respect to robot observation: joystick axes create
CPU-only intent actions, then the service resolver maps those intents on the
robot command worker. The HTTP status/action calls themselves can still block
on loopback I/O and are not a hard real-time safety path. The ARX5 input mapping
has software tests; verify it on your controller before enabling motion.

## Authority and lifecycle

Each service owns one configured robot and one arbiter; stopping one robot does
not stop other services. Model actions use a bounded FIFO. Manual actions have
one pending slot (latest wins). Acquiring manual control cancels the model task,
including inference results that arrive after takeover. Releasing manual control
allows a new model task, but never resumes the cancelled task.

Emergency stop cancels active work, clears both pending paths, and latches outside
the queue. Reset on a connected robot returns to manual authority with the
deadman released; the operator must explicitly re-enable motion. Reset before
any robot connection only clears the startup latch.

Task completion calls the adapter's hold/stop operation but keeps the robot
connected. Service shutdown closes it. Neither operation plans a return-to-home
trajectory. Model bindings only map observations and actions; keyboard/joystick
input is independent of those bindings. Robot-specific axis-to-motion mapping
lives under `robots/arx/x5` and `robots/lerobot/so101`.

## Controls and limits

- Keyboard: Space stops, R resets, Q holds and exits the input client.
- Linux joystick: button 0 acquires manual control, button 1 stops,
  button 3 releases manual control, button 4 is the held deadman; reset uses R.
  Check button numbering on the actual controller before enabling motion.
- Missing manual heartbeat for 0.25 seconds releases the deadman and holds;
  input EOF or an input error also requests a hold. Execution failures appear
  in `last_error` and latch emergency stop.

ARX5 and SO101 serialize stop with the current SDK call; a blocked SDK call
therefore delays physical stopping. FR3 explicitly permits concurrent stop.
These are software stop semantics, not a replacement for physical emergency
stop. Hardware validation must confirm button mapping, deadman release, model
takeover, cancellation, stop latency, and reset without automatic motion for
the selected device and controller.
