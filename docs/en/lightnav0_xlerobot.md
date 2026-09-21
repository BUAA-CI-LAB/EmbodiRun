# LightNav-0 with XLeRobot

This experimental binding converts local waypoints into XLeRobot base commands.
It uses an external inference service for predictions and a robot-side HTTP
service for camera access and motion. It currently runs through its own CLI;
Host configuration and unified control arbitration are not yet integrated.

## Dependencies and inference service

This binding requires an external inference service implementing the LightNav-0
contract below. You will need to supply this service: `lightnav0` is not yet
in EmbodiInfer's public model catalog.

```bash
uv sync --frozen --no-dev --group binding-lightnav0
```

This installs NumPy and Pillow only. The external robot service supplies its
device dependencies. Set `--inference-token-env` to the name
of an environment variable when the inference service uses Bearer authentication.

The binding sends one lossless PNG named `observation.images.rgb`, the navigation
instruction and capture time. It sends no simulator pose, map, target coordinate
or privileged navigation hints to the model. `lightnav0.waypoints.v1` responses
must contain ten finite `[forward_m, left_m, ccw_rad]` cumulative local waypoints
and an explicit boolean `stop`. Existing HTTP session/reset/step APIs are reused.

## Connect to the robot service

An already running XLeRobot HTTP service must provide authenticated control
ownership, wheel feedback and fresh timestamped camera frames. This repository
contains its client, not the robot-side HTTP server. Inference readiness is
checked before acquiring the robot control lease.

```bash
.venv/bin/python -m embodirun.bindings.xlerobot.lightnav0.cli \
  --robot-url http://robot-host:8080 --robot-token-env XLEROBOT_TOKEN \
  --inference-url http://inference-host:8050 \
  --inference-token-env INFERENCE_TOKEN \
  --instruction 'go to the door' --camera front --authorize-motion
```

The command explicitly authorizes motion. Without that flag the built-in route
refuses base commands. A custom `--robot-factory module:function` must return an
already connected and authorized robot. Stale camera/feedback, lost ownership,
timeouts and model stop trigger the existing bounded stop/cleanup paths.

## Development status

CPU tests cover local-frame transforms, curvature-preserving velocity limits,
stale observations, authorization, cleanup and HTTP client validation. The
localhost HTTP smoke test uses a fake upstream model. Real-checkpoint HTTP
evaluation, navigation success and physical-robot validation remain outstanding.
