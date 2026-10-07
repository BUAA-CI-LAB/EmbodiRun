# VLA: SO-101 grasping

Give a SO-101 arm the instruction **“Pick up the cube and place it in the bowl.”**
The π0.5 policy predicts actions from front and wrist camera images and joint
state. EmbodiRun collects the observations, requests an action chunk, validates
it, and executes it through the robot adapter.

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Three SO-101 recordings of the cube-to-bowl task, shown side by side.*

## Observation to action

1. Capture front and wrist images together with the six-dimensional joint and
   gripper state.
2. Send the observation and instruction to the π0.5 service.
3. Map the returned action chunk to the robot, check limits, and play 50 steps
   at 20 Hz.
4. Collect the next observation and repeat.

The Recipe uses one arm and an EmbodiInfer HTTP service. The model checkpoint
must match the robot's camera roles, feature names, and action units.

## Run the Recipe

Follow the [SO-101 grasping Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/so101_grasping.md)
to configure the robot, cameras, calibration, checkpoint, and SSH hosts.

Preview the shipped configuration from the repository root:

```bash
uv sync --frozen
bash examples/run.sh examples/engine_comparison/embodiinfer-http.yaml validate
bash examples/run.sh examples/engine_comparison/embodiinfer-http.yaml plan
```

These commands check the configuration and display the launch plan locally.
The Recipe provides the remaining setup, startup, execution, and shutdown
commands. Read [Safety](../safety.md) before enabling hardware motion.

## Extend the setup

- [Share one model service across multiple SO-101 arms](multi-robot-serving.md).
- [Compare inference engines and transports](engine-e2e-contrast.md).
- [Configure a Bi-SO-101 dual-arm policy](../pi05-bi-so101.md).
- [Browse robot and simulator support](../support-matrix.md).
