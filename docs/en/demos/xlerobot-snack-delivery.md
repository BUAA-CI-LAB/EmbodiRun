# Fetching a snack with XLeRobot

“Bring me a bag of chips.” An agent turns that request into a trip to the table,
a VLA-guided grasp, and a return journey with the snack. EmbodiRun connects the
task stages to the robot's observations and motion.

<video controls muted playsinline preload="metadata" width="720"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/xlerobot_snack_delivery/xlerobot_snack_delivery_overview.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/xlerobot_snack_delivery/xlerobot_snack_delivery.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*67-second edit showing the request, approach, grasp, and return.*

## From a request to a grasp

The agent coordinates the task; the VLA policy handles the grasp. Both use
EmbodiRun to interact with the robot:

1. **Reach the table.** The agent selects the route and requests base motion.
2. **Pick up the snack.** Camera observations feed the VLA service; EmbodiRun
   validates and executes bounded action segments through the robot owner.
3. **Bring it back.** The task switches from grasping to the return route while
   the gripper holds the bag.

The video pairs the physical scene with task-stage and route overlays, making
the transition between base motion and manipulation easy to follow.

## Build the task

The [snack-delivery example](../xlerobot-snack-delivery.md) connects recorded
routes, an optional RPent/Astra observation review, a π0.5/VLA service, and
operator-confirmed handover. EmbodiInfer serves the model; EmbodiRun manages
observations, action validation, execution, and feedback. The XLeRobot owner
holds the motor and camera connections.

Start with the example's setup guide to configure the owner and inference
service, record routes, and calibrate the grasp and handover on your robot.
Validate the emergency stop and run under operator supervision.

The [example package](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/examples/xlerobot_snack_delivery)
includes a unified manifest, task configuration, and fixture routes. After
`uv sync --frozen`, rehearse the task without hardware:

```bash
bash examples/run.sh examples/xlerobot_snack_delivery/example.yaml dry-run
```

For a calibrated deployment, follow the package README to create
`example.local.yaml`, then use `up --allow-hardware` and, in a second terminal,
`run --allow-hardware`. Delivery events, decisions, and proposals are saved under
`result/`. See [Reproduce the demos](../examples.md) for the shared command format.

- [Example and setup](../xlerobot-snack-delivery.md)
- [Agent execution workflow](../agent-workflow.md)
- [Hardware safety](../safety.md)
