# Language-guided navigation with MicroDuck

Give MicroDuck a navigation instruction and watch the observation–inference–action
loop in MuJoCo. This demo brings together ActiveVLN, EmbodiRun's inference
client, and a walking controller, with camera views and execution logs on screen.

<video controls muted playsinline preload="metadata" width="540"
       poster="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/microduck_simulation_one_click_demo.jpg">
  <source src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/microduck_simulation_one_click_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*29-second walkthrough of launch and navigation execution. First-person and
third-person views appear above the terminal log.*

## Read the run as it happens

The instruction in this recording is **“Go to the living room and stop near
the trash bin.”** The display connects each model decision to the robot's motion:

- **First-person view:** the scene from MicroDuck's camera.
- **Third-person view:** the robot turning and walking through the simulation.
- **Execution log:** successive navigation actions, inference time, and distance
  to the target.

ActiveVLN produces navigation actions through the inference service.
EmbodiRun's client manages the episode session; the MicroDuck integration
executes the decoded actions using MPC and an ONNX walking policy.

## Launch your own simulation

After setting up the optional environment, model checkpoint, scene assets,
and local configuration, start the demo with:

```bash
CONFIG=examples/microduck_vln/example.local.yaml
bash examples/run.sh "$CONFIG" validate
bash examples/run.sh "$CONFIG" check
bash examples/run.sh "$CONFIG" run
```

The launcher supports a local GPU or a Slurm allocation. Each run writes
first- and third-person videos and episode metrics under `result/`, alongside
the launcher's `run.json` and log. The local YAML sets asset paths, interpreter,
episodes, seed, and resource selection; start from the shipped `example.yaml`.

See [MicroDuck setup and configuration](../microduck-vln.md) for prerequisites,
the reference configuration, and the complete example guide.
