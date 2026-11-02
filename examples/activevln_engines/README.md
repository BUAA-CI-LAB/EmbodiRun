# ActiveVLN with alternative inference engines

EmbodiRun owns Habitat observations, sessions and action execution. A separately
running inference service owns ActiveVLN. The [recorded demo](../../docs/en/demos/activevln-engines.md)
shows EmbodiInfer, SGLang and vLLM connected through the same runtime.

The new `vllm` provider uses the versioned action API. Start its
[optional service](../../integrations/vllm_activevln/README.md) in a Python 3.12
GPU environment, and prepare Habitat/R2R in the existing separate simulator
environment. Checkpoint and scene downloads are not part of this example.

Copy `habitat-vllm.yaml` to `habitat.local.yaml`, set your dataset/scenes paths
and episode, and use a published EmbodiRun revision containing this integration
for `metadata.deploy-commit`. Keep the service endpoint reachable from the
simulator node. The validation command is offline; `init` prepares source/dependencies and `up` starts the simulator service:

```bash
CONFIG=examples/activevln_engines/habitat.local.yaml
embodirun --config "$CONFIG" validate
embodirun --config "$CONFIG" init
embodirun --config "$CONFIG" up
embodirun --config "$CONFIG" run --runtime navigation --max-steps 200 \
  --chunk-steps 72 --request-timeout 120
embodirun --config "$CONFIG" down
```

The simulator binding expands each complete metric action into 25 cm forward
or 15 degree turn primitives; `--chunk-steps 72` allows the full bounded response
to execute. It sends only RGB and the instruction to inference. Goal distance,
pose and evaluator state are not policy input. The 500-step simulator budget
still bounds execution. A STOP response ends the episode; budget exhaustion is
not success. No physical Go2 is controlled by this example.

For an ActiveVLN EmbodiInfer action service, change `provider` to `embodiinfer`
and update the endpoint. Existing SGLang π0.5 examples remain in
`examples/engine_comparison`; the ActiveVLN SGLang panel uses the experiment's
native wrapper, not that π0.5 endpoint. This example packages the vLLM path and
does not change SGLang's existing provider.

The video is one real recorded episode, not a new run produced by this packaging
change. Its timing definitions and SGLang rerun selection are documented on the
demo page. The inference-only benchmark is maintained separately in EmbodiInfer.
