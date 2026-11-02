# vLLM ActiveVLN action service

This optional package loads ActiveVLN in native **vLLM 0.30.0** and exposes the
versioned session/action HTTP API consumed by EmbodiRun. `provider: vllm` selects
that protocol client. A standard `vllm serve` OpenAI chat endpoint is not an
action service and cannot be used directly.

Only vLLM executes the model. The pinned EmbodiInfer dependency supplies the
checkpoint-specific prompt/parser and the common HTTP server contract; it does
not load another model. EmbodiRun core has no vLLM, Torch or model-parser imports.
Existing EmbodiInfer and SGLang providers keep their current contracts.

## Separate GPU environment

Use a separate Linux Python 3.12 environment compatible with vLLM 0.30.0. From
an EmbodiRun checkout:

```bash
uv venv --python 3.12 .venv-vllm-activevln
uv pip install --python .venv-vllm-activevln/bin/python -e integrations/vllm_activevln
.venv-vllm-activevln/bin/embodirun-vllm-activevln-serve \
  --checkpoint /models/activevln-r2r --device cuda:0 --host 127.0.0.1 --port 8000
```

Prepare the checkpoint separately: `Arvil/Qwen2.5-VL-3B_rl_r2r_4000`, revision
`160987313e3e869705f42400d1b8f28177044518`. The service uses BF16, greedy decoding,
repetition penalty 1.0 and one concurrent inference. It accepts PNG/JPEG RGB,
keeps private history for each episode, and only commits history after successful
generation. Reset drops that history and changes its native cache identity.
It rejects out-of-order turns, instruction changes, oversized images, exhausted
image history and partial execution-prefix requests. It does not advertise RTC.

`--max-context` and `--max-new-tokens` bound generation. `--engine-options FILE`
accepts a native vLLM JSON options object; model identity and serialized scheduling
cannot be overridden. It can select native compilation/speculation settings, but
those settings require GPU validation in the chosen environment. No benchmark
runtime or monkey patches are imported. The packaged default is not claimed to
reproduce the tuned demo's timing.

For a non-loopback listener, export a token of at least 32 characters as
`EMBODIRUN_VLLM_TOKEN` and provide the same token to the client. Do not put secrets
in a committed YAML file. `--token-env` changes the environment variable name.

## EmbodiRun configuration

The [Habitat example](../../examples/activevln_engines/README.md) connects an
existing service with `provider: vllm`, `service: external` and HTTP transport.
For managed service startup, set a model node, `source` checkpoint path,
`environment: .venv-vllm-activevln`, `environment_packages: [vllm==0.30.0]`,
`gpu: cuda:0` and a server listener. Host adds this optional package from the
EmbodiRun source checkout. Use `server_args` for the service options above.

The packaged entrypoint has not been rerun on a GPU; the demo records the earlier
measured experimental adapters.
