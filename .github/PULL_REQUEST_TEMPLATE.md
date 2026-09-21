<!--
Thanks for contributing. Keep this description short and concrete.
-->

## Summary

<!-- What changes, and why. Link the issue if there is one. -->

## Type of change

- [ ] Bug fix
- [ ] New capability (robot, simulator, model, backend, or runtime feature)
- [ ] Documentation
- [ ] Refactor with no behaviour change
- [ ] Build, packaging, or CI

## Verification

Paste the exact commands you ran and their result. State what you did **not**
verify (GPU, checkpoint, real hardware) rather than implying it works.

```text
$ uv run pytest -q
```

## Process boundaries

- [ ] `client`, `deployment`, `application`, `devices`, `model_services`
      remain the canonical domains, and `services.*` still imports.
- [ ] No inference-engine runtime code was imported into `src/embodirun`.
- [ ] Robot/simulator adapters and policy bindings stay separate.
- [ ] New robot- or model-specific dependencies are optional and declared in
      `pyproject.toml`.

## Checklist

- [ ] I updated [`docs/en/support-matrix.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/support-matrix.md)
      for any new combination, with versions, configuration, hardware,
      checkpoint, exact command, and observed result.
- [ ] I kept the English and Chinese READMEs in sync, if positioning, install
      steps, or the support matrix changed.
- [ ] No checkpoints, datasets, recordings, credentials, addresses, or personal
      data are included.
- [ ] I agree that this contribution is licensed under Apache-2.0, per
      [`CONTRIBUTING.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/CONTRIBUTING.md).
