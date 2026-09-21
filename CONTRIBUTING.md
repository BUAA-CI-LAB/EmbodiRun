# Contributing to EmbodiRun

Thanks for your interest in EmbodiRun. This document covers the development
setup, what a pull request must satisfy, and how to report problems. It is kept
in English so that there is a single authoritative version.

## Code of Conduct

This project follows the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating,
you are expected to uphold it. Report unacceptable behavior to
**cclonelycc@outlook.com**; reports are handled privately. For security
problems, follow [`SECURITY.md`](SECURITY.md) instead of opening a public
issue.

## License of contributions

EmbodiRun is licensed under Apache-2.0. See [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). By
submitting a pull request you agree that your contribution is provided under
the same license, as described in section 5 of the license. Do not contribute
code you cannot license this way, and do not paste third-party code without its
license and attribution.

## Development setup

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
uv run pytest -q
```

The core suite runs on CPU. Tests that need a GPU, a checkpoint, sglang, or
robot hardware skip themselves with an explicit reason. To run a grouped path,
install the matching group, for example
`uv sync --frozen --dev --group robot-so101`.

## Before opening a pull request

1. Run the checks and make sure they pass:
   ```bash
   uv run ruff check src tests
   uv run ruff format --check src tests
   uv run pytest -q
   ```
2. Keep the process boundaries intact:
   - `client`, `deployment`, `application`, `devices`, `model_services` are the
     canonical domains; `services.*` stays a compatibility layer.
   - `robots/` and `simulators/` hold hardware adapters; `bindings/` holds
     policy-to-robot mappings. Do not merge the two.
   - Do not import inference-engine runtime code into `src/embodirun`; talk to
     it over the versioned API.
3. Keep robot-specific dependencies optional and declared in `pyproject.toml`.
4. Do not commit checkpoints, datasets, recordings, or credentials.

## Adding a combination

If you add a robot, simulator, model, or backend combination, update
[`docs/en/support-matrix.md`](docs/en/support-matrix.md) with the versions,
configuration, hardware, checkpoint, exact command, and observed result. Do not
mark a combination **Tested** based only on code presence.

## Reporting issues

Use [GitHub Issues](https://github.com/BUAA-CI-LAB/EmbodiRun/issues) for bugs,
documentation gaps, and support questions. Include the EmbodiRun revision, the
deployment YAML shape (redact addresses and secrets), the exact command, the
observed result, and whether hardware was involved. Never paste tokens,
credentials, or personal data into an issue.

**Do not open a public issue for a security problem.** Follow
[`SECURITY.md`](SECURITY.md) instead.

## Documentation

User documentation has parallel English and Simplified Chinese trees: `docs/en/`
and `docs/zh/`. `mkdocs.yml` builds `docs/en/` and `mkdocs.zh.yml` builds
`docs/zh/`; the shared `docs/assets/`, `docs/stylesheets/`, and
`docs/requirements.txt` stay at the `docs/` root. Keep the English `README.md`
and Chinese `README.zh-CN.md` in sync when you change positioning, install
steps, or the support matrix.

The two navigation trees in `mkdocs.yml` and `mkdocs.zh.yml` mirror each other:
when you add or rename a page, add the matching entry to both and translate the
page in the same change.

### Simplified Chinese documentation

Read the Docs serves the Chinese site as the `embodirun-zh` project, a
**translation** of `embodirun`, from this same repository and branch: the
language prefix and the flyout menu come from that relationship, not from the
build.

`embodirun-zh` builds the root `.readthedocs.yaml`, the same file as
`embodirun`. That file selects `mkdocs.zh.yml` from `READTHEDOCS_LANGUAGE`,
which Read the Docs sets from the project language, so no custom *Build
configuration file* path is needed. Create the project once from the Read the
Docs dashboard, set its language to **Simplified Chinese** (`zh-cn`), and add
it as a translation of `embodirun`.

Conventions:

- **English is canonical.** If a Chinese page and an English page disagree, the
  English one is correct and the Chinese one is a bug.
- **Translate deliberately.** These pages carry scoped claims — what a result
  does and does not establish — and a translation that smooths those into
  confident statements is worse than no translation. Do not add, drop, or
  strengthen a claim.
- **Keep relative links relative.** The Chinese tree mirrors the English tree
  path for path, so a link such as `quickstart.md` or `demos/x.md` resolves in
  both. Only external links stay absolute.
- **Keep cross-page anchors stable.** When a translated heading is the target of
  a `page.md#anchor` link, keep the English slug with an attr_list id, e.g.
  `## 优化 profile {#optimization-profiles}`.
- **Governance and legal pages stay English.** `contributing.md`,
  `code-of-conduct.md`, and `license.md` exist only under `docs/en/`. The Chinese
  navigation links to the English pages; do not add translated copies.
- `docs/en/assets`, `docs/en/stylesheets`, `docs/zh/assets`, and
  `docs/zh/stylesheets` are symlinks to the shared `docs/assets/` and
  `docs/stylesheets/`. `extra_css` cannot point outside `docs_dir`, and MkDocs
  silently emits a `<link>` without copying the file when it does. Keep symlinks
  enabled in your checkout.

The governance documents — this file, [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md),
and [`SECURITY.md`](SECURITY.md) — are maintained in English only so that there
is one authoritative text. The Chinese README links to them.
