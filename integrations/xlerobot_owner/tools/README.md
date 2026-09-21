# XLeRobot owner tools

[`so101-calibration-share/`](so101-calibration-share/README.md) is the maintained
Mac launcher for sequential SO101 calibration, either locally or over SSH.
Its example configuration contains placeholders; calibration data stays on
the selected device and outside this source package. Use `--dry-run` to inspect
the command without opening a device or SSH session.

`validate_quest_export.py` is a portable offline validation helper. Files under
`archive/` preserve lab-specific AGX launchers and storage handoff probes from
the recovered deployment. They are reference artifacts, are not package entry
points, and require the operator to supply the destination checkout, paths, and
credentials; they must not be run automatically or against a robot by install.
