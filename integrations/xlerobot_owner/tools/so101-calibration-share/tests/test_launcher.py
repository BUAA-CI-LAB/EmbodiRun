import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("so101_launcher", ROOT / "launch.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def config(mode="ssh"):
    arms = {f"arm{i}": {"id": f"arm{i}", "port": f"/dev/ttyUSB{i}", "type": "so101_follower"} for i in range(1, 5)}
    arms["arm3"]["type"] = "so101_leader"
    return {
        "mode": mode,
        "host": "192.168.1.100",
        "user": "robot",
        "ssh_port": 22,
        "identity_file": "",
        "python": "/opt/my python/bin/python",
        "sdk_src": "/opt/sdk src",
        "calibration_dir": "/opt/calibrations",
        "arms": arms,
    }


def test_validate_four_unique_roles_and_leader_args():
    cfg = config()
    item = mod.validate(cfg, "arm3")
    assert item["id"] == "arm3"
    assert mod.calibration_args(item)[2] == "--teleop.type=so101_leader"
    assert {mod.validate(cfg, arm)["id"] for arm in mod.ARMS} == set(mod.ARMS)


def test_dry_run_does_not_spawn_and_prints_paths(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("spawned")))
    assert mod.main(["--config", str(path), "--arm", "arm1", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Dry run" in output
    assert "Serial port: /dev/ttyUSB1" in output
    assert "/opt/calibrations/arm1.json" in output


def test_remote_command_roundtrip_preserves_special_port_text():
    cfg = config()
    cfg["arms"]["arm1"]["port"] = "/dev/by id/$(keep this)"
    item, command, _ = mod.make_command(cfg, "arm1")
    remote_words = __import__("shlex").split(command[-1])
    assert remote_words == mod.build_command(item)
    assert "--robot.port=/dev/by id/$(keep this)" in remote_words


def test_placeholder_and_relative_calibration_dir_fail():
    cfg = config()
    cfg["arms"]["arm1"]["port"] = "/dev/serial/by-id/REPLACE_ARM1"
    try:
        mod.validate(cfg, "arm1")
    except ValueError as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("placeholder accepted")
    cfg["arms"]["arm1"]["port"] = "/dev/ttyUSB1"
    cfg["calibration_dir"] = "relative/calibrations"
    try:
        mod.validate(cfg, "arm1")
    except ValueError as exc:
        assert "absolute" in str(exc)
    else:
        raise AssertionError("relative calibration path accepted")
