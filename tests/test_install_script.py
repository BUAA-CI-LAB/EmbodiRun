import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "requirements" / "install.sh"


def run_installer(*arguments: str, home: Path) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "HOME": str(home)}
    return subprocess.run(
        ["bash", str(INSTALLER), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_installs_repeatable_project_extras(tmp_path: Path) -> None:
    venv = tmp_path / "envs" / "venv" / "deploy"
    result = run_installer(
        "--venv",
        str(venv),
        "--python",
        "3.12",
        "--extra-deps",
        "so101,opencv",
        "--extra-deps",
        "test",
        "--pytorch",
        "skip",
        "--dry-run",
        home=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert f"{ROOT}\\[so101\\,opencv\\,test\\]" in result.stdout
    assert not venv.exists()


def test_unknown_component_fails_before_writing(tmp_path: Path) -> None:
    result = run_installer(
        "--venv",
        str(tmp_path / "venv"),
        "--python",
        "3.12",
        "--extra-deps",
        "unknown",
        "--dry-run",
        home=tmp_path,
    )
    assert result.returncode == 2
    assert "unknown --extra-deps component 'unknown'" in result.stderr


def test_default_pytorch_index_uses_uv_backend_selection(tmp_path: Path) -> None:
    result = run_installer(
        "--venv",
        str(tmp_path / "venv"),
        "--python",
        "3.12",
        "--pytorch",
        "auto",
        "--pytorch-index",
        "default",
        "--dry-run",
        home=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "requirements/pytorch.txt" in result.stdout
    assert "--torch-backend auto" in result.stdout
    assert "pypi.jetson-ai-lab.io" not in result.stdout


def test_custom_pytorch_index_overrides_source_selection(tmp_path: Path) -> None:
    result = run_installer(
        "--venv",
        str(tmp_path / "venv"),
        "--python",
        "3.12",
        "--pytorch",
        "auto",
        "--pytorch-index",
        "https://packages.example.test/pytorch",
        "--dry-run",
        home=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "requirements/pytorch.txt" in result.stdout
    assert "https://packages.example.test/pytorch" in result.stdout
    assert "--torch-backend" not in result.stdout


def test_invalid_pytorch_index_fails_before_writing(tmp_path: Path) -> None:
    result = run_installer(
        "--venv",
        str(tmp_path / "venv"),
        "--python",
        "3.12",
        "--pytorch-index",
        "not-a-url",
        "--dry-run",
        home=tmp_path,
    )
    assert result.returncode == 2
    assert "--pytorch-index must be auto, default, or an HTTP(S) URL" in result.stderr


def test_recreate_rejects_environment_root(tmp_path: Path) -> None:
    environment_root = tmp_path / "envs"
    environment_root.mkdir()
    result = run_installer(
        "--venv",
        str(environment_root),
        "--python",
        "3.12",
        "--recreate",
        "--dry-run",
        home=tmp_path,
    )
    assert result.returncode == 2
    assert "refusing to recreate broad path" in result.stderr
