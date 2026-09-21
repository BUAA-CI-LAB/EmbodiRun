import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mount,uuid,override,ok",
    [
        ("/mnt/robot-data", "6610-B9F6", "", True),
        ("/media/root/111", "6610-B9F6", "", True),
        ("", "6610-B9F6", "", False),
        ("/mnt/robot-data\n/second", "6610-B9F6", "", False),
        ("/mnt/robot-data", "wrong-volume", "", False),
        ("/mnt/robot-data", "6610-B9F6", "/mnt/robot-data", True),
    ],
)
def test_disk_mount_resolution_never_falls_back_to_system_disk(mount, uuid, override, ok):
    script = Path(__file__).parents[2] / "integrations" / "xlerobot_owner" / "tools" / "archive" / "dualsense_demo.sh"
    # Exercise only the mount checks, before mkdir/flock/control/recording.
    prefix = script.read_text().split('mkdir -p "$demo_disk/xlerobot-demos"')[0]
    shell = """
findmnt() {
  if [[ "$*" == *TARGET* ]]; then printf '%s\\n' "$TEST_MOUNT";
  else printf '%s\\n' "$TEST_UUID"; fi
}
mountpoint() { [[ "${@: -1}" == "$TEST_MOUNT" && -n "$TEST_MOUNT" ]]; }
"""
    result = subprocess.run(
        ["bash", "-c", shell + prefix + '\nprintf "%s" "$demo_disk"'],
        env={**os.environ, "TEST_MOUNT": mount, "TEST_UUID": uuid, "DEMO_DISK": override},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is ok, result.stderr
    if ok:
        assert result.stdout == mount
