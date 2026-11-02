"""Start the Go2 hardware owner from one configuration file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import OwnerConfig
from .control.sdk_loader import ensure_cyclonedds_library_dir
from .runtime import Go2Owner
from .server import serve


def main(argv=None) -> int:
    """Validate before loading native libraries; never infer motion readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--operator-ready", action="store_true")
    parser.add_argument("--validate", action="store_true", help="check configuration without hardware access")
    args = parser.parse_args(argv)
    try:
        config = OwnerConfig.load(args.config, operator_ready=args.operator_ready)
        if args.validate:
            print(
                json.dumps(
                    {
                        "valid": True,
                        "endpoint": f"http://{config.host}:{config.port}",
                        "camera": config.camera is not None,
                        "motion_enabled": False,
                    }
                )
            )
            return 0
        if config.control.mode == "live":
            ensure_cyclonedds_library_dir(config.control.cyclonedds_lib_dir, reexec_args=argv)
        serve(Go2Owner(config))
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"Go2 owner failed: {error}", file=sys.stderr)
        return 1
    return 0
