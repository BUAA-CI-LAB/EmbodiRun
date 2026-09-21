"""Software-only RPent/Astra entrypoint.

The default command validates the pure 50x12 Astra decision contract;
``--demo`` runs a local public-HTTP demo.  A real RPent caller should
construct :class:`PublicCooperativeSession` with its own ``ControlClient``
and robot-specific correction mapper; this command never opens a device, SSH
session, or camera.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the local fake model plus public Control HTTP demo",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the pure 50x12 Astra decision contract",
    )
    args = parser.parse_args(argv)
    if args.demo and args.validate_only:
        parser.error("--demo and --validate-only are mutually exclusive")
    from agents.astra_pi05.__main__ import _run_demo, _validate_only

    payload = _run_demo(12) if args.demo else _validate_only()
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
