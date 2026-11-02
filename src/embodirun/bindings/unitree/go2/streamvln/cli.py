"""Compatibility CLI alias for ``embodirun go2 edge``."""

from embodirun.deployment.go2.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
