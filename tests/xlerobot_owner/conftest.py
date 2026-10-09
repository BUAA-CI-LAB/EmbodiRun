"""Make the checkout's optional owner package importable for root pytest.

The owner is an optional integration, so the core environment does not install
its dependencies or package it as part of ``embodirun``.  Keeping this path
adjustment local to the owner tests lets a root checkout run collection after
installing the selected owner extra without pulling hardware SDKs into core
development.  Tests still use ``importorskip`` at the boundaries that need an
optional runtime such as aiohttp or aiortc.
"""

import sys
from pathlib import Path

OWNER_SRC = Path(__file__).parents[2] / "integrations" / "xlerobot_owner" / "src"
if str(OWNER_SRC) not in sys.path:
    sys.path.insert(0, str(OWNER_SRC))
