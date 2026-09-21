"""Make the checkout's optional wired teleoperation package importable for root pytest.

The integration is installed separately, so the core environment neither packages
it nor installs its extras. Keeping the path adjustment local to these tests lets
a root checkout collect them without pulling a hardware SDK into core
development. Anything that would open an arm, a camera or a socket is left to a
real node rather than mocked here.
"""

import sys
from pathlib import Path

INTEGRATION_SRC = Path(__file__).parents[2] / "integrations" / "so101_wired_teleop" / "src"
if str(INTEGRATION_SRC) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_SRC))
