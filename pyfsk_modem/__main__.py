"""PyFSK-Modem package CLI execution entry point."""

from __future__ import annotations

import sys
from pyfsk_modem.cli import main

if __name__ == "__main__":
    sys.exit(main())
