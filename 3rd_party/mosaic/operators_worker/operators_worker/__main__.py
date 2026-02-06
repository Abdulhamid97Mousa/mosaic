"""Allow running operators_worker as a module: python -m operators_worker"""

import sys
from operators_worker.cli import main

if __name__ == "__main__":
    sys.exit(main())
