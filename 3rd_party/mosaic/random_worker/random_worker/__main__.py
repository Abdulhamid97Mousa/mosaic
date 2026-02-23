"""Allow running as: python -m random_worker"""

import sys

from random_worker.cli import main

if __name__ == "__main__":
    sys.exit(main())
