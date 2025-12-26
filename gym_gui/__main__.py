"""Entry point for gym_gui module.

This module ensures Qt API is set before any imports happen.
"""

import os
import sys
import warnings

# Suppress gymnasium environment override warnings from vendor libraries
# (minigrid/babyai environments get registered multiple times)
warnings.filterwarnings("ignore", message=r".*Overriding environment.*already in registry.*")

# Set Qt API BEFORE any other imports
os.environ.setdefault("QT_API", "PyQt6")

# Now import and run the app
from gym_gui.app import main

if __name__ == "__main__":
    sys.exit(main())

