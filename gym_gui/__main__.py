"""Entry point for gym_gui module.

This module ensures Qt API is set before any imports happen.
"""

import os
import sys
import warnings

# Suppress gymnasium environment override warnings from vendor libraries
# (minigrid/babyai environments get registered multiple times)
warnings.filterwarnings("ignore", message=r".*Overriding environment.*already in registry.*")

# Suppress NumPy compatibility warning from Keras/TensorFlow (required by dm-meltingpot)
# Keras checks for np.object which triggers FutureWarning in NumPy 1.24-1.26
# This is a known compatibility patch in Keras and can be safely ignored
warnings.filterwarnings("ignore", message=r".*np\.object.*", category=FutureWarning)

# Suppress old gym library deprecation warnings
# gym (not gymnasium) is required by: baba-is-ai, gym-multigrid, procgen-mirror
# These packages haven't migrated to gymnasium yet
warnings.filterwarnings("ignore", message=r".*rng\.randint.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*render method.*deprecated.*", category=DeprecationWarning)

# Set Qt API BEFORE any other imports
os.environ.setdefault("QT_API", "PyQt6")

# Now import and run the app
from gym_gui.app import main

if __name__ == "__main__":
    sys.exit(main())

