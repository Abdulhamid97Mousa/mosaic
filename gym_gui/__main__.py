"""Entry point for gym_gui module.

This module ensures Qt API is set before any imports happen.
"""

# CRITICAL: Set warning filters FIRST, before any imports
import warnings
import os
import sys

# Suppress NumPy compatibility warning from Keras/TensorFlow (overcooked_ai dependency)
# The warning is raised by NumPy when keras accesses deprecated np.object attribute
# Must be set before ANY imports that might trigger keras/tensorflow imports
warnings.filterwarnings("ignore", message=r".*np\.object.*", category=FutureWarning)

# Suppress ALL old gym library warnings (required by: baba-is-ai, gym-multigrid, procgen, overcooked)
# The gym deprecation message is printed at import time, not via warnings module
os.environ['GYM_IGNORE_DEPRECATION_WARNING'] = '1'
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

# Suppress gymnasium environment override warnings from vendor libraries
# (minigrid/babyai environments get registered multiple times)
warnings.filterwarnings("ignore", message=r".*Overriding environment.*already in registry.*")

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

