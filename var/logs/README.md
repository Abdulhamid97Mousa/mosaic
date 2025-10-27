# Runtime Log Output

This directory is created at runtime to collect application log files. The
contents are transient and should not be committed to version control.

- `gym_gui.log` – Main application log written when file logging is enabled.
- `*_debug.log` – Additional debug traces created by smoke harnesses or tests.

It is safe to delete everything here when you need a clean slate.
