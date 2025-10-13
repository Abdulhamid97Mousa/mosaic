# Runtime Data Assets

This directory now only stores **packaged assets** that ship with the
application (for example the text grids used by the toy-text adapters).

Writable runtime artifacts have moved to `gym_gui/var/` so that generated
episodes, telemetry databases, caches, and temporary files no longer mix with
checked-in assets. If you still see older `episodes/` or `telemetry/`
directories here, you can safely remove them after migrating any data you wish
to keep.

Sub-directories:

- `toy_text/` – Static snapshots bundled for FrozenLake, CliffWalking, Taxi, etc.
