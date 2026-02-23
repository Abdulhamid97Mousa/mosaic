from __future__ import annotations

def test_cleanrl_worker_imports_project_root() -> None:
    import cleanrl_worker  # noqa: F401

    # If import succeeded, gym_gui should now be importable because __init__ inserted
    # the repo root ahead of time.
    import gym_gui  # noqa: F401

    assert True
