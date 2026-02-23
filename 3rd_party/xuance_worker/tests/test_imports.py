"""Basic import tests for xuance_worker package.

These tests validate that the package and its dependencies are importable.
"""

from __future__ import annotations


def test_xuance_worker_package_imports() -> None:
    """Test that xuance_worker package is importable."""
    import xuance_worker

    assert hasattr(xuance_worker, "__version__")
    assert xuance_worker.__version__ == "0.1.0"


def test_xuance_worker_config_import() -> None:
    """Test that XuanCeWorkerConfig is importable."""
    from xuance_worker import XuanCeWorkerConfig

    assert XuanCeWorkerConfig is not None


def test_xuance_worker_runtime_import() -> None:
    """Test that XuanCeWorkerRuntime is importable."""
    from xuance_worker import XuanCeWorkerRuntime

    assert XuanCeWorkerRuntime is not None


def test_xuance_worker_summary_import() -> None:
    """Test that XuanCeRuntimeSummary is importable."""
    from xuance_worker import XuanCeRuntimeSummary

    assert XuanCeRuntimeSummary is not None


def test_xuance_worker_main_import() -> None:
    """Test that main function is importable."""
    from xuance_worker import main

    assert callable(main)


def test_algorithm_registry_imports() -> None:
    """Test that algorithm registry components are importable."""
    from xuance_worker.algorithm_registry import (
        Backend,
        Paradigm,
        AlgorithmInfo,
        get_algorithms,
        get_algorithm_info,
        get_algorithms_for_backend,
        get_algorithms_for_paradigm,
        get_algorithm_choices,
        get_algorithms_by_category,
        is_algorithm_available,
        get_backend_summary,
    )

    assert Backend is not None
    assert Paradigm is not None
    assert AlgorithmInfo is not None
    assert callable(get_algorithms)
    assert callable(get_algorithm_info)
    assert callable(get_algorithms_for_backend)
    assert callable(get_algorithms_for_paradigm)
    assert callable(get_algorithm_choices)
    assert callable(get_algorithms_by_category)
    assert callable(is_algorithm_available)
    assert callable(get_backend_summary)


def test_cli_imports() -> None:
    """Test that CLI functions are importable."""
    from xuance_worker.cli import parse_args, main

    assert callable(parse_args)
    assert callable(main)


def test_config_module_imports() -> None:
    """Test that config module is importable."""
    from xuance_worker.config import XuanCeWorkerConfig

    assert XuanCeWorkerConfig is not None


def test_runtime_module_imports() -> None:
    """Test that runtime module is importable."""
    from xuance_worker.runtime import (
        XuanCeWorkerRuntime,
        XuanCeRuntimeSummary,
    )

    assert XuanCeWorkerRuntime is not None
    assert XuanCeRuntimeSummary is not None
