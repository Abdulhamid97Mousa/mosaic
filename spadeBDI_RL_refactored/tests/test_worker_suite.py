import unittest

from spadeBDI_RL_refactored.tests import test_worker_integration
from spadeBDI_RL_refactored.tests import test_worker_no_matplotlib


class WorkerIntegrationSuite(unittest.TestCase):
    def test_full_training_run(self) -> None:
        test_worker_integration.test_full_training_run()

    def test_worker_dry_run(self) -> None:
        test_worker_no_matplotlib.test_worker_dry_run()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
