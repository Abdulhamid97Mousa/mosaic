import unittest
from pathlib import Path

import yaml


class DockerComposeSchemaTests(unittest.TestCase):
    def test_ejabberd_service_structure(self) -> None:
        compose_path = (
            Path(__file__).resolve().parents[1] / "infrastructure" / "docker-compose.yaml"
        )
        document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

        self.assertIn("services", document)
        ejabberd = document["services"].get("ejabberd")
        self.assertIsInstance(ejabberd, dict)

        expected_ports = {"5222:5222", "5280:5280", "5443:5443"}
        self.assertEqual(set(ejabberd.get("ports", [])), expected_ports)

        env = {entry.split("=", 1)[0]: entry.split("=", 1)[1] for entry in ejabberd["environment"]}
        self.assertEqual(env.get("EJABBERD_MACRO_HOST"), "localhost")
        self.assertEqual(env.get("CTL_ON_CREATE"), "register agent localhost secret")

        healthcheck = ejabberd.get("healthcheck", {})
        self.assertEqual(healthcheck.get("test"), ["CMD", "ejabberdctl", "status"])
        self.assertEqual(healthcheck.get("interval"), "10s")
        self.assertEqual(healthcheck.get("timeout"), "5s")
        self.assertEqual(healthcheck.get("retries"), 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
