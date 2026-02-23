import asyncio
import threading
import unittest
from typing import Any
from unittest.mock import patch

from gym_gui.services.trainer.client import TrainerClient


class _DummyChannel:
    def __init__(self) -> None:
        self.closed = False

    async def channel_ready(self) -> None:  # pragma: no cover - deterministic
        return None

    async def close(self) -> None:
        self.closed = True


class _DummyStub:
    pass


class TrainerClientMultiLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_supports_multiple_event_loops(self) -> None:
        client = TrainerClient()
        channels: list[_DummyChannel] = []
        stubs: list[Any] = []  # Use Any to avoid type conflicts with mocks

        def fake_channel(target: str, options: tuple[tuple[str, object], ...]) -> _DummyChannel:
            channel = _DummyChannel()
            channels.append(channel)
            return channel

        def fake_stub(channel: _DummyChannel) -> Any:  # Return Any for mock
            stub = _DummyStub()
            stubs.append(stub)
            return stub

        with (
            patch("gym_gui.services.trainer.client.grpc.aio.insecure_channel", side_effect=fake_channel),
            patch("gym_gui.services.trainer.client.trainer_pb2_grpc.TrainerServiceStub", side_effect=fake_stub),
        ):
            stub_main = await client.ensure_connected()
            loop = asyncio.new_event_loop()
            results: dict[str, Any] = {}  # Use Any for dict values

            def worker() -> None:
                try:
                    results["stub"] = loop.run_until_complete(client.ensure_connected())
                finally:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            thread.join(timeout=5)
            self.assertTrue(thread.is_alive() is False, "Background loop thread did not finish")
            self.assertIn("stub", results)
            stub_other = results["stub"]

        self.assertIsNot(stub_main, stub_other)
        self.assertEqual(len(channels), 2)
        self.assertEqual(len(stubs), 2)

        await client.close()
        self.assertTrue(all(channel.closed for channel in channels))
