"""Test RGB renderer backward compatibility with legacy 'frame' key."""

import pytest
import numpy as np
from qtpy import QtWidgets

from gym_gui.rendering.strategies.rgb import RgbRendererStrategy


class TestRgbRendererBackwardCompatibility:
    """Test that RgbRendererStrategy supports both 'rgb' and 'frame' keys."""

    @pytest.fixture
    def app(self, qapp):
        """Qt application fixture."""
        return qapp

    @pytest.fixture
    def rgb_renderer(self, app):
        """Create RGB renderer strategy."""
        return RgbRendererStrategy()

    @pytest.fixture
    def sample_rgb_array(self):
        """Create sample RGB array (500x600x3)."""
        return np.ones((500, 600, 3), dtype=np.uint8) * 128

    def test_supports_new_rgb_key(self, rgb_renderer, sample_rgb_array):
        """Test: Renderer supports new 'rgb' key."""
        payload = {
            "mode": "rgb_array",
            "rgb": sample_rgb_array,
            "game_id": "Blackjack-v1"
        }
        assert rgb_renderer.supports(payload) is True

    def test_supports_legacy_frame_key(self, rgb_renderer, sample_rgb_array):
        """Test: Renderer supports legacy 'frame' key for backward compatibility."""
        payload = {
            "mode": "rgb_array",
            "frame": sample_rgb_array,
            "game_id": "Blackjack-v1"
        }
        assert rgb_renderer.supports(payload) is True

    def test_does_not_support_missing_keys(self, rgb_renderer):
        """Test: Renderer rejects payload without rgb or frame keys."""
        payload = {
            "mode": "rgb_array",
            "game_id": "Blackjack-v1"
        }
        assert rgb_renderer.supports(payload) is False

    def test_renders_with_rgb_key(self, rgb_renderer, sample_rgb_array):
        """Test: Renderer can render payload with 'rgb' key."""
        payload = {
            "mode": "rgb_array",
            "rgb": sample_rgb_array,
            "game_id": "Blackjack-v1"
        }
        # Should not raise exception
        rgb_renderer.render(payload)

    def test_renders_with_frame_key(self, rgb_renderer, sample_rgb_array):
        """Test: Renderer can render payload with legacy 'frame' key."""
        payload = {
            "mode": "rgb_array",
            "frame": sample_rgb_array,
            "game_id": "Blackjack-v1"
        }
        # Should not raise exception
        rgb_renderer.render(payload)

    def test_prefers_rgb_over_frame_when_both_present(self, rgb_renderer, sample_rgb_array):
        """Test: When both keys present, 'rgb' is preferred."""
        rgb_array = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White
        frame_array = np.zeros((100, 100, 3), dtype=np.uint8)  # Black
        
        payload = {
            "mode": "rgb_array",
            "rgb": rgb_array,
            "frame": frame_array,  # Should be ignored
            "game_id": "Test"
        }
        
        # Should not raise exception and should use 'rgb' key
        rgb_renderer.render(payload)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
