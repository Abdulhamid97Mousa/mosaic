from gym_gui.utils.fps_counter import FpsCounter


def test_fps_counter_basic():
    c = FpsCounter(window_s=1.0)
    t = 100.0
    # No FPS until two samples
    assert c.tick(t) == 0.0
    # Simulate 10 Hz for 1 second
    fps_values = []
    for i in range(1, 11):
        t = 100.0 + i * 0.1
        fps_values.append(c.tick(t))
    # Last FPS should be close to 10
    assert abs(fps_values[-1] - 10.0) < 1.0
