# asyncio subprocess PIPE deadlock discussion

Source: [https://discuss.python.org/t/details-of-process-wait-deadlock/69481](https://discuss.python.org/t/details-of-process-wait-deadlock/69481)

Key takeaways:
- `asyncio.subprocess.Process.wait()` can block indefinitely if stdout/stderr pipes are full, even after the child receives a termination signal.
- Always drain or redirect PIPE outputs (e.g., via `communicate()` or background readers) to prevent log backpressure from freezing supervisory processes.
- Demonstrates a minimal Linux repro where failing to read stdout keeps `wait()` blocked until the buffer is emptied.
