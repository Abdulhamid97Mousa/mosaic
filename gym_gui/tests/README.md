# gym_gui Tests

Unit and integration tests for the `gym_gui` package.

## Structure

- **Unit Tests**: Test individual components in isolation
  - `test_trainer_client.py` - TrainerClient async gRPC operations
  - `test_telemetry_service.py` - Telemetry storage and streaming
  - More unit tests to be added as needed

- **Integration Tests**: Test component interactions
  - Telemetry pipeline (hub → bridge → GUI)
  - Service bootstrap and dependency injection
  - UI widget integration

## Running Tests

```bash
# Run all gym_gui tests
pytest gym_gui/tests/

# Run specific test file
pytest gym_gui/tests/test_trainer_client.py

# Run with coverage
pytest gym_gui/tests/ --cov=gym_gui --cov-report=html
```

## Test Requirements

Tests may require:
- Running trainer daemon (`python -m gym_gui.services.trainer_daemon`)
- Mock gRPC services for isolated unit tests
- Qt test fixtures for UI components (use `QT_QPA_PLATFORM=offscreen`)

## Adding New Tests

When adding functionality to `gym_gui`, add corresponding tests:
1. Unit tests for core logic (services, controllers, adapters)
2. Integration tests for multi-component features
3. UI tests for widgets (use pytest-qt)
