# spadeBDI_RL_refactored Tests

Unit and integration tests for the `spadeBDI_RL_refactored` package.

## Structure

- **Unit Tests**: Test individual components in isolation
  - `test_docker_compose.py` - Docker Compose configuration validation
  - `test_spade_bdi_agent.py` - SPADE-BDI agent lifecycle and XMPP connectivity
  
- **Integration Tests**: Test component interactions and full workflows
  - `test_worker_integration.py` - Full training pipeline (worker → telemetry → completion)
  - `test_worker_no_matplotlib.py` - Verify headless operation without matplotlib dependencies
  - `test_worker_suite.py` - Test suite combining multiple integration tests

## Running Tests

```bash
# Run all spadeBDI tests
pytest spadeBDI_RL_refactored/tests/

# Run specific test file
pytest spadeBDI_RL_refactored/tests/test_worker_integration.py

# Run with verbose output
pytest spadeBDI_RL_refactored/tests/ -v -s

# Run with coverage
pytest spadeBDI_RL_refactored/tests/ --cov=spadeBDI_RL_refactored --cov-report=html
```

## Test Requirements

### Docker Dependencies
Some tests require ejabberd container:
```bash
cd spadeBDI_RL_refactored/infrastructure
docker-compose up -d
```

Tests will skip automatically if ejabberd is not running or healthy.

### Environment Variables
- `TELEMETRY_OUTPUT`: Optional path for telemetry JSONL output

## Test Categories

### 1. Docker Compose Validation
- Validates `docker-compose.yaml` structure
- Checks ejabberd service configuration (ports, environment, healthcheck)

### 2. Agent Lifecycle Tests
- Lazy loading verification
- Agent creation without XMPP
- Full lifecycle: create → start → stop
- XMPP connectivity tests

### 3. Worker Integration Tests
- Full training run (5 episodes)
- Config parsing from stdin
- Telemetry emission (JSONL format)
- Policy saving
- Clean exit verification

### 4. Headless Operation Tests
- Import validation without matplotlib
- Dry run with minimal episodes
- Package-level import analysis

## Adding New Tests

When adding functionality to `spadeBDI_RL_refactored`:
1. **Unit tests** for core logic (algorithms, adapters, core components)
2. **Integration tests** for worker workflows
3. **Mock ejabberd** when testing agent interactions offline
4. **Use subprocess** for full worker tests to ensure isolation
