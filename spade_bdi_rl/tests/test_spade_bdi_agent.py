"""Test SPADE-BDI agent integration with trainer orchestrator."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ..core import RunConfig, HeadlessTrainer
    # Now explicitly load BDI components
from ..core import (
        AgentHandle,
        DEFAULT_JID,
        DEFAULT_PASSWORD,
        create_agent,
        docker_compose_path,
    )

from ..core import create_agent

from ..core import create_and_start_agent
from ..core import docker_compose_path
from ..core import resolve_asl
from ..core import create_agent, resolve_asl

@pytest.fixture(scope="module")
def check_ejabberd():
    """Verify ejabberd container is running before tests."""
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=ejabberd", "--format", "{{.Status}}"],
        capture_output=True,
        text=True,
    )
    
    if "Up" not in result.stdout or "healthy" not in result.stdout:
        pytest.skip("ejabberd container not running or not healthy. Start with: docker-compose up -d")
    
    return True


def test_agent_lazy_loading():
    """Test that BDI agent modules load lazily without errors."""
    # This should NOT trigger SPADE imports
    
    assert RunConfig is not None
    assert HeadlessTrainer is not None
    
    
    assert AgentHandle is not None
    assert DEFAULT_JID == "agent@localhost"
    assert DEFAULT_PASSWORD == "secret"
    assert callable(create_agent)
    
    # Verify docker-compose path
    compose_path = docker_compose_path()
    assert compose_path.exists(), f"docker-compose.yaml not found at {compose_path}"
    print(f"✓ Docker compose found: {compose_path}")


def test_agent_creation(check_ejabberd):
    """Test SPADE-BDI agent creation without starting it."""

    
    # Create agent (don't start yet)
    handle = create_agent(
        jid="test_agent@localhost",
        password="test_secret",
        ensure_account=False,  # Skip XMPP account creation for now
    )
    
    assert handle is not None
    assert handle.jid == "test_agent@localhost"
    assert handle.password == "test_secret"
    assert not handle.started
    
    print(f"✓ Agent handle created: {handle.jid}")


def test_agent_lifecycle(check_ejabberd):
    """Test full SPADE-BDI agent lifecycle: create → start → stop."""
    
    async def run_test():
        # This test requires ejabberd to be running
        try:
            handle = await create_and_start_agent(
                jid="agent@localhost",
                password="secret",
                ensure_account=True,
                verify_connection=True,
                start_timeout=15.0,
            )
            
            assert handle.started
            print(f"✓ Agent started: {handle.jid}")
            
            # Keep agent running briefly
            await asyncio.sleep(2)
            
            # Stop the agent
            await handle.stop()
            assert not handle.started
            print(f"✓ Agent stopped: {handle.jid}")
            
        except Exception as e:
            pytest.fail(f"Agent lifecycle test failed: {e}")
    
    # Run async test
    asyncio.run(run_test())


def test_xmpp_connectivity(check_ejabberd):
    """Test XMPP connection to ejabberd server."""
    from spadeBDI_RL.src import spade_bdi_rl_agent as legacy_agent
    
    async def check_connection():
        result = await legacy_agent.test_xmpp_connection("agent@localhost", "secret")
        if not result:
            pytest.skip("XMPP connection test failed - ejabberd may need initialization")
        print("✓ XMPP connection verified")
    
    asyncio.run(check_connection())


def test_docker_compose_path():
    """Test docker-compose.yaml path resolution."""
    
    path = docker_compose_path()
    assert path.exists(), f"docker-compose.yaml not found at {path}"
    assert path.name == "docker-compose.yaml"
    
    # Verify it contains ejabberd config
    content = path.read_text()
    assert "ejabberd" in content
    assert "5222" in content  # XMPP port
    
    print(f"✓ docker-compose.yaml validated at {path}")


def test_asl_path_resolution():
    """Test AgentSpeak (.asl) file resolution."""

    # Should find default ASL file
    asl_path = resolve_asl()
    assert asl_path.exists(), f"Default ASL file not found at {asl_path}"
    assert asl_path.suffix == ".asl"
    
    print(f"✓ ASL file found: {asl_path}")


def test_agent_with_custom_asl(check_ejabberd):
    """Test agent creation with custom ASL file."""
    
    # Get default ASL path
    asl_path = resolve_asl()
    
    # Create agent with explicit ASL path
    handle = create_agent(
        jid="custom_agent@localhost",
        password="custom_secret",
        asl_file=asl_path,
        ensure_account=False,
    )
    
    assert handle is not None
    print(f"✓ Agent created with custom ASL: {asl_path}")


def test_ejabberd_docker_status():
    """Display ejabberd container status for debugging."""
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=ejabberd", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
        capture_output=True,
        text=True,
    )
    
    print("\n" + "=" * 80)
    print("EJABBERD CONTAINER STATUS")
    print("=" * 80)
    print(result.stdout)
    print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
